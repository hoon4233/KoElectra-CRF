import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    init_logger,
    set_seed,
    compute_metrics,
    show_ner_report
)

from model import KoelectraCRF

from processor import ner_load_and_cache_examples as load_and_cache_examples
from processor import ner_tasks_num_labels as tasks_num_labels
from processor import ner_processors as processors

logger = logging.getLogger(__name__)


def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):

    results = {}

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    electra_param_optimizer = list(model.electra.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.position_wise_ff.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in electra_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in electra_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            # print(f"IN Train, batch, len(batch) :{batch}, {len(batch)}") # len == 4, input_ids, attention_mask, token_type_ids, labels
            # print(f"IN Train, batch[3], len(batch[3]) :{batch[3]}, {len(batch[3])}") # batch_size 만큼 샘플, 설정한게 8이니까 len == 8

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "label_mask": batch[4],
                "crf_mask": batch[5]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            # print(f"IN Train, Outputs :{outputs}")
            # print(f"IN Train, len(outputs) : {len(outputs)}")
            # print(f"IN Train, outputs[0] : {outputs[0]}") # IN Train, outputs[0] : 3.5203754901885986, 즉 Loss 값 1개
            # print(f"IN Train, outputs[1] : {outputs[1]}")
            # print( f"IN Train, len(outputs[1]), len(outputs[1][0]), len(outputs[1][0][0]) :{len(outputs[1])}, {len(outputs[1][0])}, {len(outputs[1][0][0])}")

            # exit()

            loss = outputs['loss']

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tmp_loss = loss.item()
            tr_loss += tmp_loss
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(f"***** Running training ({global_step} step) *****")
                    logger.info(f"  Training loss = {tmp_loss}")
                    results[str(global_step)] = tmp_loss

                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))
            # print("IS ok ?")
            # exit()

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    output_eval_file = os.path.join(args.output_dir, "training_loss.txt")
    with open(output_eval_file, "w") as f_w:
        for key in sorted(results.keys()):
            f_w.write("{} = {}\n".format(key, str(results[key])))

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    label_mask = None

    # print(f"IN eval, len(eval_dataloader) : {len(eval_dataloader)}")
    # exit()

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "label_mask" : batch[4],
                "crf_mask" : batch[5]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss = outputs['loss']
            tags = outputs['y_pred']

            # print(f"tags : {tags}")
            # print(f"len(tags) : {len(tags)}")
            # print(f"len(tags[0]) : {len(tags[0])}")

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += tmp_eval_loss.item()
            # exit()

        nb_eval_steps += 1
        # print(f"preds : {logits}")
        # print(f"preds type : {type(logits)}")
        # exit()
        if preds is None:
            # preds = logits.detach().cpu().numpy()
            # preds = np.array(tags, dtype=object)
            preds = tags
            # print(f"type(preds), type(preds) : {type(preds), type(preds[0])}")
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            label_mask = inputs["label_mask"].detach().cpu().numpy()

        else:
            # preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            for tag in tags :
                preds.append(tag)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            label_mask = np.append(label_mask, inputs["label_mask"].detach().cpu().numpy(), axis=0)

        # break

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }

    # print(f"preds : {preds}")
    # print(f"len(preds) : {len(preds)}")
    # print(f"out_label_ids : {out_label_ids}")
    # print(f"out_label_ids.shape : {out_label_ids.shape}")
    # print(f"label_mask : {label_mask}")
    # print(f"label_mask.shape : {label_mask.shape}")
    # exit()

    labels = processors[args.task](args).get_labels()

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    # pad_token_label_id = CrossEntropyLoss().ignore_index
    # pad_token_label_id, cls_token_label_id, sep_token_label_id = 0,0,0
    # unused_token_ids = (pad_token_label_id, cls_token_label_id, sep_token_label_id)

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if label_mask[i,j] == 1:
                out_label_list[i].append(label_map[out_label_ids[i][j]])

    for seq_idx, pred in enumerate(preds) :
        for j in pred :
            preds_list[seq_idx].append(label_map[j])

    # print(f"out_label_list : {out_label_list}")
    # print(f"pred_list : {preds_list}")
    # print(f"len(out_label_list) : {len(out_label_list)}")
    # print(f"len(pred_list) : {len(preds_list)}")
    # exit()


    result = compute_metrics(args.task, out_label_list, preds_list)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list))  # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))

    # exit()

    return results


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = processors[args.task](args)
    labels = processor.get_labels()
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=tasks_num_labels[args.task],
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = KoelectraCRF.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    ## 일단 실행 안 되게 하려고 if not (원래는 if)
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = KoelectraCRF.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    ## GPU 할당 변경하기
    GPU_NUM = 1  # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU

    print('Current cuda device ', torch.cuda.current_device())  # check

    # Additional Infos
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(GPU_NUM) / 1024 ** 3, 1), 'GB')

    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)

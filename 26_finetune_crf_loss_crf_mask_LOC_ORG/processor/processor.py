import os
import copy
import json
import logging

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids, label_mask, crf_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.crf_mask = crf_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def ner_convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_seq_length,
        task,
        pad_token_label_id=0,
        bos_token_label_id=0,
        eos_token_label_id=0,
):
    label_lst = ner_processors[task](args).get_labels()
    label_map = {label: i for i, label in enumerate(label_lst)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example {} of {}".format(ex_index, len(examples)))

        # tokens = tokenizer.tokenize(example.words)
        # label_ids = [label_map[x] if x in label_map else label_map["O"] for x in example.labels]
        tokens = []
        label_ids = []
        label_mask = []
        crf_mask = []

        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label.endswith("-B"):
                label_ids.extend([label_map[label]] + [label_map[label] + 1] * (len(word_tokens) - 1))
            else :
                label_ids.extend([label_map[label]] + [label_map[label]] * (len(word_tokens) - 1))

            # label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            label_mask.extend([1] + [0] * (len(word_tokens) - 1))
            crf_mask.extend([1] + [0] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            label_mask = label_mask[:(max_seq_length - special_tokens_count)]
            crf_mask = crf_mask[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        # label_ids += [pad_token_label_id]
        label_ids += [eos_token_label_id]
        # label_mask += [1]
            # 실제로 사용할 때 결과로 나온 토큰들 중 맨 처음, 마지막 안 쓰면 되니까 성능 측정 안 함
            # 로스 계산 할 땐 들어감
        label_mask += [0]
        crf_mask += [1]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        # label_ids = [pad_token_label_id] + label_ids
        label_ids = [bos_token_label_id] + label_ids
        # label_mask = [1] + label_mask
        label_mask = [0] + label_mask
        crf_mask = [1] + crf_mask

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        label_mask += [pad_token_label_id] * padding_length
        crf_mask += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(crf_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))
            logger.info("label_mask: %s " % " ".join([str(x) for x in label_mask]))
            logger.info("crf_mask: %s " % " ".join([str(x) for x in crf_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids,
                          label_mask=label_mask,
                          crf_mask=crf_mask)
        )

    return features


class NaverNerProcessor(object):
    """Processor for the Naver NER data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        return ["O", "ORG-B", "ORG-I", "LOC-B", "LOC-I"]
                # "CLS", "SEP"]

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            words, labels = data.split('\t')
            words = words.split()
            labels = labels.split()
            guid = "%s-%s" % (set_type, i)

            assert len(words) == len(labels)

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, labels=labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir,
                                                        self.args.task,
                                                        file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir,
                                                                  self.args.task,
                                                                  file_to_read)), mode)


ner_processors = {
    "naver-ner": NaverNerProcessor
}

ner_tasks_num_labels = {
    "naver-ner": 29
    # "naver-ner": 31
}


def ner_load_and_cache_examples(args, tokenizer, mode):
    processor = ner_processors[args.task](args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")

        # pad_token_label_id = CrossEntropyLoss().ignore_index
        # features = ner_convert_examples_to_features(
        #     args,
        #     examples,
        #     tokenizer,
        #     max_seq_length=args.max_seq_len,
        #     task=args.task,
        #     pad_token_label_id=pad_token_label_id
        # )
        features = ner_convert_examples_to_features(
            args,
            examples,
            tokenizer,
            max_seq_length=args.max_seq_len,
            task=args.task
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)
    all_crf_mask = torch.tensor([f.crf_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids, all_label_mask, all_crf_mask)
    return dataset

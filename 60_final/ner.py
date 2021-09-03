import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import ElectraTokenizer
import konlpy

from src import set_seed
from processor import NerProcessor, make_examples
from model import KoelectraCRF


SEED = 42
set_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'ckpt/koelectra-base-v3-naver-ner-ckpt/checkpoint-60000'

tokenizer = ElectraTokenizer.from_pretrained(
    "monologg/koelectra-base-v3-discriminator",
    do_lower_case=False
)
model = KoelectraCRF.from_pretrained(MODEL_PATH)
model.to(DEVICE)

Okt = konlpy.tag.Okt()


def del_josa(word):
    global Okt
    filtered_word = word.replace('.', '').replace(',', '').replace("'", "").replace('·', ' ').replace('=', '').replace('\n', '')
    Okt_morphs = Okt.pos(filtered_word)
    return Okt_morphs


def NER(text):
    global tokenizer, model
    processor = NerProcessor(text)
    labels = processor.get_labels()

    # dataset
    str_datset = make_examples(tokenizer, text)
    str_sampler = SequentialSampler(str_datset)
    str_dataloader = DataLoader(str_datset, sampler=str_sampler, batch_size=len(text))

    preds = None
    label_mask = None

    for batch in str_dataloader:
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "crf_mask": batch[2]
            }
            outputs = model(**inputs)
            tags = outputs['y_pred']

        preds = tags
        label_mask = inputs["crf_mask"].detach().cpu().numpy()


    text2words = processor.get_examples()
    label_map = {i: label for i, label in enumerate(labels)}
    result = set()

    for sen_idx, mask in enumerate(label_mask):
        word_idx = 0
        len_sen = len(text2words[sen_idx].words)
        for token_idx, TF in enumerate(mask):
            if TF and word_idx < len_sen:
                if label_map[preds[sen_idx][word_idx]].startswith('LOC'):
                    word2tok = del_josa(text2words[sen_idx].words[word_idx])
                    word = ""
                    len_word2tok = len(word2tok)
                    for i in range(len_word2tok):
                        if i == len_word2tok-1 and word2tok[i][1] == 'Josa':
                            break
                        word+=word2tok[i][0]
                    result.add(word)
                word_idx += 1

    print(result)
    return result


def test_tokenizer(text):
    tokenizer = ElectraTokenizer.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        do_lower_case=False
    )
    tmp_string = []
    for sentence in text:
        tmp_string.append(sentence.split())
    tokens = []
    for sen in tmp_string:
        for word in sen:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
    print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    for idx in range(len(input_ids)):
        print(tokenizer.convert_ids_to_tokens(int(input_ids[idx])))


import time

if __name__ == '__main__':
    text = ["오늘도 전국이 흐린 가운데 곳곳에 비가 내리겠습니다.",
             "지금 호남지역을 중심으로 머무르고 있는 비구름대는 차츰 충청이남지방까지 확대되겠고요.",
             "특히 오늘 오후부터 내일 새벽 사이 호남 해안과 제주도를 중심으로 벼락과 돌풍을 동반해 80mm가 넘는 많고강한 비가 내리겠습니다.",
            "수도권과 강원 남부도 낮동안 곳에 따라 가끔 약한 비가 내리거나 빗방울이 떨어지는 곳이 있겠습니다.  ",
            "오늘까지는 낮에도 대체로 25도 안팎까지 오르는데 그칠텐데요.",
             ]
    # test_tokenizer(text)
    N = 5
    start_time = time.time()
    for _ in range(N):
        ret = NER(text)
    print(f"Execution time : {time.time() - start_time}")

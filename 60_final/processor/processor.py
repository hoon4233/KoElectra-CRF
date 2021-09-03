import copy
import json

import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, words):
        self.words = words

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

    def __init__(self, input_ids, attention_mask, crf_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
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


class NerProcessor(object):
    def __init__(self, text):
        self.text = text

    def get_labels(self):
        return ["O",
                "PER-B", "PER-I", "FLD-B", "FLD-I", "AFW-B", "AFW-I", "ORG-B", "ORG-I",
                "LOC-B", "LOC-I", "CVL-B", "CVL-I", "DAT-B", "DAT-I", "TIM-B", "TIM-I",
                "NUM-B", "NUM-I", "EVT-B", "EVT-I", "ANM-B", "ANM-I", "PLT-B", "PLT-I",
                "MAT-B", "MAT-I", "TRM-B", "TRM-I"]

    @classmethod
    def _strip(cls, input_text):
        for i in range(len(input_text)):
            input_text[i] = input_text[i].strip()
        return input_text

    def _create_examples(self, text):
        examples = []
        for (i, sentence) in enumerate(text):
            words = sentence.split()
            examples.append(InputExample(words=words))
        return examples

    def get_examples(self):
        return self._create_examples(self._strip(self.text))


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    for (ex_idx, example) in enumerate(examples):
        tokens = []
        crf_mask = []

        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            crf_mask.extend([True] + [False] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            crf_mask = crf_mask[:(max_seq_length - special_tokens_count)]

        # Add [SEP]
        tokens += [tokenizer.sep_token]
        crf_mask += [False]

        # Add [CLS]
        tokens = [tokenizer.cls_token] + tokens
        crf_mask = [False] + crf_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        crf_mask += [False] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(crf_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          crf_mask=crf_mask)
        )

    return features


def make_examples(tokenizer, text):
    MAX_SEQ_LEN = 128
    processor = NerProcessor(text)
    examples = processor.get_examples()
    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_seq_length=MAX_SEQ_LEN
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_crf_mask = torch.tensor([f.crf_mask for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_crf_mask)
    return dataset
import random
import logging

import torch
import numpy as np

from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ElectraForTokenClassification,
)

CONFIG_CLASSES = {
    "koelectra-base-v3": ElectraConfig,
}

TOKENIZER_CLASSES = {
    "koelectra-base-v3": ElectraTokenizer,
}


MODEL_FOR_TOKEN_CLASSIFICATION = {
    "koelectra-base-v3": ElectraForTokenClassification,
}


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
            # "precision": seqeval_metrics.precision_score(labels, preds),
            # "recall": seqeval_metrics.recall_score(labels, preds),
            # "f1": seqeval_metrics.f1_score(labels, preds),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)

    if task_name == "naver-ner":
        return f1_pre_rec(labels, preds, is_ner=True)
    else:
        raise KeyError(task_name)

import csv
import os
from collections import defaultdict

EVAL_FILES_PATH = './eval_results/'
TRAINING_LOSS_FILES_PATH = './training_loss/'

def compare(file_num):
    global EVAL_FILES_PATH, TRAINING_LOSS_FILES_PATH

    HEADER = ["Step", "Tr_loss", "test_loss", "f-1", "precision", "recall"]
    EVAL_FILE = 'eval_results_' + file_num + '.txt'
    RESULT_PATH = f'./result/result_{file_num}.tsv'
    result = defaultdict(lambda : [0, 0, 0, 0, 0]) # tr_loss, test_loss, f-1, precision, recall

    if not file_num.startswith('base'):
        LOSS_FILE = 'training_loss_' + file_num + '.txt'
        with open(TRAINING_LOSS_FILES_PATH+LOSS_FILE, 'r') as f :
            lines = f.readlines()
            for line in lines:
                step, _, loss = line.split()
                result[int(step)][0] = loss

    with open(EVAL_FILES_PATH+EVAL_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines :
            name, _, val = line.split()
            item, step = name.split('_')
            step = int(step)

            if item == 'loss':
                result[step][1] = val

            elif item == 'f1':
                result[step][2] = val

            elif item == 'precision':
                result[step][3] = val

            elif item == 'recall':
                result[step][4] = val

            else:
                print(item)
                raise KeyError

    with open(RESULT_PATH, 'w') as f:
        wr = csv.writer(f, delimiter = '\t')
        wr.writerow(HEADER)
        for step in sorted(result.keys()):
            wr.writerow([str(step)]+result[step])


def main():
    global EVAL_FILES_PATH
    LEN_SUFFIX, LEN_POSTFIX = 13, -4
    START_NUM, END_NUM = 58, 59
    cases = sorted([case[LEN_SUFFIX:LEN_POSTFIX] for case in os.listdir(EVAL_FILES_PATH) if case.endswith('.txt')])
    base_flag = True

    for case in cases:
        if base_flag:
            if case.startswith('base'):
                compare(case)
        else:
            if case.isnumeric() and (START_NUM<=int(case)<=END_NUM):
                compare(case)


if __name__=='__main__':
    main()
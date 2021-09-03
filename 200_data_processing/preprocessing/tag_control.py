import csv
from copy import deepcopy

TAGS = ["O", "LOC-B", "LOC-I"]
# ORI_FILE_PATH = 'train.tsv'
# RESULT_FILE_PATH = 'train.tsv'
# ORI_FILE_PATH = 'test.tsv'
# RESULT_FILE_PATH = 'new_test.tsv'

def tag_control(ORI_FILE_PATH, RESULT_FILE_PATH):
    global TAGS
    with open(ORI_FILE_PATH, 'r') as f :
        LINES = f.readlines()

    NEW_LINES = []
    count = 0
    all_tag0_count = len(LINES)

    for idx, line in enumerate(LINES) :
        sentence, tags = line.split('\t')

        ori_tags = deepcopy(tags)
        tags = tags.split()

        for idx, tag in enumerate(tags) :
            if tag not in TAGS :
                tags[idx] = TAGS[0]

        for tag in tags :
            if tag != TAGS[0] :
                all_tag0_count -= 1
                break

        tags[-1] = tags[-1]+'\n'
        tags = " ".join(tags)

        if ori_tags[-1] != tags[-1] :
            print(f"ori : {ori_tags}, tags : {tags}")

        new_line = sentence + '\t' + tags
        NEW_LINES.append(new_line)

        if len(line.split('\t')) == 2 :
            count += 1

    print(f"ALL count : {count}")
    print(f"Not O tag : {count-all_tag0_count}, all O tag : {all_tag0_count}")

    with open(RESULT_FILE_PATH, 'w') as f :
        f.writelines(NEW_LINES)

tag_control('../data/train.tsv', 'new_train.tsv')
tag_control('../data/test.tsv', 'new_test.tsv')
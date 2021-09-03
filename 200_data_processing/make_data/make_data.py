import os
import re
FILE_PATH = '../data/kmounlp/split_txt/'
STOP_POINT = 9000
BASE_NON = 1000

def make_dataset(file_path):
    global STOP_POINT, BASE_NON
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith('txt')]
    num_files = len(files)
    print(f'num of files : {num_files}')

    sentences = []
    tags = []
    LOC_tag_count = 0

    B_tag = "LOC-B"
    I_tag = 'LOC-I'
    None_tag = 'O'

    p = re.compile('[\S]*?:LOC>[\S]*?')
    start_p = re.compile('[\W]*?<')

    for file_idx, file in enumerate(sorted(files)):
        with open(file_path+file, 'r') as f :
            lines = f.readlines()
            check = 0

            for line_idx, line in enumerate(lines):
                line = line.strip()

                if line and ord(line[0]) == 65279: # handle BOM (byte order mark)
                    line = line.replace(line[0], '')

                if not line.startswith('## '):
                    continue

                if check == 0:  # ## 1
                    check += 1

                elif check == 1: # ## 스포츠동아 김민정 기자 ricky337@donga.com
                    sentences.append(line[3:])
                    check += 1

                elif check == 2: # ## <스포츠동아:ORG> <김민정:PER> 기자 <ricky337@donga.com:POH>
                    line = line[3:].split()
                    tmp_tags = []
                    for word_idx, word in enumerate(line):
                        if p.match(word):
                            if start_p.match(word):
                            # if word.startswith('<'):
                                tmp_tags.append(B_tag)
                            else:
                                for i in range(len(tmp_tags)-1, -1, -1):
                                    if start_p.match(line[i]):
                                    # if line[i].startswith('<'):
                                        tmp_tags[i] = B_tag
                                        break
                                    else:
                                        tmp_tags[i] = I_tag
                                tmp_tags.append(I_tag)

                            LOC_tag_count += 1

                        else:
                            tmp_tags.append(None_tag)

                    tags.append(tmp_tags)
                    check = 0

                    check_sentence_len = len(sentences[-1].split())
                    check_tag_len = len(tags[-1])
                    if check_sentence_len != check_tag_len:
                        print(f"file_name : {file}, file_idx : {file_idx}, line_idx : {line_idx}")
                        print(f"len_sentence : {check_sentence_len}, len_tag : {check_tag_len}")
                        print(f"sentence : [{sentences[-1]}], tag : {tags[-1]}")
                        raise RuntimeError

                else:
                    print(f"Wrong sentence : {line}")
                    raise RuntimeError


    num_sentences = len(sentences)
    num_tags = len(tags)
    print(f"num_setences, num_tags : {num_sentences}, {num_tags}")
    print(f"LOC_tag_count : {LOC_tag_count}")

    if num_sentences != num_tags :
        print(f"Error, num_sentences != num_tags")
        raise RuntimeError

    all_zero_tag_strings = 0
    non_zero_tag_strings = 0
    base_non_zero = 0

    with open('result.tsv', 'w') as f :
        for idx, (sentence, tag) in enumerate(zip(sentences, tags)) :
            if idx == STOP_POINT:
                break

            non_flag = False
            for t in tag:
                if t != None_tag:
                    non_flag = True
                    break

            if base_non_zero < BASE_NON:
                if non_flag:
                    base_non_zero += 1
                    non_zero_tag_strings += 1
                    row = sentence + '\t' + " ".join(tag) + '\n'
                    f.write(row)
                STOP_POINT += 1

            else:
                if non_flag:
                    non_zero_tag_strings += 1
                else:
                    all_zero_tag_strings += 1

                row = sentence+'\t'+" ".join(tag)+'\n'
                f.write(row)

    print(f"all_zero_tag_strings : {all_zero_tag_strings}")
    print(f"non_zero_tag_strings : {non_zero_tag_strings}")

make_dataset(FILE_PATH)
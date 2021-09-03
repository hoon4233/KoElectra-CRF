import csv
import os
import json
import pickle

START_NUM = 0
END_NUM = 59
EXCEPTION_LIST = [i for i in range(1, 10, 1)] + [27, 28] + [i for i in range(36, 41, 1)] + [56, 100]


def get_dir_path(target_path, target_dir):
    SMALL_CASE = [i for i in range(19,23,1)]
    if target_dir == "config" or target_dir == "data":
        path = os.path.join(target_path, target_dir, 'naver-ner')
    elif target_dir == "ckpt":
        case_num = int(target_path.split('_')[0])
        if case_num in SMALL_CASE:
            path = os.path.join(target_path, target_dir, 'koelectra-small-v3-naver-ner-ckpt')
        else:
            path = os.path.join(target_path, target_dir, 'koelectra-base-v3-naver-ner-ckpt')
    else:
        print("IN get_dir_path, target_dir is wrong")
        raise NameError

    return path


def rename_ori_data_and_put_new_data(target_path):
    VISIT_FILE_NAME = "visit.pickle"

    # Load visit
    if os.path.exists(VISIT_FILE_NAME):
        with open(VISIT_FILE_NAME, 'rb') as f:
            VISIT = pickle.load(f)
    else:
        VISIT = dict()

    # check visit and update visit
    if VISIT.get(target_path):
        return
    else:
        VISIT[target_path] = True

    # rename
    ckpt_path = get_dir_path(target_path, 'ckpt')
    os.chdir(ckpt_path)
    os.system("mv test ori_test")
    os.system("mv eval_results.txt ori_eval_results.txt")
    os.chdir("../../../")

    # put new data
    data_path = get_dir_path(target_path, 'data')
    os.system(f"cp new_test.tsv ./{data_path}/new_test.tsv")

    # save visit
    with open(VISIT_FILE_NAME, 'wb') as f:
        pickle.dump(VISIT, f)

def del_cache_file(path, target, num_cd):
    os.chdir(path)
    print(os.listdir())
    cache_files = [file for file in os.listdir() if file.startswith(target)]
    print(cache_files)
    for file in cache_files:
        os.system(f"rm -rf {file}")
    for _ in range(num_cd):
        os.chdir("..")

def del_cache(target_path):
    # model cache
    del_cache_file(target_path, "__pycache__", 1)

    # data cache
    data_cache_path = os.path.join(target_path, "data")
    del_cache_file(data_cache_path, "cached", 2)

    # processor cache
    processor_cache_path = os.path.join(target_path, "processor")
    del_cache_file(processor_cache_path, "__pycache__", 2)

    # src cache
    src_cache_path = os.path.join(target_path, "src")
    del_cache_file(src_cache_path, "__pycache__", 2)


def eval(target_path):
    # make new config file
    config_path = get_dir_path(target_path, 'config')
    config_file = [file for file in os.listdir(config_path) if file.endswith('.json')]
    config_file = config_file[0]
    config_file = os.path.join(config_path, config_file)
    new_config_file = os.path.join(config_path, 'new_config.json')

    with open(config_file, 'r') as jf:
        config = json.load(jf)

    config["test_file"] = "new_test.tsv"
    config["do_train"] = False

    with open(new_config_file, 'w') as jf :
        json.dump(config, jf, indent=2)

    # chdir & run model to eval & chdir
    os.chdir(target_path)
    os.system("python3 run_ner.py --task naver-ner --config_file new_config.json")
    os.chdir("../")


def com_per(target_path):
    HEADER = ["name", "precision", "recall", "f1-score", "support"]
    COM_RESULT_PATH = "./per_com_result.tsv"

    eval_result_path = os.path.join(get_dir_path(target_path, 'ckpt'), "test")
    eval_result_files = sorted([file for file in os.listdir(eval_result_path) if file.endswith('txt')])
    result = dict()
    LEN_SUFFIX, LEN_POSTFIX = 5, -4

    for file in eval_result_files:
        file_path = os.path.join(eval_result_path, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('LOC'):
                    line = line.split()
                    result[target_path+'_Step_'+file[LEN_SUFFIX:LEN_POSTFIX]] = line[1:]

    if os.path.isfile(COM_RESULT_PATH):
        with open(COM_RESULT_PATH, 'a') as f:
            wr = csv.writer(f, delimiter='\t')
            for case in sorted(result.keys()):
                wr.writerow([case]+result[case])

    else:
        with open(COM_RESULT_PATH, 'w') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(HEADER)
            for case in sorted(result.keys()):
                wr.writerow([case]+result[case])


def change_result(result, tmp_result, check_point):
    result[0] = check_point
    for i in range(1, 5, 1):
        result[i] = tmp_result[i]


def com_ori_loc_per(target_path):
    HEADER = ["name", "precision", "recall", "f1-score", "support"]
    COM_RESULT_PATH = "./per_ori_com_result.tsv"

    eval_result_path = os.path.join(get_dir_path(target_path, 'ckpt'), "ori_test")
    eval_result_files = sorted([file for file in os.listdir(eval_result_path) if file.endswith('txt')])
    result = dict()
    LEN_SUFFIX, LEN_POSTFIX = 5, -4

    INF = 1e9
    tmp_result = [INF, 0, 0, 0, 0] # checkpoint, precision, recall, f1, support
    for file in eval_result_files:
        file_path = os.path.join(eval_result_path, file)
        check_point = int(file[LEN_SUFFIX:LEN_POSTFIX])
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('LOC'):
                    line = line.split()
                    line = [line[0]] + list(map(float, line[1:]))
                    if tmp_result[3] < line[3]: # compare f1
                        change_result(tmp_result, line, check_point)
                    elif tmp_result[3] == line[3]:
                        if tmp_result[1] < line[1]: # compare precison
                            change_result(tmp_result, line, check_point)
                        elif tmp_result[1] == line[1]:
                            if tmp_result[2] < line[2]: # compare recall
                                change_result(tmp_result, line, check_point)
                            elif tmp_result[2] == line[2]:
                                if tmp_result[0] > check_point:
                                    change_result(tmp_result, line, check_point)

    result[target_path+'_Step_'+str(tmp_result[0])] = list(map(str, tmp_result[1:]))
    # print(result)
    # exit()

    if os.path.isfile(COM_RESULT_PATH):
        with open(COM_RESULT_PATH, 'a') as f:
            wr = csv.writer(f, delimiter='\t')
            for case in sorted(result.keys()):
                wr.writerow([case]+result[case])

    else:
        with open(COM_RESULT_PATH, 'w') as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerow(HEADER)
            for case in sorted(result.keys()):
                wr.writerow([case]+result[case])


def main():
    global START_NUM, END_NUM, EXCEPTION_LIST
    case_dirs = sorted([dir for dir in os.listdir() if os.path.isdir(dir) and not dir.startswith('result_') and dir[:2].isnumeric()])

    for dir in case_dirs:
        case_num = int(dir.split('_')[0])
        if START_NUM<=case_num<=END_NUM and case_num not in EXCEPTION_LIST:
            print(f"Start case : {dir}")
            # rename_ori_data_and_put_new_data(dir)
            # del_cache(dir)
            # eval(dir)
            # com_per(dir)
            # com_ori_loc_per(dir)

def del_gitignore():
    case_dirs = sorted(
        [dir for dir in os.listdir() if os.path.isdir(dir) and not dir.startswith('result_') and dir[:2].isnumeric()])
    for case in case_dirs:
        os.chdir(case)
        os.system(f"rm -rf .gitignore")
        os.chdir('..')






if __name__ == '__main__':
    del_gitignore()
    # main()








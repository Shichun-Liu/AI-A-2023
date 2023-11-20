import numpy as np
from tqdm import tqdm

from utils_part1 import *
from check_part1 import check

# 超参数
state_number = 33
tag2idx = {
    "O": 0,
    "B-NAME": 1,
    "M-NAME": 2,
    "E-NAME": 3,
    "S-NAME": 4,
    "B-CONT": 5,
    "M-CONT": 6,
    "E-CONT": 7,
    "S-CONT": 8,
    "B-EDU": 9,
    "M-EDU": 10,
    "E-EDU": 11,
    "S-EDU": 12,
    "B-TITLE": 13,
    "M-TITLE": 14,
    "E-TITLE": 15,
    "S-TITLE": 16,
    "B-ORG": 17,
    "M-ORG": 18,
    "E-ORG": 19,
    "S-ORG": 20,
    "B-RACE": 21,
    "M-RACE": 22,
    "E-RACE": 23,
    "S-RACE": 24,
    "B-PRO": 25,
    "M-PRO": 26,
    "E-PRO": 27,
    "S-PRO": 28,
    "B-LOC": 29,
    "M-LOC": 30,
    "E-LOC": 31,
    "S-LOC": 32,
}

def main():
    data, _, _ = DataProcess("../NER/Chinese/train.txt")
    character_number = len(vocab(data))
    idx2tag = {}
    for key, value in tag2idx.items(): idx2tag[value] = key

    # 从汉字找到编号、从编号找到汉字
    char2idx = {}
    idx2char = {}
    line = vocab(data)
    for i, v in enumerate(line):
        char2idx[v] = i
        idx2char[i] = v

    print("开始训练")
    # A是状态转移矩阵，B是观测概率矩阵，pi是初始状态矩阵
    A = np.zeros((state_number, state_number), dtype=np.float64)
    B = np.zeros((state_number, character_number + 1), dtype=np.float64)  # 加1用于记录未出现的字
    pi = np.zeros(state_number, dtype=np.float64)
    A, B, pi = train(data, A, B, pi, tag2idx, char2idx)
    print("训练完毕")

    save_parameter(
        A_path="./weights/A.txt",
        B_path="./weights/B.txt",
        pi_path="./weights/pi.txt",
        A=A, B=B, pi=pi
    )
    A, B, pi = load_parameter(
        A_path="./weights/A.txt", 
        B_path="./weights/B.txt", 
        pi_path="./weights/pi.txt"
    )

    print("开始预测")
    _, valid_sentence, _ = DataProcess("../NER/Chinese/validation.txt")
    predict_tag = []
    for s in valid_sentence:
        p = viterbi(A, B, pi, s, word2idx=char2idx, idx2tag=idx2tag)
        assert len(p) == len(s), f"预测的状态序列长度与观测矩阵不同,len(p)={len(p)},len(s)={len(s)}"
        predict_tag.append([s, p])

    data2txt(predict_tag, "./my_Chinese_validation_result.txt")
    print("预测结束")
    check(
        language="Chinese",
        gold_path="../NER/Chinese/validation.txt",
        my_path="./my_Chinese_validation_result.txt",
    )


if __name__ == "__main__":
    main()
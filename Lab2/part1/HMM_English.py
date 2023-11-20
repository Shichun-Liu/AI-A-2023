import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

from utils_part1 import *
from check_part1 import *

# 超参数
state_number = 9
tag2idx = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}

if __name__ == "__main__":
    data, _, _ = DataProcess("../NER/English/train.txt")
    word_number = len(vocab(data))
    idx2tag = {}
    for key, value in tag2idx.items(): idx2tag[value] = key

    # 从单词找到编号、从编号找到单词
    word2idx = {}
    idx2word = {}
    line = vocab(data)
    for i, v in enumerate(line):
        word2idx[v] = i
        idx2word[i] = v
    
    print("开始训练")
    # A是状态转移矩阵，B是观测概率矩阵，pi是初始状态矩阵
    A = np.zeros((state_number, state_number), dtype=np.float64)
    B = np.zeros((state_number, word_number + 1), dtype=np.float64)  # 加1用于记录未出现的字
    pi = np.zeros(state_number, dtype=np.float64)
    A, B, pi = train(data, A, B, pi, tag2idx, word2idx)
    print("训练完毕")

    save_parameter(
        A_path="./weights/Eng-A.txt",
        B_path="./weights/Eng-B.txt",
        pi_path="./weights/Eng-pi.txt",
        A=A, B=B, pi=pi
    )
    A, B, pi = load_parameter(
        A_path="./weights/Eng-A.txt", 
        B_path="./weights/Eng-B.txt", 
        pi_path="./weights/Eng-pi.txt"
    )

    print("开始预测")

    _, valid_sentence, _ = DataProcess("../NER/English/validation.txt")

    predict_tag = []
    for s in valid_sentence:
        p = viterbi(A, B, pi, s, word2idx=word2idx, idx2tag=idx2tag)
        assert len(p) == len(s), f"预测的状态序列长度与观测矩阵不同,len(p)={len(p)},len(s)={len(s)}"
        predict_tag.append([s, p])

    data2txt(predict_tag, "./my_English_validation_result.txt")
    print("预测结束")

    print("英文结果: ")
    check(
        language="English",
        gold_path="../NER/English/validation.txt",
        my_path="./my_English_validation_result.txt",
    )

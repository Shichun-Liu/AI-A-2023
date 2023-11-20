import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

from part1_utils import *
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
    for key, value in tag2idx.items():
        idx2tag[value] = key

    # 从单词找到编号、从编号找到单词
    word2idx = {}
    idx2word = {}
    l = vocab(data)
    for i, v in enumerate(l):
        word2idx[v] = i
        idx2word[i] = v

    # A是状态转移矩阵，B是观测概率矩阵，pi是初始状态矩阵
    A = np.zeros((state_number, state_number))
    A = A.astype(np.float64)
    B = np.zeros((state_number, word_number + 1))  # 加1用于记录未出现的字
    B = B.astype(np.float64)
    pi = np.zeros(state_number)
    pi = pi.astype(np.float64)

    print("开始训练数据：")
    for i in tqdm(range(len(data))):  # 几组数据
        for j in range(len(data[i][0])):  # 每组数据中几个单词
            cur_word = data[i][0][j]  # 取出当前单词
            cur_tag = data[i][1][j]  # 取出当前标签
            B[tag2idx[cur_tag]][word2idx[cur_word]] += 1  # 对B矩阵中标签->单词的位置加一
            if j == 0:
                # 若是文本段的第一个单词，统计pi矩阵
                pi[tag2idx[cur_tag]] += 1
                continue
            pre_tag = data[i][1][j - 1]  # 记录前一个单词的标签
            # 对A矩阵中前一个标签->当前标签的位置加一
            A[tag2idx[pre_tag]][tag2idx[cur_tag]] += 1

    A[A == 0] = 1e-8
    A = np.log(A)
    B[B == 0] = 1e-8
    B = np.log(B)
    pi[pi == 0] = 1e-8
    pi = np.log(pi)
    print("训练完毕！")

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

    print("开始预测！")

    _, valid_sentence, _ = DataProcess("../NER/English/validation.txt")
    print(valid_sentence[-1])

    predict_tag = []
    for s in valid_sentence:
        p = viterbi(A, B, pi, s, word2idx=word2idx, idx2tag=idx2tag)
        assert len(p) == len(s), "预测的状态序列长度与观测矩阵不同,len(p)={},len(s)={}".format(
            len(p), len(s)
        )
        predict_tag.append([s, p])

    data2txt(predict_tag, "./my_English_validation_result.txt")
    print("预测结束！")

    print("英文结果: ")
    check(
        language="English",
        gold_path="../NER/English/validation.txt",
        my_path="./my_English_validation_result.txt",
    )

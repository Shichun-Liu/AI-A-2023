import numpy as np
from tqdm import tqdm

from part1_utils import *
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
    for key, value in tag2idx.items():
        idx2tag[value] = key

    # 从汉字找到编号、从编号找到汉字
    char2idx = {}
    idx2char = {}
    l = vocab(data)
    for i, v in enumerate(l):
        char2idx[v] = i
        idx2char[i] = v

    # A是状态转移矩阵，B是观测概率矩阵，pi是初始状态矩阵
    A = np.zeros((state_number, state_number))
    A = A.astype(np.float64)
    B = np.zeros((state_number, character_number + 1))  # 加1用于记录未出现的字
    B = B.astype(np.float64)
    pi = np.zeros(state_number)
    pi = pi.astype(np.float64)

    print("开始训练数据：")
    for i in tqdm(range(len(data))):  # 几组数据
        for j in range(len(data[i][0])):  # 每组数据中几个字符
            cur_char = data[i][0][j]  # 取出当前字符
            cur_tag = data[i][1][j]  # 取出当前标签
            B[tag2idx[cur_tag]][char2idx[cur_char]] += 1  # 对B矩阵中标签->字符的位置加一
            if j == 0:
                # 若是文本段的第一个字符，统计pi矩阵
                pi[tag2idx[cur_tag]] += 1
                continue
            pre_tag = data[i][1][j - 1]  # 记录前一个字符的标签
            # 对A矩阵中前一个标签->当前标签的位置加一
            A[tag2idx[pre_tag]][tag2idx[cur_tag]] += 1

    # 先将0加上一个1e-8，再取对数

    A[A == 0] = 1e-8
    A = np.log(A) - np.log(np.sum(A, axis=1, keepdims=True))
    B[B == 0] = 1e-8
    B = np.log(B) - np.log(np.sum(B, axis=1, keepdims=True))
    pi[pi == 0] = 1e-8
    pi = np.log(pi) - np.log(np.sum(pi))

    print("训练完毕！")
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
    print("开始预测！")

    _, valid_sentence, _ = DataProcess("../NER/Chinese/validation.txt")

    predict_tag = []
    for s in valid_sentence:
        p = viterbi(A, B, pi, s, word2idx=char2idx, idx2tag=idx2tag)
        assert len(p) == len(s), "预测的状态序列长度与观测矩阵不同,len(p)={},len(s)={}".format(
            len(p), len(s)
        )
        predict_tag.append([s, p])

    data2txt(predict_tag, "./my_Chinese_validation_result.txt")
    print("预测结束！")
    check(
        language="Chinese",
        gold_path="../NER/Chinese/validation.txt",
        my_path="./my_Chinese_validation_result.txt",
    )


if __name__ == "__main__":
    main()
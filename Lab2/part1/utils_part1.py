import numpy as np
from tqdm import tqdm

# 数据预处理
def DataProcess(path: str) -> (list, list, list):
    data = []
    sentence = []
    tag = []
    s = []
    t = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line != "\n":
                line = line.rstrip().split()
                sentence.append(line[0])
                tag.append(line[1])
            else:
                data.append([sentence, tag])
                s.append(sentence)
                t.append(tag)
                sentence = []
                tag = []
    return data, s, t


# 构造词表
def vocab(data: list) -> list:
    vocab = []
    for i in range(len(data)):
        for t in data[i][0]:
            vocab.append(t)
    return list(set(vocab))


# 保存预测结果
def data2txt(predict: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for item in predict:
            for i in range(len(item[0])):
                file.write("{} {}\n".format(item[0][i], item[1][i]))
            file.write("\n")

# 保存HMM模型参数
def save_parameter(A_path, B_path, pi_path, A, B, pi):
    np.savetxt(A_path, A, fmt="%.2f")
    np.savetxt(B_path, B, fmt="%.2f")
    np.savetxt(pi_path, pi, fmt="%.2f")


# 读取HMM模型参数
def load_parameter(A_path, B_path, pi_path):
    A = np.loadtxt(A_path)
    B = np.loadtxt(B_path)
    pi = np.loadtxt(pi_path)
    return A, B, pi


# 使用viterbi算法计算状态序列
def viterbi(A, B, pi, s, word2idx, idx2tag):
    delta = pi + B[:, word2idx.setdefault(s[0], -1)]
    # 前向传播记录路径
    path = []
    for i in range(1, len(s)):
        # 广播机制，重复加到A矩阵每一列
        tmp = delta.reshape(-1, 1) + A
        # 取最大值作为节点值，并加上B矩阵
        delta = np.max(tmp, axis=0) + B[:, word2idx.setdefault(s[i], -1)]
        # 记录当前层每一个节点的最大值来自前一层哪个节点
        path.append(np.argmax(tmp, axis=0))

    # 回溯，先找到最后一层概率最大的索引
    index = np.argmax(delta)
    best_path = [idx2tag[index]]
    # 逐层回溯，沿着path找到起点
    while path:
        tmp = path.pop()
        index = tmp[index]
        best_path.append(idx2tag[index])
    # 序列翻转
    best_path.reverse()
    return best_path

def train(data, A, B, pi, tag2idx, char2idx ):
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

    return A, B, pi

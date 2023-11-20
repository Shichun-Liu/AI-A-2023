import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
import sklearn_crfsuite


# 数据预处理，将txt文件中的数据按照换行符分割开
# data中包含3820个句子，每个句子划分成了字与标注
def DataProcess(path):
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


def word2features(sent, i):
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for label in sent]


# 保存预测结果
def data2txt(predict, path):
    with open(path, "w", encoding="utf-8") as file:
        for item in predict:
            for i in range(len(item[0])):
                file.write("{} {}\n".format(item[0][i], item[1][i]))
            file.write("\n")


if __name__ == "__main__":
    # print(DataProcess('./NER/English/train.txt')[1][0])
    # 共有3820句

    train, _, _ = DataProcess("../NER/English/train.txt")
    valid, valid_sentence, valid_tag = DataProcess("../NER/English/validation.txt")

    print("训练集长度:", len(train))
    print("验证集长度:", len(valid))
    X_train = [sent2features(s[0]) for s in train]
    y_train = [sent2labels(s[1]) for s in train]
    X_dev = [sent2features(s[0]) for s in valid]
    y_dev = [sent2labels(s[1]) for s in valid]

    # algorithm：lbfgs法求解该最优化问题，c1：L1正则系数，c2：L2正则系数，max_iterations：迭代次数，verbose：是否显示训练信息
    crf_model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True,
    )
    print("开始训练")
    crf_model.fit(X_train, y_train)
    print("训练结束")

    print("开始预测")
    labels = list(crf_model.classes_)
    y_pred = crf_model.predict(X_dev)
    # y_pred = list(chain.from_iterable(y_pred))

    print("预测结束")
    predict_tag = []
    assert len(y_pred) == len(valid_sentence), f"预测的句子数量与验证集句子数量不符,len(y_pred)={len(y_pred)},len(valid_sentence)={len(valid_sentence)}"
    for i in range(len(valid_sentence)):
        predict_tag.append([valid_sentence[i], y_pred[i]])

    data2txt(predict_tag, "./my_English_validation_result.txt")

# -*- coding: utf-8 -*-
import json
import path as p
import numpy as np
import spacy
import pickle
from tqdm import tqdm
import string

punc = string.punctuation
nlp = spacy.load('en_core_web_sm')
aaa = {}

def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/sentiwordnet.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet


# 情感图
def dependency_adj_matrix(text, senticNet, sentence_comet_mark):
    word_list = nlp(text)
    word_list = [str(x.lemma_) for x in word_list]

    aaa[text.lower()] = word_list
    print(word_list)
    seq_len = len(word_list)
    matrix_augument = np.zeros((seq_len + len(sentence_comet_mark), seq_len + len(sentence_comet_mark))).astype(
        'float32')

    for i in range(seq_len):
        for j in range(i, seq_len):
            word_i = word_list[i]
            word_j = word_list[j]
            if word_i not in senticNet or word_j not in senticNet or word_i == word_j:
                continue
            sentic = abs(float(senticNet[word_i] - senticNet[word_j]))
            matrix_augument[i][j] = sentic
            matrix_augument[j][i] = sentic
    for i in range(seq_len):
        for j in range(seq_len, seq_len + len(sentence_comet_mark)):
            word_i = word_list[i]
            if word_i not in senticNet:
                continue
            sentic = abs(float(senticNet[word_i]) - sentence_comet_mark[j - seq_len])
            matrix_augument[i][j] = sentic
            matrix_augument[j][i] = sentic
    for i in range(seq_len, seq_len + len(sentence_comet_mark)):
        for j in range(seq_len, seq_len + len(sentence_comet_mark)):
            sentic = abs(sentence_comet_mark[i - seq_len] - sentence_comet_mark[j - seq_len])
            matrix_augument[i][j] = sentic
            matrix_augument[j][i] = sentic
    return matrix_augument


def comet_sentic_score(know_file_path):
    comet_mark_for_sentences = []
    comet_mark = []
    with open(know_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                # print(len(comet_mark_for_sentences))
                comet_mark.append(comet_mark_for_sentences)
                comet_mark_for_sentences = []
            else:
                know_mark = 0
                words = line.strip('\n').split('\t')
                for word in words:
                    if word in senticNet:
                        know_mark = know_mark + senticNet[word]
                avg_know_mark = know_mark / len(words)
                comet_mark_for_sentences.append(avg_know_mark)
    # print(comet_mark)
    print('comet_len:')
    print(len(comet_mark))
    return comet_mark


def process(filename, comet_mark):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename + '.sentic', 'wb')
    for i in tqdm(range(0, len(lines), 2)):
        text = lines[i].lower().strip()
        adj_matrix = dependency_adj_matrix(text, senticNet, comet_mark[int(i / 2)])
        idx2graph[i] = adj_matrix
        print(len(adj_matrix))

    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close()


if __name__ == '__main__':
    senticNet = load_sentic_word()
    train_comet_mark = comet_sentic_score(p.dataset_path +'/' + p.know_file[0])
    test_comet_mark = comet_sentic_score(p.dataset_path + '/' + p.know_file[1])
    process(p.dataset_path + '/final_train.raw', train_comet_mark)
    process(p.dataset_path + '/final_test.raw', test_comet_mark)

    json_string = json.dumps(aaa)
    f = open(p.dataset_path + '/nlp.json', "w", encoding="utf-8")
    f.write(json_string)
    f.close()

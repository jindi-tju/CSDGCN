# -*- coding: utf-8 -*-
import path as p
import numpy as np
import spacy
import pickle
import json
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

bbb = {}


def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(document)
    bbb[text] = []
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for token in document:
        bbb[text].append(token.text)
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    return matrix


def process(filename):
    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()

    idx2graph = {}

    with open(filename + '.graph.new', 'wb') as fout:
        for i in tqdm(range(0, len(lines), 2)):
            text = lines[i].lower().strip()
            adj_matrix = dependency_adj_matrix(text)
            idx2graph[i] = adj_matrix
        pickle.dump(idx2graph, fout)


if __name__ == '__main__':
    process(p.dataset_path + '/final_train.raw')
    process(p.dataset_path + '/final_test.raw')

    json_string = json.dumps(bbb)
    f = open(p.dataset_path + '/nlp1.json', "w", encoding="utf-8")
    f.write(json_string)
    f.close()

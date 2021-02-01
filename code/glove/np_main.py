import os
import sys
import re

from corpus import Corpus
from np_mittens import GloVe


def main_entry(argv):
    # hyper-parameters
    embedding_dim = int(argv[1])
    learning_rate = 0.005
    max_iter = 10000
    corpus_path = '../../data/corpus/pinyin_wiki_all.txt'
    dictionary_path = '../../data/dictionary/pinyin_dict.txt'
    corpus_obj_output_path = f'../../output/glove/corpus_obj_{embedding_dim}d'
    output_dir, log_dir = '../../output/glove', '../../output/log_dir'
    embedding_save_path = output_dir + f'/glove_num5_{0}.{embedding_dim}d'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 读取出现频次最高的400k个词作为词汇表
    vocab = []
    with open(dictionary_path, 'r') as vbf:
        for line in vbf.readlines():
            vocab.append(line.strip())

    # 建立词典，统计共现矩阵
    dictionary = {}
    for i, word in enumerate(vocab):
        dictionary[word] = i
    corpus = []
    with open(corpus_path, 'r') as cf:
        for line in cf.readlines():
            if line.strip().split():
                corpus.append(line.strip().split())

    if os.path.exists(corpus_obj_output_path):
        corpus_obj = Corpus.load(corpus_obj_output_path) # self.dictionary, self. matrix
    else:
        corpus_obj = Corpus(dictionary=dictionary)
        corpus_obj.fit(corpus, window=10, ignore_missing=True)  # 得到稀疏的上三角矩阵
        corpus_obj.save(corpus_obj_output_path)

    glove = GloVe(n=embedding_dim, xmax=100, alpha=0.75,
                  max_iter=max_iter, learning_rate=learning_rate, tol=1e-4,
                  display_progress=100, log_dir=log_dir, log_subdir=None, test_mode=False)
    corpus_obj.matrix = corpus_obj.matrix.toarray()
    for i in range(corpus_obj.matrix.shape[0]):
        for j in range(i + 1, corpus_obj.matrix.shape[0]):
            if corpus_obj.matrix[i][j] > 0.:
                corpus_obj.matrix[j][i] = corpus_obj.matrix[i][j]
    wordvectors = glove.fit(corpus_obj.matrix).round(decimals=6)
    with open(embedding_save_path, 'w') as wvf:
        wvf.write(f'{len(wordvectors)} {embedding_dim}\n')
        for i, wv in enumerate(wordvectors):
            wvf.write(vocab[i] + ' ' + str(list(wv))[1:-1].replace(', ', ' ') + '\n')


if __name__ == '__main__':

    main_entry(sys.argv)


from corpus import Corpus
from np_mittens import GloVe

if __name__ == '__main__':
    # 读取出现频次最高的400k个词作为词汇表
    vocab = []
    with open('../../output/vocabs_100.txt', 'r') as vbf:
        for line in vbf.readlines():
            vocab.append(line.strip())

    # 建立词典，统计共现矩阵
    dictionary = {}
    for i, word in enumerate(vocab):
        dictionary[word] = i
    corpus = []
    with open('../../input/wiki.500.txt', 'r') as cf:
        for line in cf.readlines():
            corpus.append([])
            for word in line.split():
                corpus[-1].append(word)
    corpus_obj = Corpus(dictionary=dictionary)
    corpus_obj.fit(corpus, window=10, ignore_missing=True) # 得到稀疏的上三角矩阵
    corpus_obj.save('../../output/corpus_obj')
    # corpus_obj = Corpus.load('../output/corpus_obj') # self.dictionary, self. matrix
    
    glove = GloVe(n=100, xmax=100, alpha=0.75,
                 max_iter=10000, learning_rate=0.05, tol=1e-4,
                 display_progress=100, log_dir=None, log_subdir=None, test_mode=False)
    corpus_obj.matrix = corpus_obj.matrix.toarray()
    for i in range(corpus_obj.matrix.shape[0]):
        for j in range(i + 1, corpus_obj.matrix.shape[0]):
            if (corpus_obj.matrix[i][j] > 0.):
                corpus_obj.matrix[j][i] = corpus_obj.matrix[i][j]
    wordvectors = glove.fit(corpus_obj.matrix).round(decimals=6)
    with open('../../output/glove100.wv', 'w') as  wvf:
        for i, wv in enumerate(wordvectors):
            wvf.write(vocab[i] + ' ' + str(list(wv))[1:-1].replace(', ', ' ') + '\n')

    # 读取wordvectors
    # with open('../../output/glove100.wv', 'r') as rvf:
    #     vectors = []
    #     vocabs = []
    #     for line in rvf.readlines():
    #         vectors.append([])
    #         items = line.strip().split(' ')
    #         vectors[-1].extend([float(num) for num in items[1:].split(' ')])
    #         vocabs.append(items[0])
    #     print(vocabs)

import sys
import os
import re
from ast import literal_eval

from gensim.models import KeyedVectors

from generate_corpus import generate_pinyin_corpus, convert_ner_corpus_to_txt
from word2vec import get_wordvec


def get_most_similar(wv_path, word_list):
    """
        读取wordvec文件，并找出word_list的各个词的最相似词，打印出来
    :param wv_path:
    :param word_list:
    :return:
    """
    wv_model = KeyedVectors.load_word2vec_format(wv_path, binary=False)
    for word in word_list:
        print(wv_model.most_similar(word))


def compare_two_wordvec_dict(wv_path1, wv_path2):
    """
        比较两个wordvec词典集合的差异
    :param wv_path1:
    :param wv_path2:
    :return:
    """
    dict1 = set()
    dict2 = set()
    with open(wv_path1, 'r', encoding='utf8') as fin:
        for ith, line in enumerate(fin.readlines()):
            if ith == 0:
                continue
            word = line.strip().split(' ', maxsplit=1)[0]
            if word:
                dict1.add(word)
    with open(wv_path2, 'r', encoding='utf8') as fin:
        for ith, line in enumerate(fin.readlines()):
            if ith == 0:
                continue
            word = line.strip().split(' ', maxsplit=1)[0]
            if word:
                dict2.add(word)

    diff_set = dict1 - dict2 if len(dict2 - dict1) == 0 else dict2 - dict1
    print(f'Difference between two dicts: {diff_set}, dict1 len: {len(dict1)}; dict2 len: {len(dict2)}')


def extend_pinyin_dict(full_dict_path, pypinyin_dict_path, output_path='../data/dictionary/pinyin_ext.txt'):
    """
        拓展pinyin词典，并保存
    :param full_dict_path:
    :param pypinyin_dict_path:
    :param output_path:
    :return:
    """
    fout = open(output_path, 'w', encoding='utf8')
    word_list = [(line.strip().split('\t', maxsplit=1)[0], literal_eval(line.strip().split('\t', maxsplit=1)[-1])) for
                 line in open(full_dict_path, encoding='utf8')]
    set2 = set(line.strip() for line in open(pypinyin_dict_path, encoding='utf8'))

    for word, pinyin_list in word_list:
        miss_set = set()
        for pinyin in pinyin_list:
            if pinyin not in set2:
                miss_set.add(pinyin)
        if len(miss_set) > 0:
            fout.write(word + '\t' + str(list(miss_set)) + '\n')


# argv1=embedding_dim, argv2=word_pinyin
def main(argv):
    # punctuations_path is a given dict
    generate_pinyin_corpus(corpus_path='../data/corpus/ner_corpus.txt',
                           output_path='../data/corpus/pinyin_ner_corpus.txt',
                           punctuations_path='../data/dictionary/chinese_punctuations.txt',
                           save_dict_path='../data/dictionary/pinyin_dict_ner.txt',
                           max_lines=None)

    # 生成word2vec
    embedding_dim = int(argv[1])
    output_dir = '../output/word2vec'
    os.makedirs(output_dir, exist_ok=True)
    get_wordvec(corpus_path='../data/corpus/pinyin_wiki_ner.txt',
                vec_save_path=os.path.join(output_dir, f'word2vec_num5.{embedding_dim}d'),
                embedding_dim=embedding_dim)

    # dict_size = 1412
    # # wv_path = f'~/NLP/corpus/pinyin/glove/glove_num5.{dict_size}.{argv[1]}d'
    # wv_path = f'~/NLP/corpus/pinyin/word2vec/word2vec_num5.{dict_size}.{argv[1]}d'
    # get_most_similar(wv_path=wv_path, word_list=[argv[2]])


if __name__ == '__main__':
    main(sys.argv)

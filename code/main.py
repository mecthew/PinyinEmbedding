import sys
import os
import re
from ast import literal_eval

from gensim.models import KeyedVectors

from generate_corpus import generate_pinyin_corpus
from word2vec import get_wordvec


def get_most_similar(wv_path, word_list):
    wv_model = KeyedVectors.load_word2vec_format(wv_path, binary=False)
    for word in word_list:
        print(wv_model.most_similar(word))


def extend_pinyin_dict(full_dict_path, pypinyin_dict_path, output_path='../data/dictionary/pinyin_ext.txt'):
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
    # generate_pinyin_corpus(corpus_path='../data/corpus/chinese_wiki.txt',
    #                        output_path='../data/corpus/pinyin_wiki_all.txt',
    #                        punctuations_path='../data/dictionary/chinese_punctuations.txt',
    #                        save_dict_path='../data/dictionary/pinyin_dict.txt',
    #                        max_lines=None)
    # save_pinyin_dict(corpus_path='../data/corpus/pinyin_wiki.1000.txt',
    #                  output_path='../data/dictionary/pinyin_dict.txt')

    embedding_dim = int(argv[1])
    output_dir = '../output/word2vec'
    os.makedirs(output_dir, exist_ok=True)
    get_wordvec(corpus_path='../data/corpus/pinyin_wiki_all.txt',
                vec_save_path=os.path.join(output_dir, f'word2vec_num5.{embedding_dim}d'),
                embedding_dim=embedding_dim)

    # wv_path = f'../output/glove/glove_num5.{argv[1]}d'
    # wv_path = f'../output/word2vec/word2vec_num5.{argv[1]}d'
    # get_most_similar(wv_path=wv_path, word_list=[argv[2]])


if __name__ == '__main__':
    main(sys.argv)

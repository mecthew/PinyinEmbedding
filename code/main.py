import sys
import os
import re

from remove_eng import remove_eng_words, tag_non_punctuations, save_pinyin_dict
from gensim.models import KeyedVectors


def get_most_similar(wv_path, word_list):
    wv_model = KeyedVectors.load_word2vec_format(wv_path, binary=False)
    for word in word_list:
        print(wv_model.most_similar(word))


# argv1=embedding_dim, argv2=pinyin
def main(argv):
    # remove_eng_words(corpus_path='../data/corpus/chinese_wiki.txt',
    #                  output_path='../data/corpus/pinyin_wiki.txt')
    # tag_non_punctuations(corpus_path='../data/corpus/pinyin_wiki.txt',
    #                      non_pinyin_path='../data/corpus/non_pinyin.txt',
    #                      punctuations_path='../data/corpus/chinese_punctuations.txt',
    #                      replace_word='[OTHER]')
    # save_pinyin_dict(corpus_path='../data/corpus/filter_wiki.txt',
    #                  output_path='../data/dictionary/pinyin_dict.txt')
    embedding_cnt = 0
    wv_path = f'../output/glove_num5_{embedding_cnt}.{argv[1]}d'
    while os.path.exists(wv_path):
        embedding_cnt += 1
        wv_path = re.sub('_\d+\.', f'_{embedding_cnt}.', wv_path)
    wv_path = re.sub('_\d+\.', f'_{embedding_cnt - 1}.', wv_path)
    get_most_similar(wv_path=wv_path, word_list=[argv[2]])


if __name__ == '__main__':
    main(sys.argv)

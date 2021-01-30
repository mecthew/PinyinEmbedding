from remove_eng import remove_eng_words, tag_non_punctuations


def main():
    # remove_eng_words(corpus_path='../data/corpus/chinese_wiki.txt',
    #                  output_path='../data/corpus/pinyin_wiki.txt')
    tag_non_punctuations(corpus_path='../data/corpus/pinyin_wiki.txt',
                         non_pinyin_path='../data/corpus/non_pinyin.txt',
                         punctuations_path='../data/corpus/chinese_punctuations.txt',
                         replace_word='[OTHER]')


if __name__ == '__main__':
    main()

import re
import re
import time

import unicodedata
from pypinyin import lazy_pinyin, Style


def remove_eng_words(corpus_path, output_path, eng_replace_word='[ENG]', digit_replace_word='[DIGIT]'):
    t_start = time.time()
    punctuations = set()
    non_pinyin_charater_path = output_path.rsplit('/', maxsplit=1)[0] + '/non_pinyin.txt'
    with open(corpus_path, 'r', encoding='utf8') as fin, open(output_path, 'w', encoding='utf8') as fout:
        total_num = 0
        mix_eng_num = 0
        for line in fin:
            text = strip_accents(line.strip())
            if not text:
                continue
            # 先替换英文单词再替换数字
            text = re.sub('[a-zA-Z]+(-[a-zA-Z]+)?(\'?[a-zA-Z]*)', ' ' + eng_replace_word + ' ', text)
            text = re.sub('(-?\d+)(\.\d+)?%?', ' ' + digit_replace_word + ' ', text)
            text = text.split()
            token_list = []
            for t in text:
                if t == eng_replace_word:
                    token_list.append(t)
                    mix_eng_num += 1
                elif t == digit_replace_word:
                    token_list.append(t)
                else:
                    token_list.extend(list(t))

            pinyin_list = lazy_pinyin(token_list, style=Style.TONE3, neutral_tone_with_five=True, strict=False)
            fout.write(' '.join(pinyin_list) + '\n')
            for token in pinyin_list:
                if len(token) == 1:
                    punctuations.add(token)
            total_num += 1
        with open(non_pinyin_charater_path, 'w', encoding='utf8') as fout2:
            fout2.write('\n'.join(list(punctuations)))
        print(f'Total lines: {total_num}; Non-eng rate: {(total_num - mix_eng_num) / total_num};'
              f' Mix-eng rate: {mix_eng_num / total_num}')
        print(f"Finish transformation, cost {time.time() - t_start}s for {total_num} sentences")


def tag_non_punctuations(corpus_path, non_pinyin_path, punctuations_path, replace_word='[OTHER]'):
    non_pinyin_set = set([line.strip() for line in open(non_pinyin_path)])
    punctuations_set = set([line.strip() for line in open(punctuations_path)])

    output_path = corpus_path.rsplit('/', maxsplit=1)[0] + '/filter_wiki.txt'
    with open(corpus_path, 'r', encoding='utf8') as fin, \
            open(output_path, 'w', encoding='utf8') as fout:
        for line in fin:
            tokens = line.strip().split()
            tokens = [replace_word if t in non_pinyin_set and t not in punctuations_set else t for t in tokens]
            fout.write(' '.join(tokens) + '\n')


def save_pinyin_dict(corpus_path, output_path):
    with open(corpus_path, 'r') as fin, open(output_path, 'w', encoding='utf8') as fout:
        word_dict = set()
        for line in fin.readlines():
            tokens = line.strip().split()
            word_dict = word_dict.union(set(tokens))
        fout.write('\n'.join(list(word_dict)))


# 去掉类似café的音调
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

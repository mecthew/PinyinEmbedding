import multiprocessing
import re
import time
from itertools import repeat

import unicodedata
from pypinyin import lazy_pinyin, Style

NCPU = multiprocessing.cpu_count()


def generate_pinyin_corpus(corpus_path, output_path, punctuations_path,
                           save_dict_path,
                           eng_replace_word='[ENG]', digit_replace_word='[DIGIT]',
                           unk_replace_word='[UNK]',
                           max_lines=None):
    t_start = time.time()
    non_pinyin_set = set()
    corpus_set = set()
    punctuations = set(line.strip() for line in open(punctuations_path, encoding='utf8'))
    non_pinyin_character_path = save_dict_path.replace('\\', '/').rsplit('/', maxsplit=1)[0] + '/non_pinyin.txt'
    with open(corpus_path, 'r', encoding='utf8') as fin, open(output_path, 'w', encoding='utf8') as fout:
        total_num = 0
        mix_eng_num = 0
        text_corpus = []
        for line in fin:
            if not line.strip():
                continue
            text_corpus.append(line.strip())
            total_num += 1
            if max_lines and total_num >= max_lines:
                break
        pinyin_list, mix_eng_num, corpus_set = multiprocess_text2pinyin(
            corpus=text_corpus, punctuations=punctuations,
            eng_replace_word=eng_replace_word,
            digit_replace_word=digit_replace_word,
            unk_replace_word=unk_replace_word,
            num_thread=NCPU
        )

        for pinyins in pinyin_list:
            fout.write(' '.join(pinyins) + '\n')
        with open(save_dict_path, 'w', encoding='utf8') as fout3:
            fout3.write('\n'.join(list(corpus_set)))
        with open(non_pinyin_character_path, 'w', encoding='utf8') as fout2:
            fout2.write('\n'.join(list(non_pinyin_set)))
        print(f'Total lines: {total_num}; Non-eng rate: {(total_num - mix_eng_num) / total_num};'
              f' Mix-eng rate: {mix_eng_num / total_num}')
        print(f"Finish transformation, cost {time.time() - t_start}s for {total_num} sentences")


def multiprocess_text2pinyin(corpus, punctuations,
                             eng_replace_word, digit_replace_word, unk_replace_word,
                             num_thread=NCPU):
    corpus_size = len(corpus)
    text_blocks = []
    for i in range(num_thread):
        if i == num_thread - 1:
            text_blocks.append(corpus[i * corpus_size // num_thread:])
        else:
            text_blocks.append(corpus[i * corpus_size // num_thread: (i + 1) * corpus_size // num_thread])
    params = (punctuations, eng_replace_word, digit_replace_word, unk_replace_word)
    with multiprocessing.Pool(num_thread) as pool:
        res = pool.starmap(text2pinyin, zip(text_blocks, repeat(params)))
        pool.close()
        pool.join()

    concat_pinyin_list, total_eng_num = [], 0
    corpus_set = set()
    for pinyin_list, mix_eng_num, word_set in res:
        concat_pinyin_list.extend(pinyin_list)
        total_eng_num += mix_eng_num
        corpus_set = corpus_set.union(word_set)
    return concat_pinyin_list, total_eng_num, corpus_set


def text2pinyin(texts, args):
    punctuations, eng_replace_word, digit_replace_word, unk_replace_word = args
    mix_eng_num = 0
    pinyin_list = []
    word_set = set()
    for text in texts:
        text = strip_accents(text.strip())
        # 先替换英文单词再替换数字
        text = re.sub('[a-zA-Z]+(-[a-zA-Z]+)?(\'?[a-zA-Z]*)', ' ' + eng_replace_word + ' ', text)
        text = re.sub('(-?\d+)(\.\d+)?%?', ' ' + digit_replace_word + ' ', text)
        if eng_replace_word in text:
            mix_eng_num += 1

        pinyins = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, strict=False)
        pinyins = clean_pinyins(pinyins, punctuations, eng_replace_word,
                                digit_replace_word, unk_replace_word)
        pinyin_list.append(pinyins)
        word_set = word_set.union(pinyins)
    return pinyin_list, mix_eng_num, word_set


def clean_pinyins(pinyin_list, punctuations,
                  eng_replace_word, digit_replace_word, unk_replace_word):
    processed_list = []
    for pinyin in pinyin_list:
        # 英文或数字
        if pinyin in [eng_replace_word, digit_replace_word]:
            processed_list.append(pinyin)
        # pinyin
        elif pinyin[-1] in [str(i) for i in range(1, 6)] and ord('a') <= ord(pinyin[0]) <= ord('z'):
            processed_list.append(pinyin)
        # 标点符号
        elif len(pinyin) == 1 and pinyin in punctuations:
            processed_list.append(pinyin)
        else:
            for tok in pinyin.split():
                if tok in [eng_replace_word, digit_replace_word]:
                    processed_list.append(tok)
                else:
                    for subtok in list(tok):
                        if subtok in punctuations:
                            processed_list.append(subtok)
                        else:
                            processed_list.append(unk_replace_word)
    return processed_list


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

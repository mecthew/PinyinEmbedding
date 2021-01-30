import re

from tqdm import tqdm


def construct_dictionary(corpus_path):
    """构建词典
    
    Arguments:
        corpus_path {str} -- [已分好词的语料库所在路径，每篇文章文章或者段落占一行]
    
    Returns:
        dictionary [dict] -- [word的索引词典]
        vocabs [list] -- [词汇表]
    """
    with open(corpus_path, 'r') as sf:
        # 超参数
        xmax = 100  # 频率阈值
        cnt = 0 # 统计超过频率阈值的词个数
        vocasize = 400000

        worddict = {}
        wiki = tqdm(iter(sf.readlines()), desc="已处理0篇")
        p = 0
        for article in wiki:
            for word in article.split():
                worddict[word] = 0
            p += 1
            wiki.set_description('已处理%d篇' % p)
        print("wiki.zh.20191020语料库总词汇量:", len(worddict))

        sf.seek(0)
        wiki = tqdm(iter(sf.readlines()), desc="已处理0篇")
        p = 0
        for article in wiki:
            for word in article.split():
                worddict[word] += 1
            p += 1
            wiki.set_description('已处理%d篇' % p)

        wv = tqdm(worddict.values(), desc='统计频次超过%d的词个数' % xmax)
        for v in wv:
            if v > xmax:
                cnt += 1
        print("出现频次超过%d的词有%d个" %(xmax, cnt))

        # 按word出现次数从大到小排序
        wordcnt_tuple = sorted(worddict.items(), key=lambda w: w[1], reverse=True)

        # 收集频次最高的vocasize个词
        dictionary = {}
        vocabs = []
        p = 0
        for items in wordcnt_tuple:
            zh = True
            for _char in items[0]:
                if not '\u4e00' <= _char <= '\u9fa5':
                    zh = False
                    break

            # if not bool(re.search('[a-zA-Z]', items[0])):
            if zh:
                dictionary[items[0]] = p
                vocabs.append(items[0])
                p += 1
                if p == vocasize:
                    break
            
        return dictionary, vocabs

if __name__ == '__main__':
    dictionary, vocabs = construct_dictionary('../../input/wiki.zh.segs.txt')
    with open('../../output/vocabs.txt', 'w') as vf:
        for word in vocabs:
            vf.write(word + '\n')

from pyltp import Segmentor
import os
from tqdm import tqdm

INPUT_PATH = '/home/brooksj/PycharmProjects/NLP12345/input'
LTPMODEL_PATH = os.path.join(INPUT_PATH, 'ltp_data_v3.4.0')
cws_model_path = os.path.join(LTPMODEL_PATH, 'cws.model')

seg = Segmentor()
seg.load_with_lexicon(cws_model_path, os.path.join(INPUT_PATH, 'lexicon_ex.txt'))

with open('./wiki.zh.txt.jian', 'r') as rf, open('./wiki.zh.segs.txt', 'w') as wf:
    wiki = tqdm(iter(rf.readlines()), desc=u'已分词0篇文章')
    i = 0
    for line in wiki:
        for sent in line.split('\s+'):
            words = list(seg.segment(sent))
            wf.write(' '.join(words) + ' ')
        wf.write('\n')
        i += 1
        if i % 100 == 0:
            wiki.set_description(u'已分词%d篇文章' % i)


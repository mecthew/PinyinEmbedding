from gensim.models import word2vec
import logging


# 对分词后的文本训练Word2vec模型
def get_wordvec(corpus_path, vec_save_path, embedding_dim):
    # 获取日志信息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    # 加载分词后的文本
    # sentences = word2vec.Text8Corpus(corpus_path)   # 这个函数读取分好词的文本会出现多个token组成词的问题
    sentences = word2vec.LineSentence(corpus_path)
    # 训练模型
    # alpha为初始learning rate
    model = word2vec.Word2Vec(sentences, size=embedding_dim, alpha=0.025,
                              hs=0, min_count=1, window=5, iter=10, workers=8)
    # # model.save(model_name)
    vec_save_path = vec_save_path.rsplit('.', maxsplit=1)[0] + f'.{len(model.wv.vocab)}.' + vec_save_path.rsplit('.', maxsplit=1)[-1]
    model.wv.save_word2vec_format(vec_save_path, binary=False)


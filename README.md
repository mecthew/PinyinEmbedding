# PinyinEmbedding

## 使用方法
+ 数据使用中文wiki，包括9,192,286个句子。数据存放在data/corpus目录下。
+ generate_pinyin_corpus函数使用pypinyin工具包将中文转为pinyin，无法转换的英文单词、数字替换成[ENG]、[DIGIT]，标点符号维持原样，其他token替换成[UNK].
+ 训练pinyin embedding
   1. 使用generate_pinyin_corpus函数先生成pinyin训练语料
   1. python main [embedding_dim] 生成word2vec embedding（main.py文件，可自行修改）
   2. python np_main [embedding_dim] 生成glove embedding（glove/np_main.py文件） 
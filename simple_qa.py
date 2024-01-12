import json
from typing import Union, Optional
import numpy as np
import string
import re
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
import pdb

json_path = '.train-v2.0.json'

txt_path = './glove.6B.100d.txt'

stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))
vectorizer = TfidfVectorizer()  # 初始化一个tf-idf向量转换器vectorizer

def read_corpus(filename: str) -> Union[list, list]:
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到qlist，alist中
    qlist = ['问题1'，'问题2'，...,'问题n']
    alist = ['答案1'，'答案2'，...,'答案n']
    务必要让每一个问题都对应一个答案（下标位置一致）
    """
    qlist = []
    alist = []

    data = json.load(open(filename))
    data = data['data'] #list

    for da in data:
        paragraphs = da['paragraphs'] #list
        for para in paragraphs:
            qas = para['qas'] #list
            for x in qas:
                if x['answers'] != []:
                    qlist.append(x['question'])
                    alist.append(x['answers'][0]['text'])
    assert len(qlist) == len(alist), '问题列表和答案列表长度不一致'
    return qlist, alist

#TODO: 统计一下在qlist总共出现了多少单词，以及出现了多少个不同的单词
#   这里虚高做简单的分词，对于英文根据空格来粉刺即可，其他过滤暂不考虑（只需分词）
#去标点符号，分册，得到词-词频的字典
def segmentWords(lst: list) -> Union[int, dict]:
    """
    :param lst: 待分词的文本列表
    :return: 返回词频总数，词-词频的字典
    """
    total = 0
    word_dict = {}
    for line in lst:
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        sentence = pattern.sub('', line)
        words = sentence.split()
        for word in words:
            word_dict[word] = word_dict.get(word, 0) + 1
            total += 1
    return total, word_dict

# TODO: 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
#       把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
#       需要使用matplotlib里的plot函数。y轴是词频
def count_words(lst: dict) -> Union[list, list]:
    """
    :param lst: 待统计的文本列表
    :return: 返回单词列表，词频列表
    """
    word_freq = []
    word_list = []

    word_sorted = sorted(lst.items(), key=lambda k:k[1], reverse = True) #按词频排序
    for line in word_sorted:
        word_list.append(line[0])
        word_freq.append(line[1])

    return word_freq, word_list

def show_words(lst: list) -> None:
    """
    :param lst: 待统计的文本列表
    :return: None
    """
    # print(word_freq[:100])
    # print(word_list[:100])
    # x = range(total_diff_words)
    # plt.plot(x, word_freq, 'ro')
    # plt.ylabel("word frequency")
    #plt.show()

    temp = [n for n in lst if n <= 50]
    plt.plot(range(len(temp)), temp, color='r', linestyle='-', linewidth=2)
    plt.ylabel("word frequency")
    plt.show()

# 预处理：去标点符号，去停用词，stemming,将数字转换为'#number'表示
def preprocessing(lst: list) -> Union[dict, list]:
    """
    :param lst: 待预处理文本列表
    :return: 返回预处理后的文本字典，列表（过滤掉常用词汇the之类）
    """
    new_lst = []
    word_dic = {}
    for line in lst:
        pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
        sentence = pattern.sub('', line)
        sentence = sentence.lower()
        words = sentence.split()
        temp = []
        #stopwoeds = set(stopwords.words('english'))    过滤掉常用的词汇
        for word in words:
            if word not in stopwords:
                word = "#number" if word.isdigit() else word
                w = stemmer.stem(word)
                word_dic[w] = word_dic.get(w, 0) + 1
                temp.append(w)
        new_lst.append(temp)
    return word_dic, new_lst

#画出100分为类的词频统计图
def drawgraph(dic: dict, name: str) -> None:
    """
    :param dic: 词频字典
    :param name: 文件名
    :return: 画出的统计图
    """
    freq = list(dic.values())
    freq.sort(reverse=True)
    temp = [n for n in freq if n <= 50]
    plt.plot(range(len(temp)), temp, 'r-')
    plt.ylabel(name)
    plt.show()

#过滤词频小于2，大于10000的单词
def filterword(dic: dict, lst: list, low: int, high: int) -> list:
    """
    :param dic: 词频字典
    :param lst: 待过滤的文本列表
    :param low: 过滤词频小于low的单词
    :param high: 过滤词频大于high的单词
    :return: 过滤后的文本列表
    """
    temp = []
    new_list = []
    for k, v in dic.items():
        if low <= v <= high:
            temp.append(k)
    for line in lst:
        words = ([word for word in line if word in temp])
        #' '.join(words) 将列表中的元素以空格连接起来,多个字符串合成一个字符串
        new_list.append(' '.join(words))
    return new_list


#TODO: 矩阵X有什么特点？ 计算一下它的稀疏度
def caculate_sparsity(X: Optional[TfidfVectorizer]) -> float:
    """
    :param X: 矩阵X
    :return: 稀疏度
    """
    t = 0
    #(86821, 14950)的维度，86821个过滤后的单词，14950种单词（也就是最大的编号）
    x_mat = X.toarray()
    n = len(x_mat)
    m = len(x_mat[0])
    for i in range(n):
        for j in range(m):
            if x_mat[i][j] != 0:
                t += 1
    return t / (n * m)


# 预处理输入的文本
def inputprocessor(input_str: str) -> list:
    """
    :param input_str: 输入的文本字符串
    :return: 预处理后的文本列表 
    """
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    sentence = pattern.sub("", input_str)
    sentence = sentence.lower()
    words = sentence.split()
    result = []
    for word in words:
        if word not in stopwords:
            word = "#number" if word.isdigit() else word
            w = stemmer.stem(word)
            result.append(w)
    return result


# 求两个set的交集
def intersections(set1: set, set2: set) -> set:
    return set1.intersection(set2)
 
#加载glove词向量
def loadGlove(path: str) -> Union[dict, list]:
    """
    :param path: glove词向量文件路径
    :return: 词向量词典和词向量矩阵
    """
    vocab = {}
    embedding = []
    vocab["UNK"] = 0
    embedding.append([0]*100)
    file = open(path, 'r', encoding='utf8')
    i = 1
    for line in file:
        row = line.strip().split()
        vocab[row[0]] = i
        embedding.append(row[1:])
        i += 1
    print("Finish load Glove")
    file.close()
    return vocab, embedding
 
# 转换为词向量
def word_to_vec(words: list, vocab: dict, emb: list) -> list:
    """
    :param words: 待转换的单词列表
    :param vocab: 词向量词典
    :param emb: 词向量矩阵
    :return: 转换后的词向量列表
    """
    vec = []
    
    for word in words:
        if word in vocab:
            idx = vocab[word]
            vec.append(emb[idx])
        else:
            idx = 0
            vec.append(emb[idx])
    return vec

def top5results_emb(input_q: str) -> str:
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q，转换成句子向量
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    # 问题预处理
    result = inputprocessor(input_q)
    # 输入问题的词向量
    input_q_vec = word_to_vec(result, vocabs, emb)
    
    # 根据倒排表筛选出候选问题索引
    candidates = []
    for word in result:
        if word in inverted_idx:
            ids = inverted_idx[word]
            candidates.append(set(ids))
    candidate_idx = list(reduce(intersections, candidates))  # 候选问题索引
    
    # 计算相似度得分, 隐藏次的大小用100来表示
    scores = []
    for i in candidate_idx:
        sentence = new_qlist[i].split()
        vec = word_to_vec(sentence, vocabs, emb)
        score = cosine_similarity(input_q_vec, vec)[0]
        scores.append((i, score[0]))
    scores_sorted = sorted(scores, key=lambda k:k[1], reverse=True)
    
    # 根据索引检索top 5答案
    answers = []
    i = 0
    for (idx, score) in scores_sorted:
        if i < 5:
            answer = alist[idx]
            answers.append(answer)
        i += 1
    
    return answers

def create_inverted_list(new_qlist: list) -> dict:
    """
    :param qlist: 筛选过的问题列表
    :return: 倒排表
    """
    inverted_idx = {}  # 定一个一个简单的倒排表, 倒排表就是所有的词出现在哪些句子当中
    for i in range(len(new_qlist)):
        for word in new_qlist[i].split():
            if word not in inverted_idx:
                inverted_idx[word] = [i]
            else:
                inverted_idx[word].append(i)
    for k in inverted_idx:
        inverted_idx[k] = sorted(inverted_idx[k])

    return inverted_idx


def run_show(qlist: list, alist: list) -> None:
    """
    :param qlist: 问题列表
    :param alist: 答案列表
    :return: None
    """
    word_total, q_dict = segmentWords(qlist)
    a_total, a_dic = segmentWords(alist)
    total_diff_words = len(q_dict.keys())

    # 将字典转换为列表
    items = list(q_dict.items())

    print("总共{}个单词，总共{}个不同的单词。".format(word_total, total_diff_words))

    word_freq_q, _ = count_words(q_dict)
    word_freq_a, _ = count_words(a_dic)

    show_words(word_freq_q)
    show_words(word_freq_a)

def main(json_path: str, txt_path: str) -> dict:
    """
    :param json_path: 语料库的路径
    :param txt_path: 预训练的词向量路径
    :return: new_qlist, alist, inverted_idx, vocabs, emb
    """
    qlist, alist = read_corpus(json_path)
    #run_show(qlist, alist)

    # 预处理
    q_dict,q_list = preprocessing(qlist)
    a_dict,a_list = preprocessing(alist)

    # 可视化预处理结果
    # drawgraph(q_dict,"word frequency of qlist")
    # drawgraph(a_dict, "word frequency of alist")

    # 过滤词频小于2，大于10000的单词   
    new_qlist = filterword(q_dict, q_list, 2, 10000)
    new_alist = filterword(a_dict, a_list, 2, 10000)
   
    # X = vectorizer.fit_transform(new_qlist)  # 转换, 得到X矩阵
    """
    (0, 10307)    0.5179626821145004
    (0, 1421)     0.4623335969616919
    (0, 12670)    0.4750834595803454
    (0, 1529)     0.5406089265729778
    (1, 5692)     0.5319449141434635
    (1, 2843)     0.581212084677154
    (1, 877)      0.38879849330344335
    (1, 1529)     0.47755926600487236
    """
    # sparsity = caculate_sparsity(X)
    # sparsity = 0.00034930747261915386

    # 构建倒排表
    inverted_idx = create_inverted_list(new_qlist)

    # 读取每一个单词的嵌入。这个是 D*H的矩阵，这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，那句子向量怎么表达？
    # 其中，最简单的方式 句子向量 = 词向量的平均（出现在问句里的）， 如果给定的词没有出现在词典库里，则忽略掉这个词。
    vocabs, emb = loadGlove(txt_path)

    return new_qlist, alist, inverted_idx, vocabs, emb


if __name__ == '__main__':
    
    new_qlist, alist, inverted_idx, vocabs, emb= main(json_path=json_path, txt_path=txt_path)
    
    # TODO: 编写几个测试用例，并输出结果
    print (top5results_emb("when did Beyonce start becoming popular"))
    print (top5results_emb("what languge does the word of 'symbiosis' come from"))


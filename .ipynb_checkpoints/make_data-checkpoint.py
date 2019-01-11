import pickle
import pandas as pd
import MeCab
import numpy as np
import re
from tqdm import tqdm

class make_data():
    def __init__(self):
        self.text_label = pd.read_csv('./data/tweet_data.csv')
        self.text_label.columns = ["text","label"]
        self.tagger = MeCab.Tagger('-Owakati -d /home/machi52/.local/lib/mecab/dic/mecab-ipadic-neologd')
        self.term_vec    = pickle.loads(open('./data/glove_vectors.pkl', 'rb').read())
    def read_data(self):
        texts = self.text_label.text.apply(lambda x: self.tagger.parse(str(x)).split()).values
        labels = self.text_label.label.apply(lambda x: x)

        return texts,labels

    def word_label_id(self,texts):
        word2id = {}
        for text in texts: 
            for word in text:
                if word in word2id:
                    continue
                word2id[word] = len(word2id)

        id2word = {}
        for key, value in word2id.items():
            id2word[value] = key

        label2id = {}
        for i, label in enumerate((self.text_label).label.value_counts().keys()):
            label2id[label] = i
        
        term_vec = self.term_vec
        return word2id,id2word,label2id,term_vec     

    def embedding_vector(self,term_vec,word2id):
        embedding_vectors = np.random.uniform(-0.01, 0.01, (len(word2id), 300))
        count = 0
        count_oov = 0
        vocab = list(term_vec.keys())
        oov = set()
        for word in tqdm(word2id.keys()):
            if word in vocab:
                embedding_vectors[word2id[word]] = term_vec[word]
                count += 1
                continue
            count_oov += 1
            oov.add(word)
        np.savez("./data/text_emb_30k.npz",embedding_vectors)
        return embedding_vectors

    def class_vector(self,class_name):
        symbol_list = ['（','）','、']
        for label_words in class_name:
            for symbol in symbol_list:
                if symbol in label_words:
                    label_words.remove(symbol)
        value_list = [ [ term_vec[i] for i in l] for l in class_name]
        value_mean = np.array([ np.mean(l, axis=0)  for l in value_list])
        return value_mean


texts,labels = make_data().read_data()
word2id,id2word,label2id,term_vec = make_data().word_label_id(texts)
# embedding_vectors = make_data().embedding_vector(term_vec,word2id)
embedding_vectors = np.load("./data/text_emb_30k.npz")["arr_0"]

class_name = ["😊","😵","😡","😭","😱","😩"]
value_mean = make_data().class_vector(class_name)


id_texts = [[word2id[word] for word in text]for text in texts]
f = lambda x: np.pad(x, pad_width=(0, 305-len(x)), mode='constant', constant_values=-1)
id_texts = np.array(list(map(f, id_texts)), np.int32)


labels_multi = [np.array(re.sub("\[|\]","",label).split(","),dtype="float32") for label in labels]
labels_multi = np.array(labels_multi,dtype='int32')

open('./data/embedding_vectors.pkl', 'wb').write(pickle.dumps(embedding_vectors))
open('./data/class_vectors.pkl', 'wb').write(pickle.dumps(value_mean))
open('./data/id_texts.pkl', 'wb').write(pickle.dumps(id_texts))
open('./data/labels_multi.pkl', 'wb').write(pickle.dumps(labels_multi))
open('./data/word2id.pkl', 'wb').write(pickle.dumps(word2id))
import pandas as pd
import numpy as np
import cupy as cp
import chainer
import glob
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import initializers
import codecs
import MeCab
from gensim.models import KeyedVectors
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import sys
from sklearn import metrics
from chainer.training import extensions


class LEAM(chainer.Chain):
    
    def __init__(self, vocab_size, n_class=6,filter_size=55):
        super(LEAM, self).__init__()
        self.n_class = n_class
        with self.init_scope():
            self.embed = L.EmbedID(
                vocab_size, 300, ignore_label=-1) # f_0
            self.embed_class = L.EmbedID(
                n_class, 300, ignore_label=-1)
            self.conv1 = L.ConvolutionND(1, None, n_class, filter_size, 1, filter_size//2) # f_1
            self.fc2 = L.Linear(300, n_class) # f_2

    def __call__(self, x):
        V = self.embed(x)
        V_norm = F.normalize(V.transpose(0, 2, 1), axis=1)
        C = self.embed_class.W
        C_norm = F.normalize(C, axis=1)
        G = F.matmul(F.broadcast_to(C_norm, (V_norm.shape[0], C_norm.shape[0], C_norm.shape[1])), V_norm)
        u = F.relu(self.conv1(G))
        m = F.maxout(u, pool_size=self.n_class, axis=1)
        beta = F.softmax(m, axis=2)
        z = F.sum((V * F.broadcast_to(beta.transpose(0,2,1), V.shape)), axis=1)
        z = self.fc2(F.dropout(z))
        z = F.sigmoid(z)
        z_class = self.fc2(F.dropout(C))
        out = F.concat([z, z_class], axis=0)
        return out 

class pred_text():
    
    def __init__(self):
        self.embedding_vectors = pickle.loads(open('./data/embedding_vectors.pkl', 'rb').read())
        self.id_texts = pickle.loads(open('./data/id_texts.pkl', 'rb').read())
        self.labels_multi = pickle.loads(open('./data/labels_multi.pkl', 'rb').read())
        self.value_mean = pickle.loads(open('./data/class_vectors.pkl', 'rb').read())
        self.word2id    = pickle.loads(open('./data/word2id.pkl', 'rb').read())
        
        self.n_class = len(self.labels_multi[0])
        self.vocab_size = self.embedding_vectors.shape[0]
    
    def auc_fun_cnn(self,z, t):
        aucs = []
        with cuda.get_device(z):
            z = z
        t = chainer.cuda.to_cpu(t)
        z = chainer.cuda.to_cpu(z.data)

        for i in range(6):
            fpr, tpr, thresholds = metrics.roc_curve(t[::,i], z[::,i]) 
            aucs.append(metrics.auc(fpr, tpr))

        return Variable(np.array(sum(aucs)/len(aucs)))
    
    def lossfun_multi_cnn(self,z, t):
        with cuda.get_device(z):
            z = z
        loss = - (F.mean(t * F.log(z) + (1-t)*F.log1p(-z)))
        return loss

    
    def auc_fun_leam(self, z, t):
        aucs = []
        with cuda.get_device(z):
            z = z[:-self.n_class]
        t = np.array(chainer.cuda.to_cpu(t))
        z = chainer.cuda.to_cpu(z.data)
        for i in range(6):
            fpr, tpr, thresholds = metrics.roc_curve(t[::,i], z[::,i]) 
            aucs.append(metrics.auc(fpr, tpr))
        print(aucs)   
        return Variable(np.array(sum(aucs)/len(aucs)))

    def lossfun_multi_leam(self, z, t):
        with cuda.get_device(z):
                z_class = z[-self.n_class:]
                z = z[:-self.n_class]
        t_class = Variable(cp.array(range(self.n_class)))
        loss = - (F.mean(t * F.log(z) + (1-t)*F.log1p(-z))) / z.shape[0] + F.softmax_cross_entropy(z_class, t_class)
        return loss   
    

    def pred(self,text):
        emoji_index = {'😊': 0, '😡': 5, '😩': 2, '😭': 4, '😱': 1, '😵': 3}
        m = MeCab.Tagger("-Owakati -d /home/machi52/.local/lib/mecab/dic/mecab-ipadic-neologd")
        index_emoji = {index:emoji for emoji, index in emoji_index.items()}
        model = LEAM(len(self.word2id), 6)
        models_name = glob.glob("./appr-leam/models_*")[-1]
        serializers.load_npz(models_name, model)
        output = {}
        gpu_id = 0

        if gpu_id >= 0:
            model.to_gpu(gpu_id)
        model = L.Classifier(model, lossfun=self.lossfun_multi_leam, accfun=self.auc_fun_leam) 
        f = lambda x: np.pad(x, pad_width=(0, 305-len(x)), mode='constant', constant_values=-1)

        line = text
        buff = []
        for i,term in enumerate(m.parse(line).strip().split()):
            if self.word2id.get(term) != None:
                buff.append(self.word2id[term])
            elif re.match("|".join(list(emoji_index.keys())),term) != None:
                return term

        id_texts = np.array(list(map(f, [buff])), np.int32) 
        results = model.predictor(cp.array(id_texts))
        res = {index_emoji[i]:score for i,score in enumerate(results[0].data.tolist())}
        emojis = []
        scores = []
        for emoji, score in sorted(filter(lambda x:x[1]>0.001, res.items()), key=lambda x:x[1]*-1)[:20]:
            emojis.append(emoji)
            scores.append(score*100)   
        for i in range(len(emojis)):    
            output.update({emojis[i]:int(scores[i])}) 
    #     print(output)    
        return "("+" ".join(["{0}:{1}%".format(key, value) for key,value in sorted(output.items(), key=lambda x: -x[1])]) +")"

if __name__ == '__main__':
    
    text = input('テキストを入力してください：\n')
    text = re.sub('😊|😵|😡|😭|😱|😩', '', text)
    print(pred_text().pred(text))
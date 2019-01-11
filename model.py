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
    
class CNN(chainer.Chain):
    
    def __init__(self, vocab_size, n_class=6,filter_size=[3,5,7,9,11]):
        super(cnn, self).__init__()
        self.n_class = n_class
        self.filter_size = filter_size
        with self.init_scope():
            self.embed = L.EmbedID(
                vocab_size, 300, ignore_label=-1) # f_0
            self.conv1 = L.ConvolutionND(1, None, 305, filter_size[0], 1, filter_size[0]//2) # f_1
            self.conv2 = L.ConvolutionND(1, None, 305, filter_size[1], 1, filter_size[1]//2)
            self.conv3 = L.ConvolutionND(1, None, 305, filter_size[2], 1, filter_size[2]//2)
            self.conv4 = L.ConvolutionND(1, None, 305, filter_size[3], 1, filter_size[3]//2)
            self.conv5 = L.ConvolutionND(1, None, 305, filter_size[4], 1, filter_size[4]//2)
            self.fc2 = L.Linear(None, n_class) # f_2

    def __call__(self, x):
        V = self.embed(x)
        
        u1 = F.relu(self.conv1(V))       
        u2 = F.relu(self.conv2(V))
        u3 = F.relu(self.conv3(V))
        u4 = F.relu(self.conv4(V))
        u5 = F.relu(self.conv5(V))
        
        m1 = F.max_pooling_1d(u1,self.filter_size[0],1,pad=self.filter_size[0]//2)
        m2 = F.max_pooling_1d(u2,self.filter_size[1],1,pad=self.filter_size[1]//2)
        m3 = F.max_pooling_1d(u3,self.filter_size[2],1,pad=self.filter_size[2]//2)
        m4 = F.max_pooling_1d(u4,self.filter_size[3],1,pad=self.filter_size[3]//2)
        m5 = F.max_pooling_1d(u5,self.filter_size[4],1,pad=self.filter_size[4]//2)
        
        z = F.concat([m1,m2,m3,m4,m5],axis=1)
        
        m = F.reshape(z,(z.shape[0],300*z.shape[1]))
        m = F.normalize(m, axis=1)
        m = self.fc2(F.dropout(m))
        out = F.sigmoid(m)
        return out
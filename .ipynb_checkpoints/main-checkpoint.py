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

import model
    
class pred_emotion():
    def __init__(self):
        self.embedding_vectors = pickle.loads(open('./data/embedding_vectors.pkl', 'rb').read())
        self.id_texts = pickle.loads(open('./data/id_texts.pkl', 'rb').read())
        self.labels_multi = pickle.loads(open('./data/labels_multi.pkl', 'rb').read())
        self.value_mean = pickle.loads(open('./data/class_vectors.pkl', 'rb').read())
        
        self.n_class = len(self.labels_multi[0])
        self.vocab_size = self.embedding_vectors.shape[0]
    
    def auc_fun_cnn(z, t):
        aucs = []
        with cuda.get_device(z):
            z = z
        t = chainer.cuda.to_cpu(t)
        z = chainer.cuda.to_cpu(z.data)

        for i in range(6):
            fpr, tpr, thresholds = metrics.roc_curve(t[::,i], z[::,i]) 
            aucs.append(metrics.auc(fpr, tpr))

        return Variable(np.array(sum(aucs)/len(aucs)))
    
    def lossfun_multi_cnn(z, t):
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
    
    def train_leam(self,gpu_id=0,epoch_size=1000,max_epoch=15,batch_size=128):
        train_x, test_x, train_y, test_y = train_test_split(self.id_texts, self.labels_multi, random_state=0)
        train = datasets.TupleDataset(train_x, train_y)
        test = datasets.TupleDataset(test_x, test_y)

        train_iter = iterators.SerialIterator(train, batch_size)
        test_iter = iterators.SerialIterator(test, batch_size, False, False)

        
        models = model.LEAM(self.vocab_size, self.n_class)
        models.embed.W.copydata(Variable(self.embedding_vectors))
        models.embed_class.W.copydata(Variable(self.value_mean))

        if gpu_id >= 0:
            models.to_gpu(gpu_id)


        models = L.Classifier(models, lossfun=self.lossfun_multi_leam, accfun=self.auc_fun_leam)
        optimizer = optimizers.Adam(alpha=0.001)
        optimizer.setup(models)
        updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (epoch_size * max_epoch, 'iteration'), out='./appr-leam')

        trainer.extend(extensions.LogReport(trigger=(epoch_size, 'iteration')))
        trainer.extend(extensions.snapshot(filename='snapshot_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.snapshot_object(models.predictor, filename='models_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.Evaluator(test_iter, models, device=gpu_id), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.observe_lr(), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.ProgressBar(update_interval=1000))
        trainer.run()
    
    
    def train_cnn(self,gpu_id=0,epoch_size=100,max_epoch=15,batch_size=128):
        train_x, test_x, train_y, test_y = train_test_split(self.id_texts, self.labels_multi, random_state=0)
        train = datasets.TupleDataset(train_x, train_y)
        test = datasets.TupleDataset(test_x, test_y)

        train_iter = iterators.SerialIterator(train, batch_size)
        test_iter = iterators.SerialIterator(test, batch_size, False, False)

        
        models = model.CNN(self.vocab_size, self.n_class)
        models.embed.W.copydata(Variable(self.embedding_vectors))

        if gpu_id >= 0:
            models.to_gpu(gpu_id)


        models = L.Classifier(models, lossfun=self.lossfun_multi_cnn, accfun=self.auc_fun_cnn)
        optimizer = optimizers.Adam(alpha=0.001)
        optimizer.setup(models)
        updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (epoch_size * max_epoch, 'iteration'), out='./appr-cnn')

        trainer.extend(extensions.LogReport(trigger=(epoch_size, 'iteration')))
        trainer.extend(extensions.snapshot(filename='snapshot_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.snapshot_object(models.predictor, filename='models_iteration-{.updater.iteration}'), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.Evaluator(test_iter, models, device=gpu_id), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.observe_lr(), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=(epoch_size, 'iteration'))
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.ProgressBar(update_interval=1000))
        trainer.run()
    def eval_cnn(self):
        models = model.CNN(len(word2id), 6)
        models_name = glob.glob("./appr-cnn/models_*")[-1]
        serializers.load_npz(models_name, models)

        gpu_id = 0
        if gpu_id >= 0:
            models.to_gpu(gpu_id)
        models = L.Classifier(models, lossfun=lossfun_multi_cnn, accfun=auc_fun_cnn)    
        models.to_cpu()

        div = len(test)//3
        count = test_x.shape[0]//div
        Bscore_list = []
        auc_list = []
        for i in range(count):
            print(i)
            pred_y = models.predictor(np.array(test_x[i*div:(i+1)*div]))
            true_y = test_y[i*div:(i+1)*div]
            Bscore_list.append(Bscore(true_y,pred_y))
            auc_list.append(aucScore(true_y,pred_y))
            pred_y = 0
            true_y = 0

        Bscore_list = np.array(Bscore_list)
        auc_list = np.array(auc_list)
        B_ave = []
        auc_ave = []
        for i in range(6):
            B_ave.append(sum(Bscore_list[:,i])/count)
            auc_ave.append(sum(auc_list[:,i])/count)

        print("-"*50)
        print(B_ave)
        print(auc_ave) 
    def eval_leam(self):
        models = model.LEAM(self.vocab_size, self.n_class)
        models_name = glob.glob("./appr-leam/models_*")[-1]
        
        serializers.load_npz(models_name, models)
        
        train_x, test_x, train_y, test_y = train_test_split(self.id_texts, self.labels_multi, random_state=0)
        
        gpu_id = 0
        if gpu_id >= 0:
            models.to_gpu(gpu_id)
        models = L.Classifier(models, lossfun=self.lossfun_multi_leam, accfun=self.auc_fun_leam)   
        models.to_cpu()
        
        div = 2500
        count = test_x.shape[0]//div
        Bscore_list = []
        auc_list = []
        fscore_list = []
        for i in range(count):
            pred_y = models.predictor(np.array(test_x[i*div:(i+1)*div]))
            true_y = test_y[i*div:(i+1)*div]
            Bscore_list.append(self.Bscore(true_y,pred_y[:-6]))
            auc_list.append(self.aucScore(true_y,pred_y[:-6]))
            fscore_list.append(self.precision_recall_F(true_y,pred_y[:-6]))    
            pred_y_ = 0
            true_y = 0

        Bscore_list = np.array(Bscore_list)
        auc_list = np.array(auc_list)
        fscore_list = np.array(fscore_list)
        B_ave = []
        auc_ave = []
        f_ave = []
        for i in range(6):
            B_ave.append(sum(Bscore_list[:,i])/count)
            auc_ave.append(sum(auc_list[:,i])/count)
            f_ave.append(sum(fscore_list[:,i])/count)

        print("-"*50)
        print(B_ave)
        print(auc_ave)
        print(f_ave)
    
    def Bscore(self,true_y,pred_y):
        lists = []
        for i in range(6):
            lists.append(sum((true_y[:,i]-pred_y[:,i].data)**2)/pred_y.shape[0])    
        return lists  

    def aucScore(self,true_y,pred_y):
        return self.plot_graph(true_y,pred_y)

    def plot_graph(self,y_true,y_pred):        
        lists = []
        for i in range(6):
            fpr, tpr, thresholds = metrics.roc_curve(y_true[:,i], y_pred[:,i].data)
            # ついでにAUCも
            auc = metrics.auc(fpr, tpr)
            lists.append(auc)   
        return lists
    def precision_recall_F(self,true_y,pred_y):
        from sklearn.metrics import f1_score
        lists = []
        pred_y = (pred_y.data >= 0.3).astype(np.int32)
        for i in range(6):
            lists.append(f1_score(true_y[:,i], pred_y[:,i]))   
        return lists 

if __name__ == '__main__':
    if '--leam_train' in sys.argv:
        pred_emotion().train_leam()
    if '--leam_eval' in sys.argv:
        pred_emotion().eval_leam()
    if '--cnn_train' in sys.argv:
        pred_emotion().train_cnn()
    if '--cnn_eval' in sys.argv:
        pred_emotion().eval_cnn()
        
        
    
    
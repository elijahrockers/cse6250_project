#set seed
seed = 57839 + 6250
vecdims = 100
ks = [2,3,4,5,6,7,8,9,10]
import os
os.environ['PYTHONHASHSEED']=str(seed)
import sys
import pdb

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import csv
import pandas as pd
import scipy
import scipy.stats as stats
import functools
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn import metrics, preprocessing, svm
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf

import tensorflow as tf
tf.random.set_seed(seed)

from tensorflow.python.keras.metrics import Metric
from tensorflow import keras
import talos
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Flatten, Embedding, SimpleRNN, LSTM, GRU, Bidirectional,Dropout

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

os.chdir("/mnt/host/Homework/proj/replicate")

# read in datasets (already pre-processed)
def readcsv(fname,istext):
    if istext:
        with open(fname,'rt') as f:
            reader=csv.reader(f)
            next(reader)
            data = []
            for row in reader:
                for item in row:
                    data.append(item)
            f.close()
    else:
        with open(fname,'r') as f:
            reader=csv.reader(f,delimiter=';')
            next(reader)
            data = list(reader)
            data = np.asarray(data, dtype='int')
            f.close()
    return data 

# read in training, validation, and test set utterances
train_text = readcsv('./DatasetsForH1/H1_train_texts.csv',True)
val_text = readcsv('./DatasetsForH1/H1_validate_texts.csv', True)
test_text = readcsv('./DatasetsForH1/H1_test_texts.csv',True)

# read in training, validation, and test set labels
train_labels = readcsv('./DatasetsForH1/H1_train_labels.csv',False)[:,0:9]
val_labels = readcsv('./DatasetsForH1/H1_validate_labels.csv', False)[:,0:9]
test_labels = readcsv('./DatasetsForH1/H1_test_labels.csv',False)[:,0:9]

schemas = ["Attach","Comp","Global","Health","Control","MetaCog","Others","Hopeless","OthViews"]

# prepare tokenizer
max_words = 2000
t = Tokenizer(num_words = max_words)
t.fit_on_texts(train_text)
vocab_size = len(t.word_index) + 1

# integer encode all utterances
encoded_train = t.texts_to_sequences(train_text)
encoded_validate = t.texts_to_sequences(val_text)
encoded_test = t.texts_to_sequences(test_text)

# pad documents to a max length of 25 words
max_length = 25

padded_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_validate = pad_sequences(encoded_validate, maxlen=max_length, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')

# load GLoVE embedding into memory
embeddings_index = dict()
f = open('./glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create an embedding matrix
vec_dims = vecdims
embedding_matrix = np.zeros((vocab_size, vec_dims))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
# create tfidf weighted encoding matrix of utterances
train_sequences = t.texts_to_matrix(train_text,mode='tfidf')
val_sequences =  t.texts_to_matrix(val_text,mode='tfidf')
test_sequences = t.texts_to_matrix(test_text,mode='tfidf')

# we want to normalize the word vectors
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
    
# create utterance embeddings as tfidf weighted average of normalized word vectors
def seq2vec(datarow,embedmat):
  #initialize an empty utterance vector of the same length as word2vec vectors
  seqvec = np.zeros((vec_dims,))
  #counter for number of words in a specific utterance
  wordcount = 1
  #we iterate over the 2000 possible words in a given utterance
  wordind = 1
  while (wordind < len(datarow)):
    #the tf-idf weight is saved in the cells of datarow
    tfidfweight = datarow[wordind]
    if not tfidfweight is None:
      wordembed = tfidfweight * embedmat[wordind,]
      seqvec = seqvec + normalize(wordembed)
      wordcount = wordcount + 1
    wordind = wordind + 1
  return seqvec/wordcount

# go through the matrix and embed each utterances
def embed_utts(sequences,embedmat):
  vecseq = [seq2vec(seq,embedmat)for seq in sequences]
  return vecseq

# create utterance embeddings
train_embedutts = embed_utts(train_sequences,embedding_matrix)
val_embedutts = embed_utts(val_sequences,embedding_matrix)
test_embedutts = embed_utts(test_sequences,embedding_matrix)


def gof_spear(X,Y):
    #spearman correlation of columns (schemas)
    gof_spear = np.zeros(X.shape[1])    
    for schema in range(9):
        rho,p = scipy.stats.spearmanr(X[:,schema],Y[:,schema])
        gof_spear[schema]=rho
    return gof_spear

def bootstrap(iterations,sample_size, sample_embeds, sample_labels,classification,model):
    stats = np.zeros((iterations,9))
    for l in range(iterations):
        # prepare bootstrap sample
        bootstrap_sample_indx = random.sample(list(enumerate(sample_embeds)), sample_size)
        bootstrap_sample_utts = [sample_embeds[i] for (i,j) in bootstrap_sample_indx]
        bootstrap_sample_labels = [sample_labels[i] for (i,j) in bootstrap_sample_indx]
        # evaluate model
        if model=="knn":
            model_gof=my_kNN(np.array(bootstrap_sample_utts),np.array(bootstrap_sample_labels),classification)
        elif model=="svm":
            model_gof=my_svm(np.array(bootstrap_sample_utts),np.array(bootstrap_sample_labels),classification)
        elif model=="rnn":
            model_gof=my_rnn_fixed(np.array(bootstrap_sample_utts),np.array(bootstrap_sample_labels),classification)
        stats[l,:] = model_gof
    # confidence intervals
    cis = np.zeros((2,9))
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    cis[0,:] = [max(0.0, np.percentile(stats[:,i], p)) for i in range(9)]
    p = (alpha+((1.0-alpha)/2.0)) * 100
    cis[1,:] = [min(1.0, np.percentile(stats[:,i], p)) for i in range(9)]
    return cis

# configure bootstrap
n_iterations = 100
n_size = int(len(val_text) * 0.75)

# cosine distance
def cosine_dist(X,Y):
    return scipy.spatial.distance.cosine(X,Y)
    
#kNN algorithm
def knn_custom(train_X,test_X,train_y,test_y,k,dist,classification):
    #empty array to collect the results (should have shape of samples to classify)
    votes = np.zeros(test_y.shape)
    #fit the knn
    knn=NearestNeighbors(k, metric=dist)
    knn.fit(train_X)
    #collect neighbors
    i=0 # index to collect votes of the neighbors
    for sample in test_X:
        neighbors=knn.kneighbors([sample],k,return_distance=False)[0]
        if classification:
            output_y = np.zeros((k,test_y.shape[1]))
            j=0
            for neighbor in neighbors:
                output_y[j,:] = train_y[neighbor,:]
                j=j+1
            votes[i,:] = stats.mode(output_y,nan_policy='omit')[0]
        else:
            output_y = np.zeros(test_y.shape[1])
            for neighbor in neighbors:
                output_y += train_y[neighbor,:]
                votes[i,:] = np.divide(output_y,k)
        i=i+1
    return votes


# weighting model output (spearman correlations) by schema frequencies in training set and returning mean over schemas
def performance(train_y,output):
    train_y = np.array(train_y)
    train_y[train_y>0]=1
    weighting = train_y.sum(axis=0)/train_y.shape[0]
    perf = output * weighting
    return np.nanmean(np.array(perf), axis=0)

# finding best k by testing some values for k
def find_k(train_X, test_X, train_y, test_y, dist, classification):
    perf = 0
    best_k = 0
    for k in ks:
        knn_k = knn_custom(train_X, test_X, train_y, test_y,k,dist,classification)
        knn_gof_spear = gof_spear(knn_k,test_y)
        print('Results for choice of k is %s.' % k)
        print(pd.DataFrame(data=knn_gof_spear,index=schemas,columns=['gof']))
        if perf < performance(train_y,knn_gof_spear):
            perf = performance(train_y,knn_gof_spear)
            best_k = k
    return best_k


knn_class_k = find_k(train_embedutts,val_embedutts,train_labels,val_labels,cosine_dist,1)
print('Best choice for classification k is: %s' % knn_class_k)

knn_reg_k = find_k(train_embedutts,val_embedutts,train_labels,val_labels,cosine_dist,0) 
print('Best choice for regression k is: %s' % knn_reg_k)


def my_kNN(test_X,test_y,classification):
    if classification:
        my_knn=knn_custom(train_embedutts,test_X,train_labels,test_y,knn_class_k,cosine_dist,1)
    else:
        my_knn=knn_custom(train_embedutts,test_X,train_labels,test_y,knn_reg_k,cosine_dist,0)
    return my_knn, test_y


output_kNN_class, test_y = my_kNN(test_embedutts,test_labels,1)
output_kNN_reg, test_y = my_kNN(test_embedutts,test_labels,0)

print('KNN Classification Prediction')
print(pd.DataFrame(data=output_kNN_class,index=schemas,columns=['estimate']))

print('KNN Regression Prediction')
print(pd.DataFrame(data=output_kNN_reg,index=schemas,columns=['estimate']))


#SVM
'''
Support vector machine

    The second algorithm we chose are support vector machines (SVMs). Again, we train both a support vector classification (SVC) and a support vectore regression (SVR). We only try all three types of standard kernels and do not do any additional parameter tuning. Just like the kNN, the support vector machine takes as input the utterances encoded as averages of word vectors. Support vector classification and regression do not allow for multilabel output. We therefore train disjoint models, one for each schema.
    For both types of SVM, we first transform the input texts as the algorithm expects normally distributed input centered around 0 and with a standard deviation of 1.
'''

def svm_scaler(train_X):
        #scale the data
        scaler_texts = StandardScaler()
        scaler_texts = scaler_texts.fit(train_X)
        return scaler_texts

scaler_texts = svm_scaler(train_embedutts)

def svm_custom(train_X,train_y,text_scaler,kern,classification):
        models=[]
        train_X = text_scaler.transform(train_X)
        #fit a new support vector regression for each schema
        for schema in range(9):
            if classification:
                model = svm.SVC(kernel=kern)
            else:
                model = svm.SVR(kernel=kern)
            model.fit(train_X, train_y[:,schema])
            models.append(model)
        return models
        
def svm_predict(svm_models,test_X,train_y,test_y,text_scaler):
    #empty array to collect the results (should have shape of samples to classify)
    votes = np.zeros(test_y.shape)
    for schema in range(9):
        svm_model=svm_models[schema]
        prediction = svm_model.predict(text_scaler.transform(test_X))
        votes[:,schema] = prediction
    out = votes
    gof = gof_spear(out,test_y)
    perf = performance(train_y,gof)
    return out,perf
   

svr_rbf_models =  svm_custom(train_embedutts,train_labels,scaler_texts,'rbf',0)
svr_rbf_out, svr_rbf_perf = svm_predict(svr_rbf_models,val_embedutts,train_labels,val_labels,scaler_texts)
svr_lin_models = svm_custom(train_embedutts,train_labels,scaler_texts,'linear',0)
svr_lin_out, svr_lin_perf = svm_predict(svr_lin_models,val_embedutts,train_labels,val_labels,scaler_texts)
svr_poly_models = svm_custom(train_embedutts,train_labels,scaler_texts,'poly',0)
svr_poly_out, svr_poly_perf = svm_predict(svr_poly_models,val_embedutts,train_labels,val_labels,scaler_texts)

print(pd.DataFrame(data=[svr_rbf_perf,svr_lin_perf,svr_poly_perf],index=['rbf','lin','poly'],columns=['svr']))

svm_rbf_models =  svm_custom(train_embedutts,train_labels,scaler_texts,'rbf',1)
svm_rbf_out, svm_rbf_perf = svm_predict(svm_rbf_models,val_embedutts,train_labels,val_labels,scaler_texts)
svm_lin_models = svm_custom(train_embedutts,train_labels,scaler_texts,'linear',1)
svm_lin_out, svm_lin_perf = svm_predict(svm_lin_models,val_embedutts,train_labels,val_labels,scaler_texts)
svm_poly_models = svm_custom(train_embedutts,train_labels,scaler_texts,'poly',1)
svm_poly_out, svm_poly_perf = svm_predict(svm_poly_models,val_embedutts,train_labels,val_labels,scaler_texts)

print(pd.DataFrame(data=[svm_rbf_perf,svm_lin_perf,svm_poly_perf],index=['rbf','lin','poly'],columns=['svm']))

def my_svm(test_X,test_y,classification):
    if classification:
        my_svm_out, my_svm_perf=svm_predict(svm_rbf_models,test_X,train_labels,test_y,scaler_texts)
    else:
        my_svm_out, my_svm_perf=svm_predict(svr_rbf_models,test_X,train_labels,test_y,scaler_texts)
    return gof_spear(my_svm_out,test_y)

output_SVC = my_svm(test_embedutts,test_labels,1)
output_SVR = my_svm(test_embedutts,test_labels,0)

print('SVM Classification Prediction')
print(pd.DataFrame(data=output_SVC,index=schemas,columns=['estimate']))

print('SVM Regression Prediction')
print(pd.DataFrame(data=output_SVR,index=schemas,columns=['estimate']))

# bootstrap confidence intervals for SVR and SVC
bs_svc = bootstrap(n_iterations,n_size,test_embedutts,test_labels,1,"svm")
bs_svr = bootstrap(n_iterations,n_size,test_embedutts,test_labels,0,"svm")

print(f'SVM Classification 95% Confidence Intervals')
print(pd.DataFrame(data=np.transpose(bs_svc),index=schemas,columns=['low','high']))

print(f'SVM Regression 95% Confidence Intervals')
print(pd.DataFrame(data=np.transpose(bs_svr),index=schemas,columns=['low','high']))


#RNN
'''
Training Multilabel RNN

    We used as inspiration for the architecture of all RNNs the paper: Kshirsagar, R., Morris, R., & Bowman, S. (2017). Detecting and explaining crisis. arXiv preprint arXiv:1705.09585. However, we used long short-term memory (LSTM) instead of a gated recurrent unit (GRU).
'''

def multilabel_model(train_X, train_y, test_X, test_y,params):
    # build the model
    model = Sequential()
    e = Embedding(vocab_size, vecdims, weights=[embedding_matrix], input_length=max_length, trainable=False)
    #embedding layer
    model.add(e)
    #LSTM layer
    model.add(Bidirectional(LSTM(params['lstm_units'])))
    #dropout layer
    model.add(Dropout(params['dropout']))
    #output layer
    model.add(Dense(9, activation='sigmoid'))
    # compile the model
    model.compile(optimizer=params['optimizer'], loss=params['losses'], metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    out = model.fit(train_X, train_y, 
                    validation_data=[test_X,test_y],
                    batch_size=params['batch_size'], 
                    epochs=params['epochs'], 
                    verbose=0)
    return out, model

def grid_search(train_X, test_X, train_y, test_y):
    #define hyperparameter grid
    p={'lstm_units':[50,100],
       'optimizer':['rmsprop','Adam'],
       'losses':['binary_crossentropy','categorical_crossentropy','mean_absolute_error'],
       'dropout':[0.1,0.5],
       'batch_size': [32,64],
       'epochs':[100]} 
    #scan the grid
    tal=talos.Scan(x=train_X,
                   y=train_y,
                   x_val=test_X,
                   y_val=test_y,
                   model=multilabel_model,
                   params=p,
                   experiment_name='multilabel_rnn',
                   print_params=True,
                   clear_session=True)
    return tal

'''
# wall time to run grid search: ~ 2h 10min
#run the small grid search
tal = grid_search(padded_train, padded_validate, train_labels, val_labels)
#analyze the outcome
analyze_object=talos.Analyze(tal)
analysis_results = analyze_object.data
#let's have a look at the results of the grid search
print('Grid Search RNN:')
print(analysis_results)

#we choose the best model of the grid search on the basis of the MAE metric, lower values are better
mlm_model = tal.best_model(metric='mean_absolute_error', asc=True)
#to get an idea of how our best model performs, we check predictions on the validation set
prediction_mlm_val = mlm_model.predict(padded_validate)
output_mlm_val = gof_spear(prediction_mlm_val,val_labels)

print('RNN MLM:')
print(pd.DataFrame(data=output_mlm_val,index=schemas,columns=['estimate']))

#the predictions make sense considering what we got from KNN and SVM. We deploy the model.
# talos.Deploy(tal,'mlm_rnn',metric='mean_absolute_error',asc=True)
'''

'''
#we restore the deployed Talos experiment
restore = talos.Restore('./mlm_rnn.zip')
#to get the best performing parameters, we get the results of the Talos experiment
scan_results = restore.results

# select the row with the smallest mean absolute error
print(scan_results[scan_results.mean_absolute_error == scan_results.mean_absolute_error.min()]) 
'''

def mlm_fixed(train_X, train_y, test_X, test_y):
    # build the model
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    #embedding layer
    model.add(e)
    #LSTM layer
    model.add(Bidirectional(LSTM(100)))
    #dropout layer
    model.add(Dropout(0.1))
    #output layer
    model.add(Dense(9, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    out = model.fit(train_X, train_y, 
                    validation_data=[test_X,test_y],
                    batch_size=32, 
                    epochs=100, 
                    verbose=0)
    return out, model
'''
# wall time to run: ~ 1h 54min
for i in range(30):
    #we train the model
    res, model = mlm_fixed(padded_train, train_labels, padded_validate, val_labels)
    #we save models to files to free up working memory
    model_name = 'Data/MLMs/mlm_' + str(i)
    model.save(model_name + '.h5')
'''

'''
per schema RNN

Training Per-Schema RNNs

    We also train separate RNNs per schema. For this, we can use the output layer to compute a probability for each of the four possible labels. This way, the labels are treated as separate classes. We take over the parameter values from the multilabel model for the number of LSTM units, the dropout rate, the loss function, the evaluation metric, the batch size, and the number of epochs. To obtain the probability for each class, the units of the output layer have a softmax activation function. For the evaluation, the class with the highest probability is chosen per model. The resulting models are written to files and loaded again for prediction.

'''

# generate predictions with the per-schema models
def predict_schema_mlm(test_text, test_labels,fixed=None):
    if fixed is None:
        all_preds = np.zeros((test_labels.shape[0],test_labels.shape[1],30))
        all_gofs = np.zeros((30,9))
        for j in range(30):
            model_name = "Data/MLMs/mlm_" + str(j)
            model = keras.models.load_model(model_name + '.h5')
            preds = model.predict(test_text)
            gofs = gof_spear(preds,test_labels)
            all_preds[:,:,j] = preds
            all_gofs[j,:] = gofs
    else:
        model_name = "Data/MLMs/mlm_" + str(fixed)
        model = keras.models.load_model(model_name + '.h5')
        all_preds = model.predict(test_text)
        all_gofs = gof_spear(all_preds,test_labels)
    return all_gofs,all_preds
    
    
# define separate models
def perschema_models(train_X, train_y, test_X, test_y):
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    # compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['mean_absolute_error'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(train_X, train_y,
              validation_data=[test_X,test_y],
              batch_size=32, 
              epochs=100, 
              verbose=0)
    out=model.predict(test_X)
    gof,p=scipy.stats.spearmanr(out,test_y,axis=None)
    return gof, model

'''
# wall time to run: ~ 16h
#train models
for j in range(30):
    directory_name = "Data/PSMs/per_schema_models_" + str(j)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    for i in range(9):
        train_label_schema = np_utils.to_categorical(train_labels[:,i])
        val_label_schema = np_utils.to_categorical(val_labels[:,i])
        val_output_slm, model = perschema_models(padded_train,train_label_schema,padded_validate,val_label_schema)
        #we write trained models to files to free up working memory
        model_name = '/schema_model_' + schemas[i]
        save_model_under = directory_name + model_name
        model.save(save_model_under + '.h5')
'''

#load single models
def load_single_models(directory):
    single_models = []
    for i in range(9):
        model_name ='/schema_model_' + schemas[i]
        get_from = directory + model_name
        model = keras.models.load_model(get_from + '.h5')
        single_models.append(model)
    return single_models

#generate predictions with the per-schema models
def predict_schema_psm(test_text, test_labels,fixed=None):
    if fixed is None:
        all_preds = np.zeros((test_labels.shape[0],test_labels.shape[1],30))
        all_gofs = np.zeros((30,9))
        for j in range(30):
            directory_name = "Data/PSMs/per_schema_models_" + str(j)
            preds = np.zeros(test_labels.shape)
            gofs=[]
            single_models = load_single_models(directory_name)
            for i in range(9):
                model = single_models[i]
                out = model.predict(test_text)
                out = out.argmax(axis=1)
                preds[:,i] = out
                gof,p=scipy.stats.spearmanr(out,test_labels[:,i])
                gofs.append(gof)
            all_preds[:,:,j] = preds
            all_gofs[j,:] = gofs
    else:
        directory_name= "Data/PSMs/per_schema_models_" + str(fixed)
        all_preds = np.zeros(test_labels.shape)
        all_gofs = []
        single_models = load_single_models(directory_name)
        for i in range(9):
            model = single_models[i]
            out = model.predict(test_text)
            out = out.argmax(axis=1)
            all_preds[:,i] = out
            gof,p=scipy.stats.spearmanr(out,test_labels[:,i])
            all_gofs.append(gof)
    return all_gofs,all_preds    

def my_rnn(test_X,test_y,single):
    if single:
        gof,preds=predict_schema_psm(test_X,test_y)
    else:
        gof,preds=predict_schema_mlm(test_X,test_y)
    #make a sum of all classification values
    gof_sum = np.sum(gof,axis=1)
    #sort sums
    gof_sum_sorted = np.sort(gof_sum)
    #pick element that is closest but larger than median (we have even number of elements)
    get_med_element = gof_sum_sorted[15]
    #get index of median
    gof_sum_med_idx = np.where(gof_sum==get_med_element)[0]
    #choose this as the final model to use in H2 and to report in the paper
    gof_out = gof[gof_sum_med_idx]
    return np.transpose(gof_out),gof_sum_med_idx

'''
# wall time to run: ~ 6min
# predicting testset with multilabel model
output_mlm,idx_mlm = my_rnn(padded_test,test_labels,0)
# predicting testset with perschema models
output_psm,idx_psm = my_rnn(padded_test,test_labels,1)        
              
print('RNN Multilabel Model Testset Output')
print(pd.DataFrame(data=output_mlm,index=schemas,columns=['estimate']))

print('RNN Per-Schema Testset Output')
print(pd.DataFrame(data=output_psm,index=schemas,columns=['estimate']))
'''

def my_rnn_fixed(test_X,test_y,single):
    if single:
        gof,preds=predict_schema_psm(test_X,test_y,idx_psm[0])
    else:
        gof,preds=predict_schema_mlm(test_X,test_y,idx_mlm[0])
    return gof
    
'''
# wall time to run: ~ 37min
#bootstrapping the 95% confidence intervals
bs_mlm = bootstrap(n_iterations,n_size,padded_test,test_labels,0,"rnn")
bs_psm = bootstrap(n_iterations,n_size,padded_test,test_labels,1,"rnn")

print(f'Multilabel RNN Classification 95% Confidence Intervals')
print(pd.DataFrame(data=np.transpose(bs_mlm),index=schemas,columns=['low','high']))
print(f'Per-Schema RNN Classification 95% Confidence Intervals')
print(pd.DataFrame(data=np.transpose(bs_psm),index=schemas,columns=['low','high']))

output_psm_flat = [item for sublist in output_psm for item in sublist]
output_mlm_flat = [item for sublist in output_mlm for item in sublist]

print(f'Estimates of all models')
outputs = np.concatenate((output_kNN_class,output_kNN_reg,output_SVC, output_SVR, output_psm_flat, output_mlm_flat))
outputs=np.reshape(outputs,(9,6),order='F')
print(pd.DataFrame(data=outputs,index=schemas,columns=['kNN_class','kNN_reg','SVC','SVR','PSM','MLM']))

print(f'Lower CIs of all models')
lower_cis = np.concatenate((bs_knn_class[0],bs_knn_reg[0],bs_svc[0], bs_svr[0], bs_psm[0], bs_mlm[0]))
lower_cis=np.reshape(lower_cis,(9,6),order='F')
print(pd.DataFrame(data=lower_cis,index=schemas,columns=['kNN_class','kNN_reg','SVC','SVR','PSM','MLM']))

print(f'Upper CIs of all models')
upper_cis = np.concatenate((bs_knn_class[1],bs_knn_reg[1],bs_svc[1], bs_svr[1], bs_psm[1], bs_mlm[1]))
upper_cis=np.reshape(upper_cis,(9,6),order='F')
print(pd.DataFrame(data=upper_cis,index=schemas,columns=['kNN_class','kNN_reg','SVC','SVR','PSM','MLM']))
'''


'''
H2
'''

pdb.set_trace()
gofH2,predsH2=predict_schema_psm(padded_test,test_labels,idx_psm[0])

predsH2 = predsH2.astype(int)
print(predsH2[:,0:5])
diag_rho = [scipy.stats.spearmanr(predsH2[i,:], test_labels[i,0:9], nan_policy='omit')[0] for i in range(predsH2.shape[0])]

df_predsH2 = pd.DataFrame(data=predsH2,columns=['AttachPred','CompPred',"GlobalPred","HealthPred","ControlPred","MetaCogPred","OthersPred","HopelessPred","OthViewsPred"])
df_predsH2["Corr"] = pd.DataFrame(diag_rho)

print(df_predsH2.head())

df_predsH2.to_csv("Data/PredictionsH2.csv", sep=';', header=True, index=False, mode='w')








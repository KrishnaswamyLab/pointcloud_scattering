import numpy as np
import pickle
from collections import defaultdict
import scipy.stats.mstats
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.svm import SVC
from scipy import sparse
from collections import Counter

# This file written by Feng Gao, github.com/FengGmsu/manifold

def cross_validate(split_size,train_all_feature,train_all_Y,test_feature,test_Y,G_pool,C_pool):
    results = []
    train_idx,val_idx = Kfold(len(train_all_feature),split_size)
    prediction = []
    
    
    test_feature = np.reshape(test_feature,(len(test_feature),len(test_feature[0])))
    
    
    for k in range(split_size):
        train_feature = [train_all_feature[i] for i in train_idx[k]]
        train_Y = [train_all_Y[i] for i in train_idx[k]]
        val_feature = [train_all_feature[i] for i in val_idx[k]]
        val_Y = [train_all_Y[i] for i in val_idx[k]]
        
        
        train_feature = np.reshape(train_feature,(len(train_feature),len(train_feature[0])))
        val_feature = np.reshape(val_feature,(len(val_feature),len(val_feature[0])))
        
    
        print('inner_loop',k,'start')
        
        print('start best para search')
        test_score,preds,best_c,best_g = run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y)
        print('best c is',best_c)
        print('best g is',best_g)
        results.append(test_score)
        prediction.append(preds)
        print('this run accuracy is', results[-1])
        print('inner_loop',k,'ends')
    
    prediction = np.array(prediction)
    pre = []
    for i in range(prediction.shape[1]):
        pre.append(Counter(prediction[:,i]).most_common(1)[0][0])
    test_acc = np.mean(np.equal(pre,test_Y))
    return (results,test_acc)




def run_train(train_feature,train_Y,G_pool,C_pool,val_feature,val_Y,test_feature,test_Y):
    temp = 0
    for c in C_pool:
        for g in G_pool:
            model = SVC(kernel='rbf',C=c,gamma=g)
            model.fit(train_feature,train_Y)
            score = model.score(val_feature,val_Y)
            if score >temp:
                temp =score
                test_score = model.score(test_feature,test_Y)
                preds = model.predict(test_feature)
                best_c = c
                best_g = g
    return (test_score,preds,best_c,best_g)
#def write_to_file(index):
    #with open('shuffle_index_0.txt','w') as file:
        #for i in index:
            #file.write(str(i))
            #file.write('\t')


def Kfold(length,fold):
    size = np.arange(length).tolist()
    train_index = []
    val_index = []
    rest = length % fold
    fold_size = int(length/fold)
    temp_fold_size = fold_size
    for i in range(fold):
        temp_train = []
        temp_val = []
        if rest>0:
            temp_fold_size = fold_size+1
            rest = rest -1
            temp_val = size[i*temp_fold_size:+i*temp_fold_size+temp_fold_size]
            temp_train = size[0:i*temp_fold_size] + size[i*temp_fold_size+temp_fold_size:]
        else:
            temp_val = size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size
                            :(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size]
            temp_train = size[0:(length % fold)*temp_fold_size+(i-(length % fold))*fold_size] + size[(length % fold)*temp_fold_size+(i-(length % fold))*fold_size+fold_size:]
        train_index.append(temp_train)
        val_index.append(temp_val)
    return (train_index,val_index)

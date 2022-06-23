import numpy as np
from scipy import sparse
import scipy.stats.mstats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from utility import *

def read_label_shape(i):
    return i%10
    
features = []
Y = []
for i in range(100):
    if i == 90:
        continue
    temp = np.loadtxt('FAUST_FEATURES/' + 'feature' + str(i) + '.txt')
    features.append(temp)
    Y.append(read_label_shape(i))
    
features = np.vstack(features)  
features_z = scipy.stats.mstats.zscore(features, 0)
this_max = np.amax(features_z[~np.isinf(features_z)])
features_z[np.isinf(features_z)] = this_max
imp = SimpleImputer(missing_values=np.nan,strategy='mean')


G_pool = [0.000001, 0.000005, 0.00001,0.0001,0.001, 0.01, 0.1]
C_pool = [25,100,150, 250,500, 750, 1000, 1250]

# 80/20 train/test split with 10-fold cross-validation
n_splits = 10
X_train, X_test, y_train, y_test = train_test_split(features_z, Y, test_size=0.2)
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)
result, prediction_acc = cross_validate(n_splits, X_train, y_train, X_test, y_test, G_pool, C_pool)
print("Cross-validation result: %s\n" % result)
print("Prediction accuracy: %f\n" % prediction_acc)



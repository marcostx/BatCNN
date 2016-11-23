import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,accuracy_score, recall_score, precision_score
import scipy
from random import shuffle

def load_dataset(filename):
    f = open(filename)
    x = []
    y = []
    for line in f:
        v = line.rstrip('\n').split(',')
        vf = [float(i) for i in v[:-1]]
        x.append(vf)
        y.append(float(v[-1]))
    return x,y


def inductor(x,y):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 8),  max_iter=1000,random_state=1)

    clf.fit(x,y)

    return clf


if __name__ == '__main__':
    fname = sys.argv[1]

    print("loading data ..")
    x,y = load_dataset(fname)
    x = np.array(x)
    y = np.array(y) 
    n = len(x)
    kf = StratifiedKFold(n_splits=3, shuffle=True)

    for train_index, test_index in kf.split(x,y):

        shuffle(train_index)
        shuffle(test_index)
        xtrain = x[train_index]
        ytrain = y[train_index]
        xtest  = x[test_index]
        ytest  = y[test_index]

        print("training ...")
        clf = inductor(xtrain,ytrain)
        
        print("predicting ...")
        ypred  =  clf.predict(xtest)
        print "(accuracy : %4.3f) "%(accuracy_score(ytest,ypred))
        print "(f1 :  %4.3f) "%(f1_score(ytest,ypred, average='micro'))
        print "(recall : %4.3f) "%(recall_score(ytest,ypred,average='micro'))
        print "(precision : %4.3f) "%(precision_score(ytest,ypred,average='micro'))

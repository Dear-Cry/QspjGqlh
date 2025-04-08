import pickle
import numpy as np
import os

def load_a_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

        X = data['data']
        y = data['labels']

        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)

        return X, y

def load_train(dir):
    Xs = []
    ys = []
    for i in range(1, 6):
        X, y = load_a_batch(os.path.join(dir, 'data_batch_%d' % (i, )))
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    return X_train, y_train

def load_test(dir):
    X_test, y_test = load_a_batch(os.path.join(dir, 'test_batch'))
    return X_test, y_test

def filter_by_class(X, y, class_num):
    X = X[np.isin(y, range(class_num))]
    y = y[np.isin(y, range(class_num))]
    return X, y

def train_validation_split(X_train, y_train, train_num, validation_num):
    X_val = X_train[: validation_num]
    y_val = y_train[: validation_num]
    X_train = X_train[validation_num : validation_num + train_num]
    y_train = y_train[validation_num : validation_num + train_num]
    print("Training set class distribution:", np.unique(y_train, return_counts=True))
    print("Validation set class distribution:", np.unique(y_val, return_counts=True))
    return X_train, y_train, X_val, y_val

def standardlization(X):
    X.astype(np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / (std + 1e-8)
    return X

def flatten(X):
    X = np.reshape(X, newshape=(X.shape[0], -1))
    return X


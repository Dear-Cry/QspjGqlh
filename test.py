import numpy as np
import os

import mlp
import preprocess

# Load the data
DIR = 'data/cifar10/' 
cifar10_path = os.path.join(DIR, 'cifar-10-batches-py')
X_test, y_test = preprocess.load_test(cifar10_path)

class_num = 10
test_num = 10000

X_test, y_test = preprocess.filter_by_class(X_test, y_test, class_num)
X_test = preprocess.standardlization(X_test)
X_test = preprocess.flatten(X_test)

model = mlp.MLP()
model.load_model_without_BN(r'.\best_models\best_model.pickle')

model.set_mode('test')
y_pred = model.predict(X_test)
accuracy_on_test = np.mean(y_test == y_pred)
print(f"Accuracy on test set: {accuracy_on_test:.4f}")
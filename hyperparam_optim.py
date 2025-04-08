import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

import mlp
import optimizer
import preprocess

# Load the data
cifar10_path = os.path.join('data/cifar10/', 'cifar-10-batches-py')
X_train, y_train = preprocess.load_train(cifar10_path)

# Data preprocessing
np.random.seed(0)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_num = 10
classes = classes[: class_num]
validation_num = 10000
train_num = 5000 * class_num - validation_num

X_train, y_train = preprocess.filter_by_class(X_train, y_train, class_num)
X_train, y_train, X_val, y_val = preprocess.train_validation_split(X_train, y_train, train_num, validation_num)
X_train = preprocess.standardlization(X_train)
X_val = preprocess.standardlization(X_val)
X_train = preprocess.flatten(X_train)
X_val = preprocess.flatten(X_val)
print("Shape of training set:", X_train.shape)
print("Shape of validation set:", X_val.shape)

param_grid = {
    'lr': [0.02],
    'h1': [256],
    'h2': [64],
    'reg': [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
}

all_combinations = [dict(zip(param_grid.keys(), values)) 
                    for values in itertools.product(*param_grid.values())]

P = X_train.shape[1]
C = class_num
act_funcs = ['relu', 'relu']
results = []

for params in all_combinations:
    print(f"\nTraining with params: {params}")
    
    h1, h2 = params['h1'], params['h2']
    layer_dims = [P, h1, h2, C]
    regs = [params['reg']] * 3 
    
    model = mlp.MLP(layer_dims, act_funcs, regs=regs)
    opti = optimizer.SGD(model.parameters, lr=params['lr'])
    scheduler = optimizer.MultiStepLR(opti, milestones=[300, 500, 750, 1000], gamma=0.5)
    
    epochs = 1000 
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        AL = model.forward(X_train)
        train_loss, dAL = model.cross_entropy_loss(AL, y_train)
        grads = model.backward(dAL, y_train)
        opti.update(grads)
        scheduler.step()
        
        val_pred = model.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    train_pred = model.predict(X_train)
    train_acc = np.mean(train_pred == y_train)
    
    results.append({
        'params': params,
        'train_acc': train_acc,
        'val_acc': best_val_acc
    })

param_labels = [f"lr={p['lr']}\nh1={p['h1']}\nh2={p['h2']}\nreg={p['reg']}" 
                for p in [res['params'] for res in results]]
train_accs = [res['train_acc'] for res in results]
val_accs = [res['val_acc'] for res in results]

plt.figure(figsize=(20, 8))
x = np.arange(len(param_labels))
plt.bar(x - 0.2, train_accs, width=0.4, label='Training Accuracy')
plt.bar(x + 0.2, val_accs, width=0.4, label='Validation Accuracy')
plt.xticks(x, param_labels, rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('Hyperparameter Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("hyperparameter_comparison.png")
plt.show()

best_result = max(results, key=lambda x: x['val_acc'])
print(f"\nBest Params: {best_result['params']}, Val Acc: {best_result['val_acc']:.4f}")
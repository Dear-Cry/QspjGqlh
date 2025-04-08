import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import download
import preprocess
import visualize
import mlp
import optimizer


# Download CIFAR-10 dataset
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DIR = 'data/cifar10/' 
download.download_CIFAR10(URL, DIR)

# Load the data
cifar10_path = os.path.join(DIR, 'cifar-10-batches-py')
X_train, y_train = preprocess.load_train(cifar10_path)

# Data preprocessing
np.random.seed(0)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
visualize.show_some_examples(X_train, y_train, classes, 4)
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

# Create a mlp 
P = X_train.shape[1]
H1 = 256
H2 = 64
C = class_num
layer_dims = [P, H1, H2, C]
act_funcs = ['relu', 'relu']
use_bn = [True, True]
regs = [0.1, 0.1, 0.1]
model = mlp.MLP(layer_dims, act_funcs, regs=regs, use_bn=use_bn)
opti = optimizer.SGD(model.parameters, lr = 0.005)
scheduler = optimizer.MultiStepLR(opti, milestones=[500, 800, 1200, 1600, 1800], gamma=0.5)

# Train
model.set_mode('train')
save_dir=r'./best_models'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
best_acc = 0.0
tolerance = 0.007
epochs = 1600
print_every = 50
train_losses = []
val_losses = []
val_accuracies = []
best_val_loss = float('inf')
patience = 30 
no_improve_epochs = 0
batch_size = 64
for epoch in range(epochs):
    model.set_mode('train')
    np.random.seed(epoch + 42)  
    idx = np.random.permutation(X_train.shape[0])  
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train[idx]
    
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        AL = model.forward(X_batch)
        train_loss, dAL = model.cross_entropy_loss(AL, y_batch)
        grads = model.backward(dAL, y_batch)
        opti.update(grads)  

    scheduler.step()

    train_losses.append(train_loss)

    model.set_mode('test')
    AL_val = model.forward(X_val)
    val_loss, _ = model.cross_entropy_loss(AL_val, y_val)
    val_losses.append(val_loss)
    val_pred = model.predict(X_val)
    val_acc = np.mean(val_pred == y_val)
    val_accuracies.append(val_acc)
    model.set_mode('train')

    if epoch % print_every == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc + tolerance > best_acc:
            save_path = os.path.join(save_dir, 'best_model.pickle')
            model.save_model(save_path)
            print(f"best accuracy performence has been updated: {best_acc:.5f} --> {val_acc:.5f}")
            best_acc = val_acc

model.set_mode('test')
y_pred = model.predict(X_train)
accuracy_on_test = np.mean(y_train == y_pred)
print(f"Final Train Accuracy: {accuracy_on_test:.4f}")

plt.figure(figsize=(12, 8))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("TVLEntiredata.png")

plt.figure(figsize=(12, 8))
plt.plot(val_accuracies, 'g-', label='Validation Accuracy')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("VAEntiredata.png")

plt.show()

visualize.plot_weight_histogram(model, layer=1)
visualize.plot_weight_histogram(model, layer=2)
visualize.plot_weight_histogram(model, layer=3)

visualize.plot_weights_as_images(model, layer=1, n_cols=16)





import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_some_examples(X, y, classes, n=4):
    rand_idices = np.random.randint(0, X.shape[0], size=n)
    X_show = X[rand_idices]
    y_show = y[rand_idices]
    images = np.hstack(np.array([np.asarray(np.reshape(x, (32, 32, 3)), dtype=np.int64) for x in X_show]))
    label_texts = [classes[label] for label in y_show]
    title = " | ".join(label_texts) 
    plt.imshow(images)
    plt.title(title)  
    plt.show()

def plot_weight_histogram(model, layer=1):
    W = model.parameters[f'W{layer}']
    plt.hist(W.flatten(), bins=50, alpha=0.7)
    plt.title(f"Layer {layer} Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"W{layer}_histogram.png")

def plot_weights_as_images(model, layer=1, n_cols=16):
    W = model.parameters[f'W{layer}']
    weights = W.reshape(-1, 32, 32, 3) 
    
    n_plots = W.shape[0]
    n_rows = n_plots // n_cols
    
    plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(n_plots):
        plt.subplot(n_rows, n_cols, i+1)
        img = (weights[i] - weights[i].min()) / (weights[i].max() - weights[i].min())
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Layer {layer} Weight Visualization")
    plt.savefig(f"W{layer}_image_patches.png")



import numpy as np
import matplotlib.pyplot as plt
import os

FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def _ensure_dir(path):
    if path:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

def calculate_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"训练曲线已保存至 {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=FASHION_MNIST_CLASSES, save_path=None):
    cm = compute_confusion_matrix(y_true, y_pred, len(classes))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix on Test Set')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)
    plt.show()

def visualize_weights(weights, num_neurons=16, save_path=None, cmap='RdBu'):
    W_T = weights.T 
    num_neurons = min(num_neurons, W_T.shape[0])
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    title_suffix = "(RdBu Heatmap)" if cmap == 'RdBu' else "(Grayscale)"
    plt.suptitle(f"First Layer Weights {title_suffix}", fontsize=16)
    
    for i in range(num_neurons):
        plt.subplot(grid_size, grid_size, i + 1)
        weight_img = W_T[i].reshape(28, 28)
        
        vmax = np.abs(weight_img).max()
        plt.imshow(weight_img, cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.axis('off')
        plt.title(f'Neuron {i+1}')
        
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"权重可视化已保存至 {save_path}")
    plt.show()

def visualize_error_cases(X_test, y_true, y_pred, classes=FASHION_MNIST_CLASSES, num_samples=30, save_path=None):
    errors = np.where(y_pred != y_true)[0]
    
    if len(errors) == 0:
        print("！？强 强？！没有分错的样本！")
        return
        
    print(f"在测试集中共找到 {len(errors)} 个分类错误的样本。")
    num_samples = min(num_samples, len(errors))
    
    #固定选取错例的随机种子
    np.random.seed(42) 
    selected_indices = np.random.choice(errors, num_samples, replace=False)
    
    cols = 6
    rows = int(np.ceil(num_samples / cols))
    
    plt.figure(figsize=(3 * cols, 3 * rows))
    plt.suptitle(f"Error Analysis: {num_samples} Misclassified Samples", fontsize=20, y=1.02)
    
    for i, idx in enumerate(selected_indices):
        plt.subplot(rows, cols, i + 1)
        
        img = X_test[idx].reshape(28, 28)
        true_label = classes[y_true[idx]]
        pred_label = classes[y_pred[idx]]
        
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"大规模错例分析图已保存至 {save_path}")
    plt.show()
    
import numpy as np
import gzip
import os
import urllib.request

def download_fashion_mnist(data_dir='data'):
    #默认使用 AWS S3 源。如果测试环境网络受限，可手动将文件放入 data/ 目录即可跳过下载。
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    
    files = {
        'train-images-idx3-ubyte.gz': 26421880,
        'train-labels-idx1-ubyte.gz': 29515,
        't10k-images-idx3-ubyte.gz': 4422102,
        't10k-labels-idx1-ubyte.gz': 5148
    }
    
    os.makedirs(data_dir, exist_ok=True)
    
    for file, expected_size in files.items():
        file_path = os.path.join(data_dir, file)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)

        if not os.path.exists(file_path):
            print(f"正在尝试下载 {file}...")
            try:
                req = urllib.request.Request(base_url + file, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"{file} 下载成功!")
            except Exception as e:
                print(f"{file} 下载失败: {e}")

def load_fashion_mnist(image_path, label_path):
    target_dir = os.path.dirname(image_path) or '.'
    download_fashion_mnist(target_dir)
    
    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        raise FileNotFoundError(f"无法找到完整的数据文件 {image_path}，请按照建议手动下载。")

    try:
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        
        with gzip.open(image_path, 'rb') as imgpath:
            #直接图像将其拉伸为 784 维向量以适配 MLP 输入
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            
        images = images.astype(np.float32) / 255.0  
        return images, labels
    except Exception as e:
        print(f"读取压缩包时出错: {e}")
        return None, None

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]

class Layer:
    def __init__(self):
        self.cache = None
        self.params = {}
        self.grads = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        #权重矩阵 W 使用了 He 初始化 (np.sqrt(2.0 / in_features)) 加速 ReLU 收敛
        self.params['W'] = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.params['b'] = np.zeros(out_features)

    def forward(self, x):
        self.cache = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        x = self.cache
        self.grads['W'] = np.dot(x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0)
        dx = np.dot(dout, self.params['W'].T)
        return dx

class ReLU(Layer):
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout):
        x = self.cache
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx

class Sigmoid(Layer):
    def forward(self, x):
        #为防止 np.exp 产生溢出(OverflowError)，此处加入了 np.clip 截断
        x_safe = np.clip(x, -500, 500)
        out = 1.0 / (1.0 + np.exp(-x_safe))
        self.cache = out
        return out

    def backward(self, dout):
        out = self.cache
        return dout * out * (1.0 - out)

class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def forward(self, logits, y_true):
        #稳定的 Softmax 计算：减去行最大值防止指数爆炸
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted_logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.cache = (probs, y_true)
        
        batch_size = logits.shape[0]
        #加上 1e-9 防止 np.log(0) 导致 loss 出现 NaN
        correct_logprobs = -np.log(probs[np.arange(batch_size), y_true] + 1e-9)
        return np.sum(correct_logprobs) / batch_size

    def backward(self):
        probs, y_true = self.cache
        batch_size = probs.shape[0]
        dx = probs.copy()
        dx[np.arange(batch_size), y_true] -= 1.0
        return dx / batch_size

class MLP:
    def __init__(self, input_dim=784, hidden_dim1=128,hidden_dim2=128, num_classes=10, activation='relu'):#这里可以手动修改隐藏层参数hidden_dim1与hidden_dim2
        self.layers = []
        
        self.layers.append(Linear(input_dim, hidden_dim1))
        self.layers.append(ReLU() if activation.lower() == 'relu' else Sigmoid())
        
        self.layers.append(Linear(hidden_dim1, hidden_dim2))
        self.layers.append(ReLU() if activation.lower() == 'relu' else Sigmoid())
        
        self.layers.append(Linear(hidden_dim2, num_classes))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_params_and_grads(self):
        return [(l.params, l.grads) for l in self.layers if l.params]

class SGD:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        params_grads = self.model.get_params_and_grads()
        for params, grads in params_grads:
            for key in params.keys():
                #此处整合了 L2 正则化 (Weight Decay)
                grad_with_l2 = grads[key] + self.weight_decay * params[key]
                params[key] -= self.lr * grad_with_l2
                
    def update_lr(self, decay_rate):
        self.lr *= decay_rate
        
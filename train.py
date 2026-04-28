import os
import time
import pickle
import numpy as np
from core import load_fashion_mnist, DataLoader, MLP, CrossEntropyLoss, SGD
from utils import calculate_accuracy, plot_training_curves

def train_model(X_train, y_train, X_val, y_val, config):
    print(f"\n[{'='*40}]")
    print(f"开始训练模型 | 参数配置: {config}")
    
    model = MLP(
        input_dim=784, 
        hidden_dim1=config['hidden_dim1'], 
        hidden_dim2=config['hidden_dim2'],
        num_classes=10, 
        activation=config.get('activation', 'relu')
    )
    criterion = CrossEntropyLoss()
    optimizer = SGD(model, lr=config['lr'], weight_decay=config['weight_decay'])
    
    train_loader = DataLoader(X_train, y_train, batch_size=config['batch_size'], shuffle=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            logits = model.forward(batch_X)
            loss = criterion.forward(logits, batch_y)
            train_losses.append(loss)
            
            dout = criterion.backward()
            model.backward(dout)
            optimizer.step()
            
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        optimizer.update_lr(config.get('lr_decay', 0.95))
        
        val_logits = model.forward(X_val)
        val_loss = criterion.forward(val_logits, y_val)
        history['val_loss'].append(val_loss)
        
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = calculate_accuracy(val_preds, y_val)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{config['epochs']}] | Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = config.get('save_path', 'best_model.pkl')
            
            #此处将超参数(config)与模型权重一并保存，方便 eval 时自动读取结构
            checkpoint = {
                'config': config,
                'params': [layer.params.copy() for layer in model.layers if layer.params]
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"发现更优模型 (Val Acc: {val_acc*100:.2f}%)，已保存至 {save_path}!")

    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.2f}s | 最佳验证集准确率: {best_val_acc*100:.2f}%")
    return history, best_val_acc


def grid_search(X_train, y_train, X_val, y_val):
    print("\n" + "="*50)
    print("启动超参数网格搜索 (Grid Search)")
    print("="*50)
    
    #修改此处字典可自定义网格搜索的候选参数范围
    search_space = {
        'lr': [0.01, 0.05],
        'hidden_dim1': [256, 512],
        'hidden_dim2': [128, 256],
        'weight_decay': [0.0, 1e-4]
    }
    
    best_overall_acc = 0.0
    best_config = None
    results_log = []
    
    base_config = {
        'epochs': 50,
        'batch_size': 128,
        'lr_decay': 0.98,
        'activation': 'relu'
    }
    
    search_idx = 1
    total_searches = (len(search_space['lr']) * len(search_space['hidden_dim1']) * len(search_space['hidden_dim2']) * len(search_space['weight_decay']))
    
    weights_dir = 'outputs/weights'
    os.makedirs(weights_dir, exist_ok=True)

    for lr in search_space['lr']:
        for hd1 in search_space['hidden_dim1']:
            for hd2 in search_space['hidden_dim2']:
                for wd in search_space['weight_decay']:
                    
                    print(f"\n>>> 正在测试第 {search_idx}/{total_searches} 组配置...")
                    current_config = base_config.copy()
                    temp_save_name = os.path.join(weights_dir, f'model_search_{search_idx}.pkl')
                    current_config.update({
                        'lr': lr, 
                        'hidden_dim1': hd1, 
                        'hidden_dim2': hd2, 
                        'weight_decay': wd, 
                        'save_path': temp_save_name
                    })
                    
                    _, best_acc = train_model(X_train, y_train, X_val, y_val, current_config)
                    results_log.append((current_config, best_acc))
                    
                    if best_acc > best_overall_acc:
                        best_overall_acc = best_acc
                        best_config = current_config
                        best_model_path = os.path.join(weights_dir, 'grid_search_best_model.pkl')
                        if os.path.exists(best_model_path):
                            os.remove(best_model_path)
                        os.rename(current_config['save_path'], best_model_path)
                    else:
                        if os.path.exists(current_config['save_path']):
                            os.remove(current_config['save_path'])
                            
                    search_idx += 1
                
    print("\n" + "="*50)
    print(f"网格搜索结束! 最优验证集准确率: {best_overall_acc*100:.2f}%")
    print(f"最优配置: {best_config}")
    print("="*50)
    
    return best_config

def random_search(X_train, y_train, X_val, y_val, num_trials=20):
    """
    随机搜索 (Random Search)：在连续和离散空间中随机采样超参数
    num_trials: 探索的总次数
    """
    print("\n" + "="*50)
    print(f"启动超参数随机搜索 (Random Search) - 共设定 {num_trials} 次试验")
    print("="*50)
    
    best_overall_acc = 0.0
    best_config = None
    results_log = []
    
    #随机搜索的基础配置
    base_config = {
        'epochs': 50,
        'batch_size': 128,
        'lr_decay': 0.98,
        'activation': 'relu'
    }
    
    weights_dir = 'outputs/weights'
    os.makedirs(weights_dir, exist_ok=True)

    for trial in range(1, num_trials + 1):
        #这里用 float() 和 int() 包装了一下，防止 np.int64 导致后续保存字典时报错
        lr = float(np.random.uniform(0.04, 0.08))       
        wd = float(np.random.uniform(0, 3e-4))       
        hd1 = int(np.random.choice([256, 384, 512]))
        hd2 = int(np.random.choice([128, 256])) 
        
        current_config = base_config.copy()
        temp_save_name = os.path.join(weights_dir, f'model_random_{trial}.pkl')
        current_config.update({
            'lr': lr, 
            'hidden_dim1': hd1, 
            'hidden_dim2': hd2, 
            'weight_decay': wd, 
            'save_path': temp_save_name
        })
        
        print(f"\n>>> 正在测试第 {trial}/{num_trials} 组随机配置...")
        print(f"采样参数: lr={lr:.4f}, hd1={hd1}, hd2={hd2}, weight_decay={wd:.6f}")
        
        _, best_acc = train_model(X_train, y_train, X_val, y_val, current_config)
        results_log.append((current_config, best_acc))
        
        if best_acc > best_overall_acc:
            best_overall_acc = best_acc
            best_config = current_config
            best_model_path = os.path.join(weights_dir, 'random_search_best_model.pkl')
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            os.rename(current_config['save_path'], best_model_path)
        else:
            if os.path.exists(current_config['save_path']):
                os.remove(current_config['save_path'])
                
    print("\n" + "="*50)
    print(f"随机搜索结束! 最优验证集准确率: {best_overall_acc*100:.2f}%")
    print(f"最优配置: {best_config}")
    print("="*50)
    
    return best_config

if __name__ == '__main__':
    #此 block 仅用于文件内单独调试，项目正式运行请用main.py
    print("正在加载 Fashion-MNIST 数据集...")
    X, y = load_fashion_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    
    num_val = 10000
    num_trials=20
    indices = np.random.permutation(X.shape[0])
    X, y = X[indices], y[indices]
    
    X_val, y_val = X[:num_val], y[:num_val]
    X_train, y_train = X[num_val:], y[num_val:]
    print(f"数据划分完毕! 训练集: {X_train.shape[0]} 张, 验证集: {X_val.shape[0]} 张")
    
    MODE = 'train'
    
    if MODE == 'train':
        default_config = {
            'epochs': 20,
            'batch_size': 128,
            'lr': 0.05,
            'hidden_dim': 256,
            'weight_decay': 1e-4,
            'lr_decay': 0.95,
            'activation': 'relu',
            'save_path': 'best_model.pkl'
        }
        history, _ = train_model(X_train, y_train, X_val, y_val, default_config)
        plot_training_curves(history['train_loss'], history['val_loss'], history['val_acc'], save_path="training_curves.png")
        
    elif MODE == 'search':
        best_cfg = grid_search(X_train, y_train, X_val, y_val)

    elif MODE == 'random':
        best_cfg = random_search(X_train, y_train, X_val, y_val, num_trials)    

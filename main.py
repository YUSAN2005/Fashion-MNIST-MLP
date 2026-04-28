import argparse
import numpy as np
import os

from core import load_fashion_mnist
from train import train_model, grid_search,random_search
from eval import evaluate_pipeline
from utils import plot_training_curves

def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST MLP Classifier - 从零手搓神经网络")
    
    #这是整个项目的核心入口。常规参数可通过命令行/CLI直接指定，无需直接修改代码。
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'search', 'eval','random'],
                        help="运行模式：'train'(单次训练), 'search'(网格搜索最优超参数), 'random'(随机搜索), 'eval'(加载模型并测试评估)")
    
    parser.add_argument('--epochs', type=int, default=20, help="训练的总轮数 (Epochs)")
    parser.add_argument('--batch_size', type=int, default=128, help="批次大小 (Batch Size)")
    parser.add_argument('--lr', type=float, default=0.05, help="初始学习率")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="学习率衰减率")
    parser.add_argument('--hidden_dim1', type=int, default=256, help="第一层隐藏层神经元数量")
    parser.add_argument('--hidden_dim2', type=int, default=128, help="第二层隐藏层神经元数量")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 正则化强度 (Weight Decay)")
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'], help="激活函数类型")
    parser.add_argument('--num_trials', type=int, default=20, help="随机搜索的试验次数 (仅在 mode='random' 时生效)")
    
    #默认的输出路径。权重存放在 outputs/weights 目录下。
    parser.add_argument('--model_path', type=str, default='outputs/weights/best_model.pkl', help="模型权重保存或读取的路径")
    
    args = parser.parse_args()

    print("\n" + "="*50)
    print(f"欢迎使用 Fashion-MNIST MLP 训练框架")
    print(f"当前运行模式: [{args.mode.upper()}]")
    print("="*50)

    if args.mode == 'eval':
        evaluate_pipeline(model_path=args.model_path)
        
    else:
        print("正在加载并划分数据集...")
        try:
            #数据集应存放于项目根目录的 'data/' 文件夹中
            X, y = load_fashion_mnist('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
        except FileNotFoundError:
            print("错误：找不到训练集文件，请确认 data/ 文件夹下有对应文件！")
            return
            
        #使用固定种子 123，确保每次批改时训练集/验证集的切分完全一致
        np.random.seed(123) 
        indices = np.random.permutation(X.shape[0])
        X, y = X[indices], y[indices]
        
        #验证集大小被硬编码为 10000。如需调整划分比例(如 8:2)，请修改此处变量
        num_val = 10000
        X_val, y_val = X[:num_val], y[:num_val]
        X_train, y_train = X[num_val:], y[num_val:]
        
        print(f"数据加载成功! 训练集: {X_train.shape[0]} 样本 | 验证集: {X_val.shape[0]} 样本")
        
        if args.mode == 'train':
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            config = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'hidden_dim1': args.hidden_dim1,
                'hidden_dim2': args.hidden_dim2,
                'weight_decay': args.weight_decay,
                'activation': args.activation,
                'lr_decay': args.lr_decay,
                'save_path': args.model_path
            }
            history, best_acc = train_model(X_train, y_train, X_val, y_val, config)
            
            print("\n正在生成训练曲线 (training_curves.png)...")
            plot_training_curves(history['train_loss'], history['val_loss'], history['val_acc'], save_path="outputs/plots/training_curves.png")
            
        elif args.mode == 'search':
            best_config = grid_search(X_train, y_train, X_val, y_val)
            print(f"\n提示：可以使用以下命令基于网格搜索到的最优配置重新训练：")
            print(f"python main.py --mode train --lr {best_config['lr']} --hidden_dim1 {best_config['hidden_dim1']} --hidden_dim2 {best_config['hidden_dim2']} --weight_decay {best_config['weight_decay']}")
        
        elif args.mode == 'random':
            best_config = random_search(X_train, y_train, X_val, y_val, num_trials=args.num_trials)
            print(f"\n提示：可以使用以下命令基于随机搜索的最优配置重新训练：")
            print(f"python main.py --mode train --lr {best_config['lr']} --hidden_dim1 {best_config['hidden_dim1']} --hidden_dim2 {best_config['hidden_dim2']} --weight_decay {best_config['weight_decay']}")

if __name__ == '__main__':
    main()
    
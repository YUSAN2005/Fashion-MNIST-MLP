import pickle
import numpy as np

from core import load_fashion_mnist, MLP
from utils import calculate_accuracy, plot_confusion_matrix, visualize_weights, visualize_error_cases

#这里的 model_path 默认对接 main.py 中设定好的 outputs/weights/best_model.pkl
def evaluate_pipeline(model_path='outputs/weights/best_model.pkl'):
    print("="*50)
    print("启动模型测试与分析流程")
    print("="*50)

    print("正在加载测试集数据...")
    try:
        X_test, y_test = load_fashion_mnist('data/t10k-images-idx3-ubyte.gz', 'data/t10k-labels-idx1-ubyte.gz')
        print(f"测试集加载完毕! 包含 {X_test.shape[0]} 张图像。")
    except FileNotFoundError:
        print("错误：找不到测试集文件，请确认 t10k-images 和 t10k-labels 文件在当前目录下！")
        return

    print(f"\n正在初始化模型并加载权重 ({model_path})...")
    print(f"\n正在解析模型检查点并加载权重 ({model_path})...")
    
    try:
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        #代码会自动从 pkl 的 config 字典中提取训练时的结构，避免手动输入维度引发不匹配报错
        train_config = checkpoint.get('config', {})
        h_dim1 = train_config.get('hidden_dim1', 256)
        h_dim2 = train_config.get('hidden_dim2', 128)
        act = train_config.get('activation', 'relu') 
        print(f"检测到训练参数：hidden_dim1={h_dim1}, hidden_dim2={h_dim2}, activation={act}")
        
        model = MLP(input_dim=784, hidden_dim1=h_dim1, hidden_dim2=h_dim2, num_classes=10, activation=act)
        saved_weights = checkpoint['params']    
        
        weight_idx = 0
        for layer in model.layers:
            if layer.params:
                layer.params = saved_weights[weight_idx]
                weight_idx += 1
        print("模型权重加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到权重文件 {model_path}，请先运行 train.py 训练模型！")
        return

    print("\n正在测试集上进行推理计算...")
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)

    test_acc = calculate_accuracy(y_pred, y_test)
    print(f"独立测试集最终准确率 (Test Accuracy): {test_acc * 100:.2f}%")
    
    print("\n正在生成实验报告所需的图表...")

    print(" -> 绘制并保存混淆矩阵 (confusion_matrix.png)...")
    plot_confusion_matrix(y_test, y_pred, save_path='outputs/plots/confusion_matrix.png')

    first_layer_weights = model.layers[0].params['W']
    total_neurons = first_layer_weights.shape[1]
    #此处默认提取第一层的前 16 个神经元进行可视化。如需生成更多，修改 num_neurons 即可
    print(" -> 绘制并保存权重热力图 (weights_heatmap.png)...")
    visualize_weights(first_layer_weights, num_neurons=16, 
                      save_path='outputs/plots/weights_heatmap.png', cmap='RdBu')

    print(" -> 绘制并保存权重灰度图 (weights_gray.png)...")
    visualize_weights(first_layer_weights, num_neurons=16, 
                      save_path='outputs/plots/weights_gray.png', cmap='gray')

    #错例分析默认展示 18 张图(3x6)。可通过调整 num_samples 增减展示数量
    print(" -> 挑选错例并保存分析图 (error_analysis.png)...")
    visualize_error_cases(X_test, y_test, y_pred, num_samples=18, save_path='outputs/plots/error_analysis.png')

    print("\n所有测试与可视化任务已完成！")

if __name__ == '__main__':
    evaluate_pipeline('best_model.pkl')
    
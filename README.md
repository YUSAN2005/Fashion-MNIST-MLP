# Fashion-MNIST MLP Classification (从零开始构建三层神经网络分类器,实现Fashion-MNIST 图像分类）

本项目为复旦大学数据科学与大数据技术专业计算机视觉课程的实验代码。
本项目基于纯 NumPy 从零构建了包含两层隐藏层的MLP模型，并在 Fashion-MNIST 数据集上实现了完整的端到端训练、超参数搜索与空间特征可视化。

---

## 项目结构

```text
Fashion-MNIST/
├── data/                   # 数据集目录 (存放 Fashion-MNIST 原始 gz 文件)
├── outputs/
│   ├── weights/            # 训练好的模型权重保存路径 (.pkl)
│   └── plots/              # 训练曲线、特征可视化与错例分析图表输出路径
├── core.py                 # MLP 网络结构、前向传播与反向传播的纯数学实现
├── utils.py                # 图像可视化、特征图绘制、混淆矩阵绘制等工具
├── train.py                # 训练循环、Grid Search 与 Random Search 调度逻辑
├── eval.py                 # 模型评估模块 (包含加载权重、测试集推理与生成分析图表)
├── main.py                 # 项目主入口命令行工具
└── README.md               # 项目说明文档
```

---

## 环境依赖 (Dependencies)

本项目秉持轻量化与底层推导原则，**不依赖**任何如 PyTorch、TensorFlow 等重型深度学习框架。

* **Python:** 3.8+
* **NumPy:** 核心矩阵运算与反向传播计算
* **Matplotlib:** 训练曲线与特征可视化渲染
* **Pickle:** (Python 内置) 用于模型权重的序列化保存与加载

**快速安装依赖：**
```bash
pip install numpy matplotlib
```
*(注：代码中包含了自动下载逻辑。如果由于网络原因下载失败，请确保手动将 Fashion-MNIST 的 4 个 `.gz` 数据集文件放置于 `data/` 目录下。)*

---

## 运行指南

本项目所有的功能均通过 `main.py` 的不同 `--mode` 参数进行统一切换。以下是常用的运行指令：

### 1. 模型评估与可视化 (Evaluation & Visualization)

本项目在测试集上取得最高准确率的模型权重文件 `best_model.pkl` 已上传至 Google Drive。若要复现报告中的评估结果，请遵循以下步骤：

1.  **下载权重文件**：请访问 [Google Drive 分享链接](这里替换为你的真实分享链接)，下载 `best_model.pkl`。
2.  **放置文件**：在项目根目录下手动创建 `outputs/weights/` 文件夹（如果不存在），并将下载好的文件放入其中。请确保路径结构如下：
    ```text
    Fashion-MNIST/
    └── outputs/
        └── weights/
            └── best_model.pkl  <-- 确保文件名和路径准确无误
    ```
3.  **运行评估脚本**：
    在终端执行以下命令。程序将自动加载权重并在 10,000 张测试集上进行推理，随后生成混淆矩阵、512维特征热力图及错例分析图：
    ```bash
    python main.py --mode eval --model_path outputs/weights/best_model.pkl
    ```
    *注：运行结束后，生成的分析图表将保存在 `outputs/plots/` 目录下。*

---

### 2. 控制台参数说明 (CLI Arguments)

本项目的主入口 `main.py` 支持丰富的命令行参数。通过控制台参数，您可以灵活地调整模型结构、训练超参数或切换运行模式，无需修改源代码。

| 参数名称 | 类型 | 默认值 | 可选范围 / 功能说明 |
| :--- | :---: | :---: | :--- |
| `--mode` | `str` | `train` | 运行模式：`train`(单次训练), `search`(网格搜索), `random`(随机搜索), `eval`(测试评估) |
| `--epochs` | `int` | `20` | 训练的总轮数 (Epochs) |
| `--batch_size` | `int` | `128` | 批次大小 (Batch Size) |
| `--lr` | `float` | `0.05` | 初始学习率 |
| `--lr_decay` | `float` | `0.95` | 学习率每轮的指数衰减率 |
| `--hidden_dim1` | `int` | `256` | 第一层隐藏层神经元数量 |
| `--hidden_dim2` | `int` | `128` | 第二层隐藏层神经元数量 |
| `--weight_decay` | `float` | `1e-4` | L2 正则化强度 (Weight Decay) |
| `--activation` | `str` | `relu` | 隐藏层激活函数。可选: `relu`, `sigmoid` |
| `--num_trials` | `int` | `20` | 随机搜索的试验次数 (仅在 `mode='random'` 时生效) |
| `--model_path` | `str` | `outputs/weights/best_model.pkl` | 模型权重保存或读取的路径 |

**快速复现最优训练配置示例：**
```bash
python main.py --mode train --lr 0.0768 --lr_decay 0.98 --hidden_dim1 512 --hidden_dim2 256 --weight_decay 0.000228 --epochs 60
```

### 3. 超参数自动化搜索 (Hyperparameter Search)
本项目实现了自动化的参数搜索流水线，运行后会自动记录最优参数并保存权重：

* **运行网格搜索 (Grid Search)** —— ：
    ```bash
    python main.py --mode search
    ```
* **运行随机搜索 (Random Search)** —— ：
    ```bash
    python main.py --mode random --num_trials 20
    ```

---

**学术诚信声明 (Academic Integrity)**
本项目为课程实验的开源存档。为维护复旦大学学术诚信规范，谢绝同校选修本课程的同学直接克隆、复制本仓库代码用于作业提交。本代码仅供学习思路参考与技术交流。
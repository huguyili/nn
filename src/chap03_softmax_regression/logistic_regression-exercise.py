#!/usr/bin/env python
# coding: utf-8
# # 优化版逻辑回归示例
# ### 生成数据集并实现逻辑回归模型

# 导入必要的库
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import matplotlib.cm as cm
from IPython.display import HTML

# 设置随机种子确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 确保在Jupyter Notebook中内联显示图形
get_ipython().run_line_magic('matplotlib', 'inline')

class LogisticRegression:
    """逻辑回归模型实现，包含前向传播、损失计算和训练功能"""
    
    def __init__(self, l2_reg=0.01):
        """
        初始化逻辑回归模型参数
        
        参数:
            l2_reg: L2正则化系数，用于防止过拟合
        """
        # 初始化权重和偏置，添加L2正则化
        self.W = tf.Variable(
            initial_value=tf.random.uniform(shape=[2, 1], minval=-0.1, maxval=0.1),
            name="weights",
            regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(shape=[1]),
            name="bias",
            trainable=True
        )
        # 定义可训练变量
        self.trainable_variables = [self.W, self.b]
    
    @tf.function
    def __call__(self, inputs):
        """
        前向传播计算预测概率
        
        参数:
            inputs: 输入特征，形状为[batch_size, 2]
            
        返回:
            预测概率，形状为[batch_size, 1]
        """
        logits = tf.matmul(inputs, self.W) + self.b
        return tf.nn.sigmoid(logits)
    
    @tf.function
    def compute_loss(self, predictions, labels):
        """
        计算二分类交叉熵损失和准确率（手动实现）
        
        参数:
            predictions: 模型预测概率，形状为[batch_size, 1]
            labels: 真实标签，形状为[batch_size, 1]
            
        返回:
            loss: 平均损失值
            accuracy: 准确率
        """
        # 确保标签为Tensor类型
        if not isinstance(labels, tf.Tensor):
            labels = tf.constant(labels, dtype=tf.float32)
            
        # 压缩维度便于计算
        pred = tf.squeeze(predictions, axis=1)
        label = tf.squeeze(labels, axis=1)
        
        # 数值稳定性处理
        epsilon = 1e-12
        pred = tf.clip_by_value(pred, epsilon, 1.0 - epsilon)
        
        # 计算交叉熵损失
        cross_entropy = -label * tf.math.log(pred) - (1 - label) * tf.math.log(1 - pred)
        loss = tf.reduce_mean(cross_entropy)
        
        # 计算L2正则化损失
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables if v is not self.b]) * 0.01
        total_loss = loss + l2_loss
        
        # 计算准确率
        predicted_labels = tf.cast(pred > 0.5, tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, label), tf.float32))
        
        return total_loss, accuracy

def generate_dataset(dot_num=100):
    """
    生成两类服从高斯分布的样本数据
    
    参数:
        dot_num: 每类样本数量
        
    返回:
        data_set: 合并后的数据集，形状为[2*dot_num, 3]
    """
    # 生成正样本 (蓝色加号)
    x_p = np.random.normal(3.0, 1.0, dot_num)
    y_p = np.random.normal(6.0, 1.0, dot_num)
    C1 = np.array([x_p, y_p, np.ones(dot_num)]).T
    
    # 生成负样本 (绿色圆圈)
    x_n = np.random.normal(6.0, 1.0, dot_num)
    y_n = np.random.normal(3.0, 1.0, dot_num)
    C2 = np.array([x_n, y_n, np.zeros(dot_num)]).T
    
    # 合并并打乱数据集
    data_set = np.concatenate((C1, C2), axis=0)
    np.random.shuffle(data_set)
    return data_set, C1, C2

def train_model(model, optimizer, x, y, epochs=200, log_interval=20):
    """
    训练逻辑回归模型
    
    参数:
        model: 逻辑回归模型实例
        optimizer: 优化器实例
        x: 输入特征
        y: 真实标签
        epochs: 训练轮数
        log_interval: 日志输出间隔
        
    返回:
        animation_frames: 训练过程中的模型参数和损失
    """
    animation_frames = []
    
    for i in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss, accuracy = model.compute_loss(predictions, y)
        
        # 计算并应用梯度
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # 记录训练过程用于可视化
        if i % 1 == 0:  # 每轮记录一次
            w1, w2 = model.W.numpy()[0, 0], model.W.numpy()[1, 0]
            b = model.b.numpy()[0]
            animation_frames.append((w1, w2, b, loss.numpy()))
        
        # 打印训练进度
        if i % log_interval == 0:
            print(f"Epoch {i}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}")
    
    return animation_frames

def visualize_training(animation_frames, C1, C2):
    """
    可视化训练过程和决策边界
    
    参数:
        animation_frames: 训练过程数据
        C1: 正样本数据
        C2: 负样本数据
        
    返回:
        anim: 动画对象
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('逻辑回归训练过程', fontsize=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # 绘制样本点
    positive_points, = ax.plot(C1[:, 0], C1[:, 1], '+', c='b', label='正样本')
    negative_points, = ax.plot(C2[:, 0], C2[:, 1], 'o', c='g', label='负样本')
    decision_boundary, = ax.plot([], [], 'r-', label='决策边界')
    legend = ax.legend(loc='upper right')
    frame_info = ax.text(
        0.02, 0.95, '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    def init():
        """初始化动画"""
        decision_boundary.set_data([], [])
        frame_info.set_text('')
        return (decision_boundary, positive_points, negative_points, frame_info)
    
    def animate(i):
        """更新每一帧"""
        w1, w2, b, loss = animation_frames[i]
        
        # 计算决策边界: w1*x + w2*y + b = 0 => y = (-w1/w2)*x - b/w2
        x_vals = np.linspace(0, 10, 100)
        if abs(w2) < 1e-6:  # 防止除零错误
            y_vals = np.ones_like(x_vals) * (-b/w1) if w1 != 0 else 5.0
        else:
            y_vals = (-w1/w2) * x_vals - b/w2
        
        decision_boundary.set_data(x_vals, y_vals)
        frame_info.set_text(
            f'迭代: {i+1}/{len(animation_frames)}\n'
            f'损失: {loss:.4f}\n'
            f'参数: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}'
        )
        return (decision_boundary, positive_points, negative_points, frame_info)
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(animation_frames), interval=50, blit=True, repeat=False
    )
    
    return anim

if __name__ == "__main__":
    # 生成数据集
    data_set, C1, C2 = generate_dataset(dot_num=100)
    
    # 准备训练数据
    x1, x2, y = list(zip(*data_set))
    x = tf.constant(list(zip(x1, x2)), dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)[:, tf.newaxis]
    
    # 初始化模型和优化器
    model = LogisticRegression(l2_reg=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # 训练模型
    print("开始训练...")
    animation_frames = train_model(model, optimizer, x, y, epochs=200, log_interval=20)
    
    # 可视化训练过程
    print("生成训练动画...")
    anim = visualize_training(animation_frames, C1, C2)
    
    # 显示动画
    HTML(anim.to_html5_video())
    
    # 绘制最终决策边界
    plt.figure(figsize=(8, 6))
    plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+', label='正样本')
    plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o', label='负样本')
    
    w1, w2, b, _ = animation_frames[-1]
    x_vals = np.linspace(0, 10, 100)
    if abs(w2) < 1e-6:
        y_vals = np.ones_like(x_vals) * (-b/w1) if w1 != 0 else 5.0
    else:
        y_vals = (-w1/w2) * x_vals - b/w2
    
    plt.plot(x_vals, y_vals, 'r-', label='最终决策边界')
    plt.title('逻辑回归最终决策边界')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

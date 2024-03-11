import numpy as np
import matplotlib.pyplot as plt

# 定义简单的神经网络类
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化神经网络模型参数。
        
        :param input_size: 输入层大小
        :param hidden_size: 隐藏层大小
        :param output_size: 输出层大小
        """
        np.random.seed(2151300)  # 确保随机数的可复现性
        self.W1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重
        self.b1 = np.zeros(hidden_size)  # 隐藏层的偏置项
        self.W2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重
        self.b2 = np.zeros(output_size)  # 输出层的偏置项
    
    def relu(self, Z):
        """
        ReLU激活函数。
        
        :param Z: 线性输出值
        :return: ReLU激活后的值
        """
        return np.maximum(0, Z)

    def d_relu(self, Z):
        """
        ReLU函数的导数。
        
        :param Z: 线性输出值
        :return: 导数值
        """
        return Z > 0
    
    def forward(self, x):
        """
        前向传播过程。
        
        :param x: 输入数据
        :return: 网络预测值
        """
        self.Z1 = x.dot(self.W1) + self.b1  # 第一层的线性部分
        self.A1 = self.relu(self.Z1)  # 应用ReLU激活函数
        self.Z2 = self.A1.dot(self.W2) + self.b2  # 第二层的线性部分
        return self.Z2
    
    def compute_loss(self, y_true, y_pred):
        """
        计算损失函数（均方误差）。
        
        :param y_true: 真实值
        :param y_pred: 预测值
        :return: 损失值
        """
        return ((y_true - y_pred) ** 2).mean()

    def backward(self, x, y, learning_rate=0.01):
        """
        反向传播过程，根据损失函数梯度更新参数。
        
        :param x: 输入数据
        :param y: 真实值
        :param learning_rate: 学习率
        """
        m = y.shape[0]  # 数据点数量
        dZ2 = self.Z2 - y  # 输出层梯度
        dW2 = self.A1.T.dot(dZ2) / m  # 权重W2的梯度
        db2 = np.sum(dZ2, axis=0) / m  # 偏置b2的梯度
        dZ1 = dZ2.dot(self.W2.T) * self.d_relu(self.Z1)  # 隐藏层梯度
        dW1 = x.T.dot(dZ1) / m  # 权重W1的梯度
        db1 = np.sum(dZ1, axis=0) / m  # 偏置b1的梯度
        
        # 使用梯度下降法更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, x_train, y_train, epochs, learning_rate=0.01, decay_rate=0.0):
        """
        训练神经网络模型，加入学习率衰减策略。
        
        :param x_train: 训练数据
        :param y_train: 训练标签
        :param epochs: 训练轮数
        :param learning_rate: 初始学习率
        :param decay_rate: 学习率衰减率
        """
        for epoch in range(epochs):
            y_pred = self.forward(x_train)  # 前向传播
            loss = self.compute_loss(y_train, y_pred)  # 计算损失
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}, Learning Rate: {learning_rate}")  # 输出当前轮次的损失值和学习率
            self.backward(x_train, y_train, learning_rate)  # 反向传播和参数更新
            
            # 应用学习率衰减
            learning_rate *= (1. / (1. + decay_rate * epoch))

# 目标函数
def target_function(x):
    """
    定义目标函数。
    
    :param x: 输入值
    :return: f(x)
    """
    return x**3

# 生成数据集
x_train = np.linspace(-np.pi, np.pi, 700).reshape(-1, 1)  # 训练数据
y_train = target_function(x_train)  # 训练数据标签
x_test = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)  # 测试数据
y_test = target_function(x_test)  # 测试数据标签

# 初始化并训练模型
model = SimpleNeuralNetwork(input_size=1, hidden_size=30, output_size=1)
model.train(x_train, y_train, epochs=100000, learning_rate=0.001,decay_rate=1e-12)

# 使用模型进行预测
y_pred = model.forward(x_test)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label='True Function')  # 真实函数
plt.plot(x_test, y_pred, label='Model Prediction', linestyle='--')  # 模型预测
plt.legend()
plt.title("Neural Network Model vs True Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

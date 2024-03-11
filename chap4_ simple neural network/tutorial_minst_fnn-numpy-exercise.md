## 准备数据


```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)
```

## Demo numpy based auto differentiation


```python
import numpy as np

class Matmul:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x, W):
        h = np.matmul(x, W)
        self.mem={'x': x, 'W':W}
        return h
    
    def backward(self, grad_y):
        '''
        x: shape(N, d)
        w: shape(d, d')
        grad_y: shape(N, d')
        '''
        x = self.mem['x']
        W = self.mem['W']
        
        ####################
        '''计算矩阵乘法的对应的梯度'''
        ####################
        return grad_x, grad_W


class Relu:
    def __init__(self):
        self.mem = {}
        
    def forward(self, x):
        self.mem['x']=x
        return np.where(x > 0, x, np.zeros_like(x))
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        ####################
        '''计算relu 激活函数对应的梯度'''
        ####################
        return grad_x
    


class Softmax:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        x_exp = np.exp(x)
        partition = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp/(partition+self.epsilon)
        
        self.mem['out'] = out
        self.mem['x_exp'] = x_exp
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s,axis=2), np.expand_dims(s, axis=1)) # (N, c, c)
        g_y_exp = np.expand_dims(grad_y, axis=1)
        tmp = np.matmul(g_y_exp, sisj) #(N, 1, c)
        tmp = np.squeeze(tmp, axis=1)
        tmp = -tmp+grad_y*s 
        return tmp
    
class Log:
    '''
    softmax over last dimention
    '''
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
        
    def forward(self, x):
        '''
        x: shape(N, c)
        '''
        out = np.log(x+self.epsilon)
        
        self.mem['x'] = x
        return out
    
    def backward(self, grad_y):
        '''
        grad_y: same shape as x
        '''
        x = self.mem['x']
        
        return 1./(x+1e-12) * grad_y
    

```

## Gradient check


```python
# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# W = np.random.normal(size=[6, 4])
# aa = Matmul()
# out = aa.forward(x, W) # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print (grad)

# with tf.GradientTape() as tape:
#     x, W = tf.constant(x), tf.constant(W)
#     tape.watch(x)
#     y = tf.matmul(x, W)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Relu()
# out = aa.forward(x) # shape(5, 4)
# grad = aa.backward(np.ones_like(out))
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.relu(x)
#     loss = tf.reduce_sum(y)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf
# x = np.random.normal(size=[5, 6], scale=5.0, loc=1)
# label = np.zeros_like(x)
# label[0, 1]=1.
# label[1, 0]=1
# label[1, 1]=1
# label[2, 3]=1
# label[3, 5]=1
# label[4, 0]=1
# print(label)
# aa = Softmax()
# out = aa.forward(x) # shape(5, 6)
# grad = aa.backward(label)
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.nn.softmax(x)
#     loss = tf.reduce_sum(y*label)
#     grads = tape.gradient(loss, x)
#     print (grads)

# import tensorflow as tf

# x = np.random.normal(size=[5, 6])
# aa = Log()
# out = aa.forward(x) # shape(5, 4)
# grad = aa.backward(label)
# print (grad)

# with tf.GradientTape() as tape:
#     x= tf.constant(x)
#     tape.watch(x)
#     y = tf.math.log(x)
#     loss = tf.reduce_sum(y*label)
#     grads = tape.gradient(loss, x)
#     print (grads)
```

# Final Gradient Check


```python
import tensorflow as tf

label = np.zeros_like(x)
label[0, 1]=1.
label[1, 0]=1
label[2, 3]=1
label[3, 5]=1
label[4, 0]=1

x = np.random.normal(size=[5, 6])
W1 = np.random.normal(size=[6, 5])
W2 = np.random.normal(size=[5, 6])

mul_h1 = Matmul()
mul_h2 = Matmul()
relu = Relu()
softmax = Softmax()
log = Log()

h1 = mul_h1.forward(x, W1) # shape(5, 4)
h1_relu = relu.forward(h1)
h2 = mul_h2.forward(h1_relu, W2)
h2_soft = softmax.forward(h2)
h2_log = log.forward(h2_soft)


h2_log_grad = log.backward(label)
h2_soft_grad = softmax.backward(h2_log_grad)
h2_grad, W2_grad = mul_h2.backward(h2_soft_grad)
h1_relu_grad = relu.backward(h2_grad)
h1_grad, W1_grad = mul_h1.backward(h1_relu_grad)

print(h2_log_grad)
print('--'*20)
# print(W2_grad)

with tf.GradientTape() as tape:
    x, W1, W2, label = tf.constant(x), tf.constant(W1), tf.constant(W2), tf.constant(label)
    tape.watch(W1)
    tape.watch(W2)
    h1 = tf.matmul(x, W1)
    h1_relu = tf.nn.relu(h1)
    h2 = tf.matmul(h1_relu, W2)
    prob = tf.nn.softmax(h2)
    log_prob = tf.math.log(prob)
    loss = tf.reduce_sum(label * log_prob)
    grads = tape.gradient(loss, [prob])
    print (grads[0].numpy())
```

    [[  0.          29.88862875   0.           0.           0.
        0.        ]
     [363.02160168   0.           0.           0.           0.
        0.        ]
     [  0.           0.           0.          12.25561872   0.
        0.        ]
     [  0.           0.           0.           0.           0.
       53.54971193]
     [689.96870449   0.           0.           0.           0.
        0.        ]]
    ----------------------------------------
    [[  0.          29.88862875   0.           0.           0.
        0.        ]
     [363.02160181   0.           0.           0.           0.
        0.        ]
     [  0.           0.           0.          12.25561872   0.
        0.        ]
     [  0.           0.           0.           0.           0.
       53.54971193]
     [689.96870497   0.           0.           0.           0.
        0.        ]]
    

## 建立模型


```python
class myModel:
    def __init__(self):
        
        self.W1 = np.random.normal(size=[28*28+1, 100])
        self.W2 = np.random.normal(size=[100, 10])
        
        self.mul_h1 = Matmul()
        self.mul_h2 = Matmul()
        self.relu = Relu()
        self.softmax = Softmax()
        self.log = Log()
        
        
    def forward(self, x):
        x = x.reshape(-1, 28*28)
        bias = np.ones(shape=[x.shape[0], 1])
        x = np.concatenate([x, bias], axis=1)
        
        self.h1 = self.mul_h1.forward(x, self.W1) # shape(5, 4)
        self.h1_relu = self.relu.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.h2_log = self.log.forward(self.h2_soft)
            
    def backward(self, label):
        self.h2_log_grad = self.log.backward(-label)
        self.h2_soft_grad = self.softmax.backward(self.h2_log_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)
        
model = myModel()

```

## 计算 loss


```python
def compute_loss(log_prob, labels):
     return np.mean(np.sum(-log_prob*labels, axis=1))
    

def compute_accuracy(log_prob, labels):
    predictions = np.argmax(log_prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predictions==truth)

def train_one_step(model, x, y):
    model.forward(x)
    model.backward(y)
    model.W1 -= 1e-5* model.W1_grad
    model.W2 -= 1e-5* model.W2_grad
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy

def test(model, x, y):
    model.forward(x)
    loss = compute_loss(model.h2_log, y)
    accuracy = compute_accuracy(model.h2_log, y)
    return loss, accuracy
```

## 实际训练


```python
train_data, test_data = mnist_dataset()
train_label = np.zeros(shape=[train_data[0].shape[0], 10])
test_label = np.zeros(shape=[test_data[0].shape[0], 10])
train_label[np.arange(train_data[0].shape[0]), np.array(train_data[1])] = 1.
test_label[np.arange(test_data[0].shape[0]), np.array(test_data[1])] = 1.

for epoch in range(50):
    loss, accuracy = train_one_step(model, train_data[0], train_label)
    print('epoch', epoch, ': loss', loss, '; accuracy', accuracy)
loss, accuracy = test(model, test_data[0], test_label)

print('test loss', loss, '; accuracy', accuracy)
```

    epoch 0 : loss 24.72223286454846 ; accuracy 0.07413333333333333
    epoch 1 : loss 23.924144333646588 ; accuracy 0.1002
    epoch 2 : loss 23.024883096483002 ; accuracy 0.1252
    epoch 3 : loss 21.789798024024133 ; accuracy 0.16403333333333334
    epoch 4 : loss 20.651041530819974 ; accuracy 0.209
    epoch 5 : loss 19.77011018744175 ; accuracy 0.23966666666666667
    epoch 6 : loss 18.820253731189734 ; accuracy 0.26993333333333336
    epoch 7 : loss 17.89411614473916 ; accuracy 0.3019
    epoch 8 : loss 17.241523788018537 ; accuracy 0.32666666666666666
    epoch 9 : loss 16.750174736771154 ; accuracy 0.34513333333333335
    epoch 10 : loss 16.34424433437691 ; accuracy 0.36118333333333336
    epoch 11 : loss 15.995649918471548 ; accuracy 0.37455
    epoch 12 : loss 15.686255253464532 ; accuracy 0.38605
    epoch 13 : loss 15.398560292015457 ; accuracy 0.39716666666666667
    epoch 14 : loss 15.108667168656357 ; accuracy 0.40665
    epoch 15 : loss 14.768259637386473 ; accuracy 0.41781666666666667
    epoch 16 : loss 14.336060964222801 ; accuracy 0.42795
    epoch 17 : loss 13.800404674290487 ; accuracy 0.4438166666666667
    epoch 18 : loss 13.238676768175882 ; accuracy 0.45813333333333334
    epoch 19 : loss 12.275000859496435 ; accuracy 0.49001666666666666
    epoch 20 : loss 11.789863414935745 ; accuracy 0.5096833333333334
    epoch 21 : loss 11.099507177500909 ; accuracy 0.5359833333333334
    epoch 22 : loss 10.947845925466538 ; accuracy 0.5452
    epoch 23 : loss 10.37985058714904 ; accuracy 0.5686333333333333
    epoch 24 : loss 10.194234392408456 ; accuracy 0.5764666666666667
    epoch 25 : loss 9.9158980895203 ; accuracy 0.5883666666666667
    epoch 26 : loss 9.73326221836977 ; accuracy 0.59515
    epoch 27 : loss 9.547157941634678 ; accuracy 0.6050833333333333
    epoch 28 : loss 9.364149820809562 ; accuracy 0.61045
    epoch 29 : loss 9.241444692225024 ; accuracy 0.6171333333333333
    epoch 30 : loss 9.033834821879237 ; accuracy 0.6230166666666667
    epoch 31 : loss 8.909350423734871 ; accuracy 0.62905
    epoch 32 : loss 8.686975386903871 ; accuracy 0.6366
    epoch 33 : loss 8.52761694967565 ; accuracy 0.6425166666666666
    epoch 34 : loss 8.280276344997858 ; accuracy 0.6510666666666667
    epoch 35 : loss 8.017204559951352 ; accuracy 0.6591833333333333
    epoch 36 : loss 7.6927551641724925 ; accuracy 0.6705333333333333
    epoch 37 : loss 7.289147376861199 ; accuracy 0.6818166666666666
    epoch 38 : loss 6.890080011599251 ; accuracy 0.69715
    epoch 39 : loss 6.621282824275839 ; accuracy 0.70885
    epoch 40 : loss 6.432028811711102 ; accuracy 0.71685
    epoch 41 : loss 6.297568845936132 ; accuracy 0.7227666666666667
    epoch 42 : loss 6.199755928239367 ; accuracy 0.72755
    epoch 43 : loss 6.1290342585715685 ; accuracy 0.73055
    epoch 44 : loss 6.063146731968448 ; accuracy 0.7336166666666667
    epoch 45 : loss 6.032675366835999 ; accuracy 0.7359
    epoch 46 : loss 5.960594129095456 ; accuracy 0.73785
    epoch 47 : loss 5.943588768247914 ; accuracy 0.73995
    epoch 48 : loss 5.8297749382365 ; accuracy 0.7436666666666667
    epoch 49 : loss 5.800353627628091 ; accuracy 0.7465833333333334
    test loss 5.590955869790617 ; accuracy 0.7588
    

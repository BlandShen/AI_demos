#房价预测用的是线性回归模型
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Agg")

BUF_SIZE = 500
BATCH_SIZE = 20
iter = 0
iters = []
train_costs = []

def draw_train_process(iters,train_costs):
    title = "training cost"
    plt.title(title ,fontsize = 24)
    plt.xlabel("iter", fontsize = 14)
    plt.ylabel("cost", fontsize = 14)
    plt.plot(iters, train_costs, color = 'red', label = 'training cost')
    plt.grid()
    #plt.show()
    plt.savefig('/home/bland/note/demo/fangjiayuce/train_cost.png')

#训练集准备
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(),buf_size=BUF_SIZE),batch_size=BATCH_SIZE)
#测试集准备
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(),buf_size=BUF_SIZE),batch_size=BATCH_SIZE)

#这是个线性回归模型，所以不需要激活函数
#例子中模型只有全连接层
#模型建立
x = fluid.layers.data(name ='x', shape=[13], dtype='float32')
y = fluid.layers.data(name ='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1,act=None)

#那训练数据时的标准是什么呢？是损失函数。问题是回归问题，损失函数用均方误差MSE，优化函数用随机梯度下降算法SGD
#定义损失函数
cost = fluid.layers.square_error_cost(input = y_predict,label = y)
avg_cost = fluid.layers.mean(cost)

#定义优化函数
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

test_program = fluid.default_main_program().clone(for_test = True)

#训练需要什么？执行器Executer
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())#初始化
feeder = fluid.DataFeeder(place = place,feed_list = [x, y])

#开始进行训练
EPOCH_NUM = 100
model_save_dir = "/home/bland/note/demo/fangjiayuce/fit_a_line.inference.model"

for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id , data in enumerate(train_reader()):
        train_cost = exe.run(program = fluid.default_main_program(), feed = feeder.feed(data), fetch_list = [avg_cost])
    print("Train_Pass :{0:}, Cost :{1:.5f}".format(pass_id, train_cost[0][0]))

    test_cost = 0
    for batch_id ,data in enumerate(test_reader()):
        test_cost = exe.run(program = test_program, feed=feeder.feed(data), fetch_list=[avg_cost])
        if batch_id %20 ==0:
            print("Test :{0:}, Cost :{1:.5f}".format(pass_id, test_cost[0][0]))
        iter = iter +BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])

    #保存训练参数到指定路径中
    if not os.path.exists(model_save_dir):
        print("error : 保存路径不存在，即将创建路径")
        os.makedirs(model_save_dir)
        
    fluid.io.save_inference_model(model_save_dir, ['x'], [y_predict], exe)

    draw_train_process(iters, train_costs)
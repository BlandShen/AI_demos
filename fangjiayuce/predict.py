import paddle
import paddle.fluid as fluid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use = "Agg"

BUF_SIZE = 500
BATCH_SIZE = 20

#获取数据
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(),buf_size = BUF_SIZE),batch_size = BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(),buf_size = BUF_SIZE), batch_size = BATCH_SIZE)

#模型定义
x = fluid.layers.data(name = 'x', shape = [13],dtype = 'float32')
y = fluid.layers.data(name = 'y', shape = [1], dtype = 'floar32')
y_predict = fluid.layers.fc(input = x, size =1, act = None)

#均方误差
cost = fluid.layers.square_error_cost(input = y_predict,label = y )
avg_cost = fluid.layers.mean(cost)

#优化函数
optimizer = fluid.optimizer.SGDOptimizer(learning_rate = 0.001)
opts = optimizer.minimize(avg_cost)

test_program = fluid.default_main_program().clone(for_test = True)

#executer
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
feeder = fluid.DataFeeder(place = place ,feed_list = [x,y])

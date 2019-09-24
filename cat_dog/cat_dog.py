import paddle
import paddle.fluid as fluid
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#定义缓冲区和批次的大小
BUF_SIZE = 128*100
BATCH_SIZE = 128

#数据集的获取（reader 的定义）
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.train10(),buf_size = BUF_SIZE),batch_size =BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.cifar.test10(),buf_size = BUF_SIZE),batch_size =BATCH_SIZE) 

#数据的定义，default_start_program
images = fluid.layers.data(name = 'images', shape = [3,32,32], dtype = 'float32')
label = fluid.layers.data(name = 'label', shape = [1], dtype = 'int64')

#网络的定义，default_main_program
def cnn(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input = img,
        filter_size =5,
        num_filters =20,
        pool_size = 2,
        pool_stride = 2,
        act = 'relu'
    )
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input = conv_pool_1,
        filter_size = 5,
        num_filters = 50,
        pool_size = 2,
        pool_stride = 2,
        act = 'relu'
    )
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input = conv_pool_2,
        filter_size = 5,
        num_filters = 50,
        pool_size = 2,
        pool_stride = 2,
        act = 'relu'
    )
    prediction = fluid.layers.fc(input = conv_pool_3, size = 10,act = "softmax")
    return prediction

#注意分类问题，定义完网络后，定义分类器
predict = cnn(images)

#损失函数的定义，准确率的定义（分类问题，有监督学习，交叉熵损失函数）
cost = fluid.layers.cross_entropy(input = predict , label = label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input = predict, label = label)

#优化函数的定义，Adam
optimizer = fluid.optimizer.AdamOptimizer(learning_rate = 0.001)
opts = optimizer.minimize(avg_cost)

#executer的定义
use_gpu = False
if input("是否使用gpu? Y/N\n").lower() == "y":
    use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
test_program = fluid.default_main_program().clone(for_test=True)

#feeder定义
feeder = fluid.DataFeeder(feed_list = [images,label],place = place)

#循环训练模型并保存
save_model_path = '/home/userroot/database2/cat_dog/cat_dog.inference.model'
EPOCH_NUM = int(input("请输入训练轮次\n"))
train_iter = 0
train_iters = []
train_costs = []
train_accs = []
test_iters = []
test_costs = []
test_accs = []
for pass_id in range(EPOCH_NUM):
    for batch_id ,data in enumerate(train_reader()):
        train_cost ,train_acc = exe.run(
            program = fluid.default_main_program(),
            feed = feeder.feed(data),
            fetch_list = [avg_cost ,acc]
        )
        train_iter += batch_id
        train_iters.append(train_iter)
        train_costs.append(train_cost[0])
        train_accs.append(train_acc[0])
        if batch_id %100 == 0:
            print("训练轮次为{0:}，批次为{1:}时，损失为{2:.5}，准确率为{3:.5}".format(pass_id, batch_id, train_cost[0], train_acc[0]))
    for batch_id,data in enumerate(test_reader()):
        test_cost,test_acc = exe.run(
            program = fluid.default_main_program(),
            feed = feeder.feed(data),
            fetch_list = [avg_cost, acc]
        )
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])
avg_test_costs = sum(test_costs)/len(test_costs)
avg_test_accs = sum(test_accs)/len(test_accs)
print("辅助测试的平均损失为{0:.5f}，辅助测试的平均准确率为{1:.5f}".format(avg_test_costs,avg_test_accs))

judge  = input("是否为本地环境？Y/N\n").lower()
if judge =="y":
    save_model_path = '/home/bland/note/demo/cat_dog/cat_dog.inference.model'

if not os.path.exists(save_model_path):
    print("保存路径不存在，请重新确认路径")
try:
    fluid.io.save_inference_model(save_model_path,['images'],[predict],exe)
    print("模型保存成功！")
except:
    print("模型保存失败！")

def draw(title,iters,costs,accs,label_cost,label_acc):
    plt.title(title, fontsize =24)
    plt.xlabel("iter",fontsize =20)
    plt.ylabel("cost/acc",fontsize=20)
    plt.plot(iters, costs, color = "red",label = label_cost)
    plt.plot(iters, accs, color = "green",label = label_acc)
    plt.legend()
    plt.grid()
    plt.show()

draw("training", train_iters, train_costs, train_accs, "training cost", "training acc")
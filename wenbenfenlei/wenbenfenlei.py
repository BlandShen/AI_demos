import paddle
import paddle.fluid as fluid
import numpy as np
import os
#多线程处理，用于获得cpu核数
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
#获取字典长度
def get_dict_len(dict_path):
    with open(dict_path,"r",encoding="utf-8") as f:
        line = eval(f.readlines()[0])
    return len(line.keys())
#这个问题是分类问题，属于有监督学习

#数据集准备，reader的定义
def data_mapper(sample):#每个样本的标签和数据转化为int
    data, label = sample
    data = [int(data) for data in data.split(',')]
    return data,int(label)

def train_reader(train_list_path):
    def reader():
        with open(train_list_path,'r') as f1:
            lines = f1.readlines()
            np.random.shuffle(lines)
            for line in lines:
                data, label = line.split('\t')
                yield data ,label
    return paddle.reader.xmap_readers(data_mapper,reader,multiprocessing.cpu_count(),1024)

def test_reader(test_list_path):
    def reader():
        with open(test_list_path,'r') as f1:
            lines = f1.readlines()
            np.random.shuffle(lines)
            for line in lines:
                data, label = line.split('\t')
                yield data ,label
    return paddle.reader.xmap_readers(data_mapper,reader,multiprocessing.cpu_count(),1024)

train_reader = paddle.batch(reader = train_reader('/home/bland/note/demo/wenbenfenlei/train_list.txt'),batch_size  = 128)
test_reader = paddle.batch(reader = test_reader('/home/bland/note/demo/wenbenfenlei/test_list.txt'),batch_size  = 128)

#数据的定义（属于default_start_program模块）
words = fluid.layers.data(name='words',shape=[1],dtype='int64',lod_level = 1)#输入是个序列
label = fluid.layers.data(name='label',shape=[1],dtype='int64')
#网络的定义（属于default_main_program模块）
def cnn(data,dict_dim,class_dim=10,emb_dim=128,hid_dim=128,hid_dim2=98):
    emb = fluid.layers.embedding(input =data,size = [dict_dim,emb_dim])
    conv_3 =fluid.nets.sequence_conv_pool(
        input = emb,
        num_filters = hid_dim,
        filter_size = 3,
        act = "tanh",
        pool_type="sqrt"
    )
    conv_4 =fluid.nets.sequence_conv_pool(
        input = emb,
        num_filters=hid_dim2,
        filter_size = 4,
        act ="tanh",
        pool_type="sqrt"
    )
    output = fluid.layers.fc(
        input=[conv_3,conv_4],size = class_dim,act='softmax'
    )
    return output
#验证程序的定义
test_program = fluid.default_main_program().clone(for_test = True)
#获取字典长度
dict_dim = get_dict_len('/home/bland/note/demo/wenbenfenlei/dict_txt.txt')
#分类器的定义
model =cnn(words,dict_dim)
#损失函数的定义(交叉熵损失函数),准确率的定义
cost = fluid.layers.cross_entropy(input = model,label =label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input = model,label = label)
#优化函数的定义
optimizer = fluid.optimizer.AdagradOptimizer(learning_rate =0.002)
opt = optimizer.minimize(avg_cost)

#执行器的定义
place = fluid.CPUPlace()#这个程序用CPU多线程运行
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#数据映射器的定义
feeder = fluid.DataFeeder(place = place, feed_list = [words, label])

#循环数据，训练模型
EPOCH_NUM = 1
model_save_path = '/home/bland/note/demo/wenbenfenlei/'
iter = 0
iters = []
train_costs = []
train_accs = []
for pass_id in range(EPOCH_NUM):
    for batch_id,data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(
            program = fluid.default_main_program(),
            feed = feeder.feed(data),
            fetch_list = [avg_cost ,acc]
        )
        iter+=batch_id
        iters.append(iter)
        train_costs.append(train_cost)
        train_accs.append(train_acc)
        if batch_id%100 ==0:
            print("训练轮次为{0:}，批次为{1:}时，损失为{2:.5}，准确率为{3:.5}".format(pass_id, batch_id, train_cost[0], train_acc[0]))
    test_costs = []
    test_accs = []
    for batch_id,data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(
            program = fluid.default_main_program(),
            feed = feeder.feed(data),
            fetch_list = [avg_cost ,acc]
        )
        test_costs.append(test_cost)
        test_accs.append(test_acc)
    avg_test_cost = (sum(test_costs)/len(test_costs))
    avg_test_acc = (sum(test_accs)/len(test_accs))
    print("验证集平均损失{0:.5}，平均准确率为{1:.5}".format(avg_test_cost[0],avg_test_acc[0]))

if not os.path.exists(model_save_path):
    print("保存路径不存在!")
try:
    fluid.io.save_inference_model(
        model_save_path,
        feeded_var_names = [words.name],
        target_vars = [model],
        executor = exe
    )
    print("模型保存成功！")
except:
    print("模型保存失败！")

def draw(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title ,fontsize = 24)
    plt.xlabel("iter", fontsize = 20)
    plt.ylabel("cost/acc", fontsize = 20)
    plt.plot(iters, costs, color='red', label = label_cost)
    plt.plot(iters, accs, color = 'green', label = label_acc)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('/home/bland/note/demo/wenbenfenlei/train_cost_acc.png')

draw("training", iters, train_costs, train_accs,"training cost", "training acc")
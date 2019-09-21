import paddle.fluid as fluid
import numpy as np
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

save_path = "/home/bland/note/demo/wenbenfenlei/"

[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(
    dirname = save_path,
    executor = exe
)

def get_data(sentence):
    with open('/home/bland/note/demo/wenbenfenlei/dict_txt.txt','r',encoding="utf-8") as f_data:
        dict_txt = eval(f_data.readlines()[0])
        #f_data.readlines()[0]readlines读出的是个一维数组,eval将取出的字符串转换为字典
    #dict_txt =dict(dict_txt)
    data = []
    for s in sentence:
        if not s in dict_txt.keys():
            s = '<unk>'
        data.append(int(dict_txt[s]))
    return data

data = []
# 获取图片数据
data1 = get_data('在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说')
data2 = get_data('综合“今日美国”、《世界日报》等当地媒体报道，芝加哥河滨警察局表示，')
data.append(data1)
data.append(data2)

# 获取每句话的单词数量
base_shape = [[len(c) for c in data]]
print(base_shape)
# 生成预测数据
tensor_words = fluid.create_lod_tensor(data, base_shape, place)

# 执行预测
result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]: tensor_words},
                 fetch_list=target_var)

# 分类名称
names = [ '文化', '娱乐', '体育', '财经','房产', '汽车', '教育', '科技', '国际', '证券']

# 获取结果概率最大的label
for i in range(len(data)):
    lab = np.argsort(result)[0][i][-1]
    print('预测结果标签为：{0:}， 名称为：{1:}， 概率为：{2:}' .format(lab, names[lab], result[0][i][lab]))
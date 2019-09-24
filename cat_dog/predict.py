import paddle
import paddle.fluid as fluid
from PIL import Image
import numpy as np
save_model_path = '/home/bland/note/demo/cat_dog/cat_dog.inference.model'
#预测程序的定义
place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
#新建预测程序作用域
inference_scope = fluid.core.Scope()
#读取要预测的图片
def load_image(img):
    im = Image.open(img)
    im = im.resize((32,32),Image.ANTIALIAS)
    im = np.array(im).astype(np.float32)
    print(im.shape)
    im = im.transpose((2,0,1))#原shape为（32,32,3）(x,y,z)，要变为（3,32,32）（z,x,y）即transpose(2,0,1)
    im = im/255.0#归一化
    im = np.expand_dims(im ,axis = 0)#因为训练保存的时候是带batch的，加一个维度（1,3,32,32）
    print("im shape维度{0:}".format(im.shape))
    return im
#加载模型
if __name__ == '__main__':
    [inference_program,
    feed_targets_names,
    fetch_targets
    ] = fluid.io.load_inference_model(save_model_path, infer_exe)

    infer_path = 'dog.jpg'
    img = load_image(infer_path)
    print(feed_targets_names)
    results = infer_exe.run(inference_program, feed ={feed_targets_names[0]:img}, fetch_list = fetch_targets)
    print(results)
    label_list = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    print("该图片预测结果为：{:}".format(label_list[np.argmax(results[0])]))
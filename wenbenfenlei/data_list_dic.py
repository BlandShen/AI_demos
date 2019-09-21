import os

#创建数据序列
def creat_data_list(data_root_path):
    with open(os.path.join(data_root_path,"dict_txt.txt"),"r",encoding="utf-8") as f_data1:
        dict_txt = eval(f_data1.readlines()[0])
    with open(os.path.join(data_root_path,"news_classify_data.txt"),"r",encoding="utf-8") as f_data2:
        lines = f_data2.readlines()
    i = 0
    for line in lines:
        title = line.split("_!_")[-1].replace("\n","")
        l = line.split("_!_")[1]
        labs = ""
        if i%10==0:
            with open(os.path.join(data_root_path,"test_list.txt"),"a",encoding="utf-8") as f_test:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs+lab+','
                labs = labs[:-1]#不要最后一个逗号
                labs = labs +'\t'+l+"\n"
                f_test.write(labs)
        else:
            with open(os.path.join(data_root_path,"train_list.txt"),"a",encoding="utf-8") as f_train:
                for s in title:
                    lab = str(dict_txt[s])
                    labs = labs+lab+','
                labs = labs[:-1]
                labs = labs +'\t'+l+"\n"
                f_train.write(labs)
        i+=1
    print("数据列表生成完成！")
#创建数据字典
def creat_data_dic(data_path,dict_path):
    i = 0
    dict_txt ={}
    dict_set = set()
    with open(data_path,"r",encoding="utf-8") as f1:
        lines = f1.readlines()
    for line in lines:
        title = line.split("_!_")[-1].replace("\n","")
        for s in title:
            dict_set.add(s)
    for s in dict_set:
        dict_txt.update({s:i})
        i+=1
    end_dict ={"<unk>":i}
    dict_txt.update(end_dict)
    with open(dict_path,"w",encoding="utf-8") as f2:
        f2.write(str(dict_txt))
    print("数据字典生成完成！")

if __name__ == "__main__":
    data_root_path = '/home/bland/note/demo/wenbenfenlei'
    data_path = os.path.join(data_root_path, "news_classify_data.txt")
    dict_path = os.path.join(data_root_path, "dict_txt.txt")
    creat_data_dic(data_path,dict_path)
    creat_data_list(data_root_path)
    
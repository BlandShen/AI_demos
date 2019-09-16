import sys
from bs4 import BeautifulSoup
import re
import urllib
from urllib import request
import xlwt

#解析url
def askURL(url):
    request = urllib.request.Request(url)#发送url请求
    try:
        response = urllib.request.urlopen(request)
        html = response.read()
        print("{0} 爬取成功！".format(url))
    except urllib.error.URLError as e:
        print("{0}爬取失败！".format(url))
        #if hasattr(e,"code"):
            #print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html

#解析url成功后，我们要从获取的html中提取有用的部分
def getData(baseurl):
    findLink = re.compile(r'<a href="(.*?)">')#规则语句
    findImgSrc = re.compile(r'<img.*src="(.*?)"',re.S)
    findTitle = re.compile(r'<span class="title">(.*)</span>')
    #findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
    #findJudge = re.compile(r'<span>(\d*)人评价</span>')
    #findInq = re.compile(r'<span class="inq">(.*)</span>')
    findBd = re.compile(r'<p class="">(.*?)</p>',re.S)
    remove = re.compile(r'                                                                  |</br>|\n|\.*')
    datalist = []

    for i in range(0,10):
        url = baseurl + str(i*25)
        html = askURL(url)
        soup = BeautifulSoup(html,"html.parser")
        for item in soup.find_all('div',class_='item'):
            data=[]#暂时存储数据的列表
            item = str(item)
            link = re.findall(findLink,item)[0]#根据规则语句，寻找data
            data.append(link)
            imgSrc = re.findall(findImgSrc,item)[0]
            data.append(imgSrc)
            titles = re.findall(findTitle,item)[0]
            data.append(titles)
            bd = re.findall(findBd,item)[0]
            bd = re.sub(remove,"",bd)
            bd = re.sub(r'<br(\s+)?\/?>(\s+)?'," ",bd)
            bd = re.sub(r'/'," ",bd)
            #data.append(bd.strip())#除去前后空白
            datalist.append(data)
    return datalist

def showData(datalist):
    print("{:*^70}".format("爬取豆瓣TOP250电影数据"))
    for item in datalist:
        print(item)

if __name__ == "__main__":
    print("开始爬取数据......")
    baseurl = 'https://movie.douban.com/top250?start='
    datalist = getData(baseurl)
    showData(datalist)
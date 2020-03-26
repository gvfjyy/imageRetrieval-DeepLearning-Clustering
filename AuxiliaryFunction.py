# -*- coding: utf-8 -*-
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn import preprocessing
import cv2

#图像预处理
def preprocess_image(filename):
    img=Image.open(filename)
    img=img.convert('L') 
    img=img.resize((64,64),Image.ANTIALIAS)
    img=(np.array(img)/255.0).reshape(-1,64,64,1)
    return img

#将图像变为方形
def pretreatment_image():
    img=cv2.imread('search_image.jpg')#opencv只会加载3通道图像，即便原图是单通道也会被改为3通道
    rows,cols,channels=img.shape#行数、列数、通道数
    if rows==cols or abs(rows-cols)<20:
        return
    aver=np.mean(img)
    length=max(rows,cols)
    image=np.zeros((length,length,channels),dtype=np.uint8)
    if aver>200:
    	image=np.ones((length,length,channels),dtype=np.uint8)*255
    if rows>cols:
        image[:,int((rows-cols)/2):int((rows-cols)/2+cols),:]=img
    elif rows<cols:
        image[int((cols-rows)/2):int((cols-rows)/2+rows),:,:]=img
    cv2.imwrite('search_image.jpg',image)
    
#将图像统一修正为白色背景
def correct_background():
    img=cv2.imread('search_image.jpg', 0)
    aver=np.mean(img)
    if aver>200:
        return
    ret,image=cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV) 
    #image=cv.medianBlur(image,5)#中值滤波降噪
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('search_image.jpg', image)

#获取特征
def get_feature(sess1,graph1,sess2,graph2):
    input=preprocess_image('search_image.jpg')
    feed_dict1 = {
            graph1['images']: input,
            graph1['keep_prob']: 1.0,
            graph1['is_training']: False 
        }
    feature1=sess1.run(graph1['fc1'], feed_dict1).reshape(1,-1)
    feature1=preprocessing.normalize(feature1, norm='l2').reshape(-1,1024)#L2归一化
    feed_dict2 = {
            graph2['images']:feature1 ,
            graph2['keep_prob']: 1.0,
            graph2['is_training']: False 
        }
    feature2=sess2.run(graph2['fc1'], feed_dict2).reshape(1,-1)
    feature2=preprocessing.normalize(feature2, norm='l2').reshape(-1)
    return feature2


#加载数据
def load_data(filename):
    result=[]
    with open(filename,encoding='utf-8') as f:
        for line in f.readlines():
            temp=line.strip('\n').split('[',1)#分开文件名和数据
            t=[]
            t.append(temp[0].strip(' '))#文件名
            t.append(temp[1].strip('[').strip(']').strip(' ').split(','))#数据
            for i in range(len(t[1])):
                t[1][i]=float(t[1][i])
            t[1]=np.array(t[1])
            t.append(0.)#存放两个样本间的距离
            result.append(t)
    return result

#计算欧氏距离
def dist_euc(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def search(feature,data,rank):#某一个图像的特征feature+特征数据集data+返回前rank个结果
    result=[]
    vec1=feature
    if len(vec1)==0:#待检索图像特征
        return result
    for i in range(len(data)):
        vec2=data[i][1]
        data[i][2]=dist_euc(vec1,vec2)
    data.sort(key=lambda x:x[2])#排序
    for j in range(rank):
        result.append(data[j][0])
    return result

#清空文件夹
def clear_folder(folder):
    full=os.listdir(folder)
    for i in full:
        path=os.path.join(folder,i)
        os.remove(path)

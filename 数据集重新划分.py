# -*- coding: utf-8 -*-
import os
from private_library import *
from PIL import Image


def folder_divide(name):
    #imagedata\001000-001262\001000\001000b00036.jpg
    site=len(name)-1
    while name[site]!='\\' and name[site]!='/':
        site=site-1
    return name[:site],name[site:]



def divide(name):
    data=load_data(name,2)
    num=0
    for i in range(len(data)):
        if num<max(data[i][1]):
            num=max(data[i][1])
    num=int(num+1)
    #根据类数创建新的文件夹
    head,tail=folder_divide(data[0][0])
    for i in range(num):
        temp=head+'\\'+str(i)
        if not os.path.exists(temp):
            os.makedirs(temp)
    #文件剪切
    for i in range(len(data)):
        img=Image.open(data[i][0])
        head,tail=folder_divide(data[i][0])
        temp=head+'\\'+str(int(max(data[i][1])))+tail
        img.save(temp)
        os.remove(data[i][0])
        


if __name__=='__main__':
    file=get_filename('result')
    print('length',len(file))
    for t in range(len(file)):
        print(file[t],'   ',t)
        divide(file[t])






    





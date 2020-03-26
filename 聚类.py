# -*- coding: utf-8 -*-
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from private_library import *


if __name__=='__main__':
    #获取未处理的数量
    yichuli=get_filename('result')#已处理
    zongji=folder_bottom('imagedata')#总的
    weichuli=[]#未处理
    x=load_data('model1_fc1_l2',2)
    for i in range(len(yichuli)):
        yichuli[i]=yichuli[i][7:]
    for i in range(len(zongji)):
        zongji[i]=zongji[i][len(zongji[i])-6:]
    for i in range(len(zongji)):
        if zongji[i] not in yichuli:
            weichuli.append(zongji[i])
    print('未处理：',len(weichuli))

    #循环处理
    for i in range(len(weichuli)):
        #获取数据
        print('processing: ',i,' ',weichuli[i])
        name=[]
        datas=[]
        for j in range(len(x)):
            if get_class(x[j][0])==weichuli[i]:
                name.append(x[j][0])
                datas.append(x[j][1])
        datas=np.array(datas)
        if len(name)<9:
            continue
        length=max(len(name)-10,np.int(np.sqrt(datas.shape[0])))
        length=min(length,20)#聚类范围

        #确当最好的k值
        result=[]
        for j in range(2,length):
            temp=[]
            kmeans_model = KMeans(n_clusters=j,init='k-means++').fit(datas)
            temp.append(j)
            temp.append(metrics.silhouette_score(datas,kmeans_model.labels_,metric='euclidean'))
            temp.append(kmeans_model.labels_)
            result.append(temp)

        #按照结果最好的标签保存
        result.sort(key=lambda x:x[1])
        res=result[len(result)-1]
        f=codecs.open('result\\'+get_class(name[0]),'w','utf-8')
        for j in range(len(name)):
            f.write(name[j]+' ['+str(res[2][j])+']\n')
        f.close()

    





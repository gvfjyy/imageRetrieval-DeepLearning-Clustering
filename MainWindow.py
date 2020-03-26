# -*- coding: utf-8 -*-
from Ui_MainWindow import Ui_MainWindow
#引入Qt模块
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QStandardPaths
from PyQt5.QtGui import QPixmap
#
import os, math
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#指定tensorflow使用cpu
import cv2 as cv
import numpy as np
from PIL import Image
from IdentificationNetwork1 import *
from IdentificationNetwork2 import *
import tensorflow as tf
from AuxiliaryFunction import *

class MainWindow(QMainWindow, Ui_MainWindow):
    #变量
    page=0#记录当前页；首页为0，框选页为1，结果页为2
    search_results=[]#存储搜索结果，搜到的图像的名字
    status_roi='rect'#记录框选方式rect和dot
    first=QPoint(0, 0)
    final=QPoint(0, 0)
    contours=[]#记录轮廓的列表
    status_mouse='left'#记录鼠标左右键
    previous_page=0
    col_page0=0
    path_image_display=''
    graph1=tf.Graph()
    graph2=tf.Graph()
    sess1 = tf.Session(graph=graph1) 
    sess2 = tf.Session(graph=graph2)
    data=None#图像特征
    rank=100#检索结果个数
    def __init__(self, parent=None):#构造函数
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        #信号与槽的连接
        self.search_image.clicked.connect(self.search_image_clicked)
        self.open_image.clicked.connect(self.select_image)
        self.process_image.clicked.connect(self.switch_page1)
        self.left2.clicked.connect(lambda:self.home(1))
        self.left3.clicked.connect(lambda:self.home(2))
        self.left4.clicked.connect(lambda:self.home(3))
        self.roi_dot.clicked.connect(self.roi_dot_clicked)
        self.roi_rect.clicked.connect(self.roi_rect_clicked)
        self.tableWidget.itemClicked.connect(self.tableWidget_clicked)
        self.right1.clicked.connect(self.switch_previous)
        self.reset.clicked.connect(lambda:self.update_page1('image_temp.jpg'))
        self.save.clicked.connect(self.roi_save)
        self.spinBox.valueChanged.connect(self.spinBox_valueChanged)
        #初始化
        self.roi_rect.setStyleSheet("background-color:rgb(152,251,152);")
        self.page=0
        self.previous_page=0
        self.path_image_display=''
        self.data=load_data('图像特征')
        self.rank=200
        with self.sess1.as_default():
            with self.graph1.as_default():
                self.graph1=build_graph(top_k=3, charset_size=5491)
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint('checkpoint1')
                if ckpt:
                    saver.restore(self.sess1, ckpt)

        with self.sess2.as_default():
            with self.graph2.as_default():
                self.graph2=build_graph2(top_k=3, charset_size=7997)
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint('checkpoint2')
                if ckpt:
                    saver.restore(self.sess2, ckpt)

    def spinBox_valueChanged(self, event):
        self.pushButton.setText("搜索结果显示个数："+str(self.spinBox.value()))
        self.rank=self.spinBox.value()
        return
        
    def resizeEvent(self, event):#窗口大小改变事件
        #QMessageBox.about(None,"Information",str(self.width())+','+str(self.height()))
        if self.page==1:
            return
        elif self.page==0:
            if len(self.search_results)==0:
                return
            else:
                self.update_results()
        elif self.page==2:
            self.update_page2()
    def tableWidget_clicked(self, item):
        self.tableWidget2.setColumnCount(0)
        self.tableWidget2.setRowCount(0)
        row=self.tableWidget.currentItem().row()
        row=int(math.floor(row/2))
        col=self.tableWidget.currentItem().column()
        #QMessageBox.about(None,"Information",str(row)+','+str(col))
        self.path_image_display=self.search_results[row*self.col_page0+col]
        if self.path_image_display[9]!='2':
            self.update_page2()
        else:
            self.update_page3()
    
    
    def update_page3(self):
        self.stackedWidget.setCurrentIndex(3)#切换窗口
        self.page=3
        path=self.path_image_display
        head=path[11:]
        site=0
        for j in range(len(head)):
            if head[j]=='-':
                site=j
                break
        num=head[site+1:len(head)-4]
        #QMessageBox.about(None,"Information", head+' '+num)
        num=int(num)
        head=head[:site]
        img=Image.open('rubbing/'+head+'.jpg')
        img.save('rubbing.jpg')
        img=cv.imread('rubbing.jpg')
        back=np.array(Image.open('tif/'+head+'.tif'))
        print(back.shape)
        row, col=back.shape
        a=9999
        b=9999
        c=0
        d=0
        for i in range(row):
            for j in range(col):
                if back[i][j]!=num:
                    continue
                if i<a:
                    a=i
                if j<b:
                    b=j
                if i>c:
                    c=i
                if j>d:
                    d=j
            
        cv.rectangle(img,(b,a),(d,c),(0,0,255),2)#RGB与BGR
        cv.imwrite('rubbing.jpg', img)
        img=QPixmap('rubbing.jpg')
        desktop = QtWidgets.QApplication.desktop()#获取屏幕大小
        if img.width()>desktop.width()*0.8:
            img=img.scaled(desktop.width()*0.8, img.height()*(desktop.width()*0.8/img.width()))
        if img.height()>desktop.height()*0.8:
            img=img.scaled(img.width()*(desktop.height()*0.8/img.height()), desktop.height()*0.8)
        self.rubbing.setMinimumSize(QtCore.QSize(img.width(), img.height()))
        self.rubbing.setMaximumSize(QtCore.QSize(img.width(), img.height()))
        self.rubbing.setPixmap(img)
        
        head=head[1:]
        flag=0
        str=''
        with open('qsj', encoding='utf-8') as f:
            for line in f.readlines():
                if flag==0:
                    if line[0]=='H' and (line.split(',')[0])[1:]==head:
                        str=str+line
                        flag=1
                elif flag!=0:
                    if line[0]!='H' and (line[0]<'A' or line[0]>'Z'):
                        str=str+line
                    else:
                        break
        temp=str.split(',')
        if len(temp)>=2:
            self.textEdit.setPlainText('释文\n'+temp[1])
        if len(temp)>=3:
            self.textEdit2.setPlainText('原文\n'+temp[2])
        
        
    def switch_previous(self):#切换至上一个页面
        self.stackedWidget.setCurrentIndex(self.previous_page)
        self.page=self.previous_page
    def home(self, site):
        self.stackedWidget.setCurrentIndex(0)
        self.page=0
        if site!=0:
            self.previous_page=site
        return
    def switch_page1(self):
        self.stackedWidget.setCurrentIndex(1)#切换窗口
        self.page=1
        self.update_page1('image_temp.jpg')
        self.status_roi='rect'
        self.spinBox.setEnabled(True)
        return
    def update_page2(self):#进入page2
        self.stackedWidget.setCurrentIndex(2)#切换窗口
        self.page=2
        path=self.path_image_display
        head=path[:30]
        tail=path[31:]
        img=QPixmap(path).scaled(100, 100)
        self.image_selected.setPixmap(img)
        #QMessageBox.about(None,"Information", head+','+tail)
        '''设置文字信息'''
        self.num_same.setText('异形字个数 '+str(len(os.listdir(head))))
        shoubu=[]
        ziku=[]
        with open('ziku','r', encoding='UTF-8') as f:
            for line in f.readlines():
                temp=line.strip('\n').strip(' ').split(',')
                ziku.append(temp)
                shoubu.append(temp[0])
        site=shoubu.index(path[24:30])
        info=ziku[site]
        self.jtz.setText('简体字 '+str(info[5]))
        self.ftz.setText('繁体字 '+str(info[2]))
        self.ldz.setText('隶定字 '+str(info[3]))
        self.jgz.setText('甲骨字 '+str(info[1]))
        '''设置tablewidget'''
        full=os.listdir(head)
        length=len(full)
        num_col=int(self.tableWidget2.width()/100)#每行的个数
        num_row=int(math.ceil(length/num_col))
        num_row=num_row*2
        self.tableWidget2.clear()
        #self.tableWidget2.setMaximumSize(QtCore.QSize(100*num_col+2, 10000))#设置控件自适应大小
        self.tableWidget2.setColumnCount(num_col)  
        self.tableWidget2.setRowCount(num_row) 
        self.tableWidget2.verticalHeader().setVisible(False)#屏蔽表头和行号
        self.tableWidget2.horizontalHeader().setVisible(False)
        self.tableWidget2.setIconSize(QSize(96,96));
        #self.tableWidget2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        #self.tableWidget.setShowGrid(False)
        for i in range(num_col):   # 让列宽和图片相同  
            self.tableWidget2.setColumnWidth(i , 100)
        for i in range(num_row):
            if i%2==0:
                self.tableWidget2.setRowHeight(i , 100)  
            else:
                self.tableWidget2.setRowHeight(i , 25)  
        for k in range(length):
            i = int(math.floor(k/num_col))*2
            j = int(k%num_col)
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled)
            if tail==full[k]:
                #QMessageBox.about(None,"Information",str(site_image))
                #QMessageBox.about(None,"Information", head+'/'+tail)
                img=Image.open(os.path.join(head, full[k]))
                img.save('temp.jpg')
                img=cv.imread('temp.jpg')
                x, y, _=img.shape
                cv.rectangle(img,(0,0),(x,y),(0,0,255),25)
                cv.imwrite('temp.jpg', img)
                item.setIcon( QIcon(QPixmap('temp.jpg')))
                self.tableWidget2.setItem(i,j,item)
                self.tableWidget2.setItem(i+1,j, QTableWidgetItem(full[k]))
                continue
            item.setIcon( QIcon(QPixmap(os.path.join(head, full[k]))))  
            self.tableWidget2.setItem(i,j,item)
            self.tableWidget2.setItem(i+1,j, QTableWidgetItem(full[k]))
    def roi_save(self):
        img = QPixmap('image_temp2.jpg')
        img.save('search_image.jpg')
        img=img.scaled(self.image2.width(), self.image2.width()/img.width()*img.height())#缩放图像使其适应控件宽度
        self.image2.setMinimumSize(QtCore.QSize(img.width(), img.height()))
        self.image2.setMaximumSize(QtCore.QSize(img.width(), img.height()))
        self.image2.setPixmap(img)
        self.home(1)
    def roi_dot_clicked(self):#选中描点框选
        self.status_roi='dot'
        self.roi_dot.setStyleSheet("background-color:rgb(152,251,152);")
        self.roi_rect.setStyleSheet("background-color:;")
    def roi_rect_clicked(self):#选中矩形框选
        self.status_roi='rect'
        self.roi_rect.setStyleSheet("background-color:rgb(152,251,152);")
        self.roi_dot.setStyleSheet("background-color:;")
    def update_page1(self, name):#刷新框选页面
        img=QPixmap(name)
        desktop = QtWidgets.QApplication.desktop()#获取屏幕大小
        if img.width()>desktop.width()*0.8:
            img=img.scaled(desktop.width()*0.8, img.height()*(desktop.width()*0.8/img.width()))
        if img.height()>desktop.height()*0.8:
            img=img.scaled(img.width()*(desktop.height()*0.8/img.height()), desktop.height()*0.8)
        self.image_roi.setMinimumSize(QtCore.QSize(img.width(), img.height()))
        self.image_roi.setMaximumSize(QtCore.QSize(img.width(), img.height()))
        self.image_roi.setPixmap(img)
        img.save('image_temp2.jpg')
        img.save('image_temp3.jpg')

    def select_image(self):#选择图像
        path, _ = QFileDialog.getOpenFileName(
            None, '请选择图片', QStandardPaths.writableLocation(QStandardPaths.DesktopLocation), '图片文件(*.jpg *.png)')
        if not path:
            return
        img = QPixmap(path)
        img.save('image_temp.jpg')#为图像设置备份
        img.save('image_temp2.jpg')
        img.save('image_temp3.jpg')
        img.save('search_image.jpg')
        img=img.scaled(self.image1.width(), self.image1.width()/img.width()*img.height())#缩放图像使其适应控件宽度
        self.image1.setMinimumSize(QtCore.QSize(img.width(), img.height()))
        self.image1.setMaximumSize(QtCore.QSize(img.width(), img.height()))
        self.image2.setMinimumSize(QtCore.QSize(img.width(), img.height()))
        self.image2.setMaximumSize(QtCore.QSize(img.width(), img.height()))
        self.image1.setPixmap(img)
        self.image2.setPixmap(img) 
        self.spinBox.setEnabled(True)
        return
    def search_image_clicked(self):
        self.spinBox.setEnabled(False)
        pretreatment_image()#图像预处理
        correct_background()#背景处理
        feature=get_feature(self.sess1,self.graph1,self.sess2,self.graph2)
        self.search_results=search(feature, self.data,self.rank)
        
        '''计算搜索结果前三的图像'''
        top=[]
        num=[]
        for i in range(len(self.search_results)):
            t=[]
            t.append(self.search_results[i].split('\\', 3)[2])
            if t not in top:
                top.append(t)
                num.append(1)
            else:
                num[top.index(t)]+=1
        for i in range(len(top)):
            top[i].append(num[i])
        top.sort(key=lambda x:x[1], reverse=True)
        '''
        if top[0][0]!='034000':
            top[2][0]=top[1][0]
            top[1][0]=top[0][0]
            top[0][0]='034000'
        '''
        #检索数量前三的图像的简体字
        shoubu=[]
        ziku=[]
        with open('ziku','r', encoding='UTF-8') as f:
            for line in f.readlines():
                temp=line.strip('\n').strip(' ').split(',')
                ziku.append(temp)
                shoubu.append(temp[0])
        #self.top1.setText('TOP1:'+top[0][0]+' '+str(top[0][1])+'%')
        self.top1.setText(top[0][0]+"："+str(ziku[shoubu.index(top[0][0])][5])+"："+str(top[0][1])+"个")
        self.top1.setStyleSheet("background-color:red;font-size:16px;text-align:center")
        
        #self.top1.setStyleSheet("background-color:red;color:green")
        if len(top)>=2:
            self.top2.setText(top[1][0]+"："+str(ziku[shoubu.index(top[1][0])][5])+"："+str(top[1][1])+"个")
            self.top2.setStyleSheet("background-color:green;font-size:16px;text-align:center")
        if len(top)>=3:
            self.top3.setText(top[2][0]+"："+str(ziku[shoubu.index(top[2][0])][5])+"："+str(top[2][1])+"个")
            self.top3.setStyleSheet("background-color:blue;font-size:16px;text-align:center")
            
        
        clear_folder('search_results')#清空文件夹
        for i in range(len(self.search_results)):
            img=Image.open(self.search_results[i])
            img.save('search_results\\'+str(i)+'.jpg')
        for i in range(len(self.search_results)):
            img=cv.imread('search_results\\'+str(i)+'.jpg')
            temp=(self.search_results[i].split('\\', 3)[3])[:6]
            if temp==top[0][0]:
                x, y, _=img.shape
                if x<110:
                    cv.rectangle(img,(0,0),(x,y),(0,0,255),5)#opencv是BGR颜色
                else:
                    cv.rectangle(img,(0,0),(x,y),(0,0,255),25)#opencv是BGR颜色
                cv.imwrite(('search_results\\'+str(i)+'.jpg'), img)
            elif temp==top[1][0]:
                x, y, _=img.shape
                if x<110:
                    cv.rectangle(img,(0,0),(x,y),(0,255,0),5)
                else:
                    cv.rectangle(img,(0,0),(x,y),(0,255,0),25)
                cv.imwrite(('search_results\\'+str(i)+'.jpg'), img)
            elif temp==top[2][0]:
                x, y, _=img.shape
                if x<110:
                    cv.rectangle(img,(0,0),(x,y),(255,0,0),5)
                else:
                    cv.rectangle(img,(0,0),(x,y),(255,0,0),25)
                cv.imwrite(('search_results\\'+str(i)+'.jpg'), img)
        self.update_results()#显示搜索结果
        return 
    def update_results(self):
        self.tableWidget.setColumnCount(0)  
        self.tableWidget.setRowCount(0)#初始化表格清空内容
        path='search_results'
        full=os.listdir(path)
        length=len(full)
        num_col=int(math.floor(self.tableWidget.width()/100))#列数
        self.col_page0=num_col
        num_row=int(math.ceil(length/num_col))#行数
        num_row=num_row*2
        #self.tableWidget.setMaximumSize(QtCore.QSize(100*num_col+2, 10000))#设置控件自适应大小
        self.tableWidget.setColumnCount(num_col)  
        self.tableWidget.setRowCount(num_row) 
        self.tableWidget.verticalHeader().setVisible(False)#屏蔽表头和行号
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.setIconSize(QSize(96,96));
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)#设置只可选中单个目标
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectItems) #只可选中单个单元格
        #self.tableWidget.setShowGrid(False)#是否显示网格

        for i in range(num_col):   # 让列宽和图片相同  
            self.tableWidget.setColumnWidth(i , 100)
        for i in range(num_row):
            if i%2==0:
                self.tableWidget.setRowHeight(i , 100)  
            else:
                self.tableWidget.setRowHeight(i , 25)  
        for k in range(length):
            i = int(math.floor(k/num_col))*2
            j = int(k%num_col)
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemIsEnabled)
            item.setIcon( QIcon(QPixmap('search_results/'+str(k)+'.jpg')))  
            self.tableWidget.setItem(i,j,item)
            '''字体设置
            textFont = QFont("song", 8, QFont.Bold)  
            newItem = QTableWidgetItem(self.search_results[k].split('/', 3)[3])
            newItem.setBackgroundColor(QColor(0,60,10))  
            newItem.setTextColor(QColor(200,111,100)) 
            newItem.setFont(textFont) 
            '''
            self.tableWidget.setItem(i+1,j,QTableWidgetItem(self.search_results[k].split('\\', 3)[3]))
            
        return
    def mousePressEvent(self, event):#鼠标按下事件
        if self.page!=1:
            return
        point=self.image_roi.mapFromGlobal(QCursor.pos())#鼠标相对于控件temp_image左上角的位置
        if event.buttons () == QtCore.Qt.LeftButton:
            self.status_mouse='left'
            if self.status_roi=='rect':
                 if point.x()>0 and point.x()<self.image_roi.width() and point.y()>0 and point.y()<self.image_roi.height():
                     self.setCursor(Qt.CrossCursor)#十字光标
                     self.first=point
            elif self.status_roi=='dot':
                x=point.x()
                y=point.y()
                if len(self.contours)==0:
                    self.contours.append((x, y))
                    img=cv.imread('image_temp3.jpg')
                    img=cv.circle(img,(x,y),1,(255,0,0),3)
                    cv.imwrite('image_temp3.jpg', img)
                    self.image_roi.setPixmap(QPixmap('image_temp3.jpg'))
                elif len(self.contours)>0:
                    temp=self.contours[len(self.contours)-1]
                    if temp[0]!=x or temp[1]!=y:
                        self.contours.append((x, y))
                        img=cv.imread('image_temp3.jpg')
                        img=cv.circle(img,(x,y),1,(255,0,0),3)
                        img=cv.line(img,(temp[0],temp[1]),(x,y),(0,255,0),1)
                        cv.imwrite('image_temp3.jpg', img)
                        self.image_roi.setPixmap(QPixmap('image_temp3.jpg'))
        elif event.buttons () == QtCore.Qt.RightButton:
            self.status_mouse='right'
        return
    def mouseReleaseEvent (self, event):#鼠标释放事件
        if self.page!=1:
            return
        self.setCursor(Qt.ArrowCursor)#标准光标
        point=self.image_roi.mapFromGlobal(QCursor.pos())#鼠标相对于控件temp_image左上角的位置
        if self.status_mouse=='left':
            if self.status_roi=='dot':
                return
            #QMessageBox.about(None,"Information",'左释放')
            if not (point.x()>0 and point.x()<self.image_roi.width() and point.y()>0 and point.y()<self.image_roi.height()):
                QMessageBox.about(None,"Information",'坐标超出界限')
                return
            self.final=point
            if self.first==self.final:
                return
            a=min(self.first.x(), self.final.x())
            b=max(self.first.x(), self.final.x())
            c=min(self.first.y(), self.final.y())
            d=max(self.first.y(), self.final.y())
            img=cv.imread('image_temp2.jpg')
            img = img[c:d,a:b]
            cv.imwrite('image_temp2.jpg', img)
            cv.imwrite('image_temp3.jpg', img)
            self.image_roi.setPixmap(QPixmap('image_temp2.jpg'))
            self.update_page1('image_temp2.jpg')
        if self.status_mouse=='right':
            temp=self.contours
            length=len(temp)
            temp=np.array(temp).reshape(length, 1, 2)
            vect=[]
            vect.append(temp)
            img=cv.imread('image_temp2.jpg')#原图
            '''根据像素平均值确定背景色'''
            aver=np.mean(img)
            background=np.zeros(img.shape, np.uint8)
            if aver>200:
                background=np.ones(img.shape,np.uint8)*255#背景颜色根据图像均值确定
            '''抠图'''
            mask=np.ones(img.shape,np.uint8)*255
            cv.drawContours(mask,vect,-1,(0,0,0),-1)
            mask=(mask/255).astype(np.uint8)#选中区域为0，其余为1 #注意数据类型的转换。。。搞了这么久错误就出在这
            mask_inv=mask.astype(np.int8)
            mask_inv=((mask-1)*-1).astype(np.uint8)#相反
            img=np.multiply(img,mask_inv)
            background=np.multiply(background,mask)
            image=img+background
            '''获取最小外接矩形'''
            temp=temp.reshape(length, 2)
            temp_x=temp[:, :1].reshape(length)
            temp_y=temp[:, 1:].reshape(length)
            a=min(temp_x)
            b=max(temp_x)
            c=min(temp_y)
            d=max(temp_y)
            image=image[c:d,a:b]
            
            cv.imwrite('image_temp2.jpg', image)
            cv.imwrite('image_temp3.jpg', image)
            self.update_page1('image_temp2.jpg')
            self.contours=[]

            
    def mouseMoveEvent(self, event):#鼠标移动事件
        if self.page!=1 or self.status_roi=='dot':
            return
        point=self.image_roi.mapFromGlobal(QCursor.pos())#鼠标相对于控件temp_image左上角的位置
        img=cv.imread('image_temp2.jpg')
        img=cv.rectangle(img, (self.first.x(), self.first.y()), (point.x(), point.y()), (0, 255, 0), 1)
        cv.imwrite('image_temp3.jpg', img)
        self.image_roi.setPixmap(QPixmap('image_temp3.jpg'))
        return
    
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.showMaximized()
    sys.exit(app.exec_())

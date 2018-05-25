#导入模块
import time
time.clock()
import numpy as np
from skimage import io,color,filters
from numba import jit
import os


#定义函数
@jit
def depoint(img):#八邻域降噪

    """传入二值化后的图片进行降噪"""
    pixdata = img
    w,h = img.shape
    #去除图像边缘噪点
    for x in range(w):
        pixdata[x,h-1]=255
        pixdata[x,0]=255
    for x in range(h):
        pixdata[w-1,x]=255
        pixdata[0,x]=255
    #对一个像素点周八个点进行搜寻并统计白点数量
    i = 0
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y] == 0 :
                if pixdata[x,y-1] == 255:#上
                    count = count + 1
                if pixdata[x,y+1] == 255:#下
                    count = count + 1
                if pixdata[x-1,y] == 255:#左
                    count = count + 1
                if pixdata[x+1,y] == 255:#右
                    count = count + 1
                if pixdata[x-1,y-1] == 255:#左上
                    count = count + 1
                if pixdata[x-1,y+1] == 255:#左下
                    count = count + 1
                if pixdata[x+1,y-1] == 255:#右上
                    count = count + 1
                if pixdata[x+1,y+1] == 255:#右下
                    count = count + 1
                if count > 5:#如果周围白点数量大于5则判定为噪点
                    i = i+1 #统计该次降噪处理了多少个噪点
                    pixdata[x,y] = 255
    return i
    
@jit
def deline(img):#去除干扰线，将长段干扰线截断
    pixdata = img
    w,h = img.shape
    n = 0
    for x in range(1,w-1):
        list = []
        y=1
        while(y<h-1):
            m=y
            count = 0
            while (m<h-1 and pixdata[x,m]==0):#当y点是黑色的，就跳入循环来计数y点下面的黑点个数
                count=count+1
                m = m+1
            if (count <=1 and count>0):#判断黑色的线条垂直宽度是否小于2px，如果小于2px就跳入循环，把他们记录到list表里
                c=count
                while c>0:
                    list.append(m-c)
                    c=c-1

            y=y+count+1
# 去掉纵向的干扰线，把找到的黑点改成白点
        if len(list)!=0:
            # print('list content')
            i=1
            while i < len(list):
                # print(x,list[i])
                pixdata[x,list[i]] = 255
                i=i+1

    for y in range(1,h-1):
        list = []
        x=1
        while(x<w-1):
            m=x
            count = 0
            while (m<w-1 and pixdata[m,y]==0):#当x点是黑色的，就跳入循环来计数x点下面的黑点个数
                count=count+1
                m = m+1
                # print(count)
            if (count <=1 and count>0):#判断黑色的线条水平宽度是否小于2px，如果小于2px就跳入循环，把他们记录到list表里
                # print(count)
                c=count
                while c>0:
                    list.append(m-c)
                    c=c-1

            x=x+count+1
# 去掉横向的干扰线，把找到的黑点改成白点
        if len(list)!=0:
        # print('list content')
            i=1
            while i < len(list):
                # print(x,list[i])
                pixdata[list[i],y] = 255
                i=i+1

@jit
def deline_1(img):#去除干扰线
    pixdata = img
    w,h = img.shape
    for x in range(1,w-1):
        if x > 1 and x != w-2:
           #获取目标像素点左右位置
            left = x - 1
            right = x + 1

        for y in range(1,h-1):
           #获取目标像素点上下位置
            up = y - 1
            down = y + 1
            if pixdata[x,y] < 5:
                if y > 1 and y != h-1:
                
                    #以目标像素点为中心点，获取周围像素点颜色
                    #0为黑色，255为白色
                    up_color = pixdata[x,up]
                    down_color =pixdata[x,down]
                    left_color = pixdata[left,y]
                    left_down_color = pixdata[left,down]
                    right_color = pixdata[right,y]
                    right_up_color = pixdata[right,up]
                    right_down_color = pixdata[right,down]
                    
                    #去除竖线干扰线
                    if down_color == 0:
                        if left_color == 255 and left_down_color == 255 and right_color == 255 and right_down_color == 255:
                            pixdata[x,y]=255
                    
                    #去除横线干扰线
                    elif right_color == 0:
                        if down_color == 255 and right_down_color == 255 and up_color == 255 and right_up_color == 255:
                            pixdata[x,y]=255



               #去除斜线干扰线
                if left_color == 255 and right_color == 255 and up_color == 255 and down_color == 255:
                    pixdata[x,y]=255

@jit
def main():
    path=os.getcwd()
    path_img=os.path.join(path,'train')
    path_save=os.path.join(path,'train_change')
    os.mkdir(path_save)
    img_file=os.listdir(path_img)
    for img in img_file:
        file_img=os.path.join(path_img,img)
        file_save=os.path.join(path_save,img)
        image = io.imread(file_img)
        image=color.rgb2gray(image) #灰度
        thresh = filters.threshold_otsu(image) #获得阈值
        image=(image >= thresh)*255#二值化
        #deline(image)
        depoint(image)
        depoint(image)
        io.imsave(file_save,image)
if __name__=='__main__':
    main()

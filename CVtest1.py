#!/usr/bin/env python
# coding: utf-8

# In[1]:


#答案卡辨識 簡單版

import cv2 as cv
import numpy as np

img = cv.imread('Photos/10.jpg')
h, w = img.shape[:2]
img2 = cv.resize(img, ((int(w/7), int(h/7)))) 

cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
cv.imshow('image', img2)
cv.waitKey(0)
cv.destroyAllWindows()


# In[2]:


image_HSV=cv.cvtColor(img2,cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([90,50,50])
upper_blue = np.array([150,255,255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(image_HSV, lower_blue, upper_blue)

mask_invert=cv.bitwise_not(mask);

# Bitwise-AND mask and original image
res = cv.bitwise_and(img2,img2, mask= mask_invert)

cv.imshow('frame',img2)
cv.imshow('mask',mask)
cv.imshow('mask_invert',mask_invert)
cv.imshow('res',res)

cv.waitKey(0)
cv.destroyAllWindows()


# In[36]:


gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

cv.namedWindow('gray', cv.WINDOW_AUTOSIZE)
cv.imshow('gray', gray)
cv.waitKey(0)
cv.destroyAllWindows()


# In[37]:


ret,th1 = cv.threshold(gray,100,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,61,60)
th3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

cv.namedWindow('image1', cv.WINDOW_AUTOSIZE)
cv.namedWindow('image2', cv.WINDOW_AUTOSIZE)
cv.namedWindow('image3', cv.WINDOW_AUTOSIZE)

cv.imshow('image1', th1)
cv.imshow('image2', th2)
cv.imshow('image3', th3)

cv.waitKey(0)

cv.destroyAllWindows()


# In[38]:


kernel = np.ones((3,3), np.uint8)
#erode ??? shoude dilate than erode
dilate = cv.erode(th2, kernel, iterations = 9)
dilate = cv.dilate(dilate, kernel, iterations = 7)

cv.namedWindow('dilate', cv.WINDOW_AUTOSIZE)
cv.imshow('dilate', dilate)
cv.waitKey(0)
cv.destroyAllWindows()


# In[39]:


import math 
def circle1(r):
    x=0
    y=0
    ans=[]
    for i in range(r+1):
        y=int(math.floor((r**2-i**2)**0.5))
        ans.append([i,y])
    return ans
#a=circle1(400)
#print(a)
#print(len(a))        
    


# In[40]:


def target_point(img):
    target_point1=[]
    y_max,x_max=dilate.shape
    #        LU    RU          LD          RD
    xy_max=[[0,0],[0,x_max-1],[y_max-1,0],[y_max-1,x_max-1]]
    
    for k in range(4):  #4 target point
        count, position_hit, position_not_hit, hit_first=0,0,0,0
        
        for j in range(x_max):    #every point
            r_point=circle1(j)
            for i in range(len(r_point)):     #every count
                if((img[abs(xy_max[k][0]-r_point[i][0]),
                        abs(xy_max[k][1]-r_point[i][1])])==0):    #hit
                
                    if(count==0):     #first point
                        target_point1.append([abs(xy_max[k][0]-r_point[i][0]),
                                             abs(xy_max[k][1]-r_point[i][1])])
#                        position_first=[abs(xy_max[k][0]-r_point[i][0]),
#                                        abs(xy_max[k][1]-r_point[i][1])]

                
                    hit_first=1
                    hit=1
                    count=count+1
                    #position=[r_point[i][0],r_point[i][1]]
                    #print(position)    #test
                    break
                hit=0
            
            if(hit_first==1 and hit==0 and count<10):  #hit
                position_hit=1
                break
            elif(hit==1 and count>=10):                #no hit
                position_not_hit=1
                loss_point=k
                break
        #print(xy_max[k][0],xy_max[k][1],count, position_hit, position_not_hit, hit_first)
    
    return target_point1,loss_point


# In[41]:


tp,tp_s=target_point(dilate)
print(tp,tp_s)


# In[42]:


dilate2=dilate.copy()
for i in range(4):
    cv.circle(dilate2,(tp[i][1],tp[i][0]),5,127,-1)
    
cv.circle(dilate2,(tp[tp_s][1],tp[tp_s][0]),10,80,-1)
cv.namedWindow('dilate2', cv.WINDOW_AUTOSIZE)
cv.imshow('dilate2', dilate2)
cv.waitKey(0)
cv.destroyAllWindows()


# In[43]:


def determine_direction(dir):
    if(abs(dir[0][0]-dir[1][0])<8):
        return 'count up'      #count up
    else:
        return 'count down'      #count down


# In[44]:


def determine_correct(dir):
    if(abs(dir[0][1]-dir[2][1])<8 and abs(dir[2][0]-dir[3][0])<8):
        return 'correct'     #correct
    else:
        return 'incorrect'     #incorrect


# In[45]:


determine_direction(tp)


# In[46]:


determine_correct(tp)


# In[47]:


def liftline(img):
    count, position_hit, position_not_hit, hit_first=0,0,0,0
    
    y_max,x_max=img.shape
    
    for j in range(x_max):
        r_point=circle1(j)
        for i in range(len(r_point)):
            if((dilate[r_point[i][0],r_point[i][1]])==0):    #hit
                
                if(position_hit==1):     #first point
                    #position_first=[y_max-1-r_point[i][0],x_max-1-r_point[i][1]]
                    return[r_point[i][1],r_point[i][0]]
                    
                
                hit_first=1
                hit=1
                count=count+1
                #position=[r_point[i][0],r_point[i][1]]
                #print(position)    #test
                
                break
            hit=0
            
        if(hit_first==1 and hit==0):
            position_hit=1 
           # print('sss')


# In[48]:


def liftline_down(img):
    count, position_hit, position_not_hit, hit_first=0,0,0,0
    
    y_max,x_max=img.shape
    
    for j in range(x_max):
        r_point=circle1(j)
        for i in range(len(r_point)):
            if((dilate[y_max-1-r_point[i][0],r_point[i][1]])==0):    #hit
                
                if(position_hit==1):     #first point
                    #position_first=[y_max-1-r_point[i][0],x_max-1-r_point[i][1]]
                    return[r_point[i][1],y_max-1-r_point[i][0]]
                    
                
                hit_first=1
                hit=1
                count=count+1
                #position=[r_point[i][0],r_point[i][1]]
                #print(position)    #test
                
                break
            hit=0
            
        if(hit_first==1 and hit==0):
            position_hit=1 
           # print('sss')


# In[49]:


liftline_point=liftline(dilate)
liftline_down_point=liftline_down(dilate)
print(liftline_point)
print(liftline_down_point)


# In[ ]:


dilate3=dilate.copy()

cv.circle(dilate3,(tp[tp_s][1],tp[tp_s][0]),3,127,-1)
cv.circle(dilate3,(liftline_point[0],liftline_point[1]),3,127,-1)
cv.circle(dilate3,(liftline_down_point[0],liftline_down_point[1]),3,127,-1)
cv.namedWindow('dilate3', cv.WINDOW_AUTOSIZE)
cv.imshow('dilate3', dilate3)
cv.waitKey(0)
cv.destroyAllWindows()


# In[28]:


y_line=liftline_down_point[1]-liftline_point[1]
x_line=tp[tp_s][1]-liftline_point[0]
print(y_line,x_line)


# In[29]:


num_y_ans=[]
num_x_ans=[]
for i in range(10):
    num_y_ans.append(math.floor(y_line/22*(i*2+1))+4)
for i in range(3):
    num_x_ans.append(math.floor(x_line/6*(i*2+1)))
    
print(num_y_ans,num_x_ans)


# In[30]:


num_xy_ans=[]
for i in num_y_ans:
    for j in num_x_ans:
        num_xy_ans.append([liftline_point[0]+i,liftline_point[1]+j])
print(num_xy_ans)


# In[31]:


dilate4=dilate.copy()

for i in range(len(num_xy_ans)):
    cv.circle(dilate4,(num_xy_ans[i][1],num_xy_ans[i][0]),3,127,-1)
    
cv.namedWindow('dilate4', cv.WINDOW_AUTOSIZE)
cv.imshow('dilate4', dilate4)
cv.waitKey(0)
cv.destroyAllWindows()


# In[32]:


num_ans=[]
for i in range(len(num_xy_ans)):
    if (dilate[num_xy_ans[i][0],num_xy_ans[i][1]]==0):
               num_ans.append(True)
    else:
               num_ans.append(False)
print(num_ans)


# In[33]:


read_ans=[]
dilate5=dilate.copy()
abc=['A','B','C']

for j in range(10):
    cv.putText(dilate5,'>', (1, 15+j*14), cv.FONT_HERSHEY_SIMPLEX,0.3, 0, 1, cv.LINE_AA)
    for i in range(3):
        if (num_ans[i+j*3]):
            cv.putText(dilate5, abc[i], (9+i*11, 15+j*14), cv.FONT_HERSHEY_SIMPLEX,0.4, 0, 1, cv.LINE_AA)
       
        read_ans.append([j,num_ans[i+j*3]])
    cv.putText(dilate5,'<', (40, 15+j*14), cv.FONT_HERSHEY_SIMPLEX,0.3, 0, 1, cv.LINE_AA)
        
cv.namedWindow('dilate5', cv.WINDOW_AUTOSIZE)
cv.imshow('dilate5', dilate5)
cv.waitKey(0)
cv.destroyAllWindows()
        
print(read_ans)
                


# In[ ]:





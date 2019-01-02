
# coding: utf-8

# In[19]:

import cv2
import os
import numpy as np
import pandas as pd
import mahotas as mt
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[29]:

ds_path = "leaf"
img_files = os.listdir(ds_path)


# In[21]:

def summation(image):
    sum = 0
    for i in range(image.shape[0]):
        for j in range (image.shape[1]):
            sum+=image[i][j]
            
    return sum


def lacunarity(image,p):
    den = 1/(image.shape[0]*image.shape[1])*summation(image)
    lac = 0
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            lac+= ((image[i][j]/den)-1)**p
            
    lac/=(image.shape[0]*image.shape[1])
    lac = lac**(1/p)
    return lac

def skewness(image,avg,stddev):
    skew = 0
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            skew += image[i][j]-avg
            
    skew = skew**3/(image.shape[0]*image.shape[1]*(stddev**3))
    return skew
    
def kurtosis(image,avg,stddev):
    kur = 0
    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            kur += image[i][j]-avg
            
    kur = kur**4/(image.shape[0]*image.shape[1]*(stddev**4))
    return kur

# def pft(image,M,max_rad):
#     cent_x = M["m10"]/M["m00"]
#     cent_y = M["m01"]/M["m00"]
#     fr = np.zeros((4,6))
#     fi = np.zeros((4,6))
#     fd = np.zeros(24)
#     for i in range(4):
#         for j in range(6):
#             fr[i][j] = 0
#             fi[i][j] = 0
#             for x in range(image.shape[0]):
#                 for y in range (image.shape[1]):
#                     radius = ((x-cent_x)**2+(y-cent_y)**2)**0.5
#                     theta = np.arctan2((y-cent_y),(x-cent_x))
#                     if(theta<0):
#                         theta = theta+2*3.14
#                     fr[i][j] = fr[i][j]+image[x][y]*np.cos(2*3.14*i*(radius/max_rad)+(j*theta))
#                     fi[i][j] = fi[i][j]-image[x][y]*np.sin(2*3.14*i*(radius/max_rad)+(j*theta))
#     for i in range(4):
#         for j in range(6):
#             if(i==0)and(j==0):
#                 dc = fr[0][0]**2+fi[0][0]**2
#                 fd[0]=dc/(2*3.14*(max_rad**2))
#             else:
#                 fd[i*4+j]=((fr[i][j]**2+fi[i][j]**2)**0.5)/dc
#     return fd
                    


# In[24]:

def create_dataset():
    names = ['area','perimeter','physiological_length','physiological_width','slimness','circularity','pft',              'mean_r','mean_g','mean_b','stddev_r','stddev_g','stddev_b','red_skew','green_skew','blue_skew','red_kurtosis','green_kurtosis','blue_kurtosis',              'contrast','correlation','inverse_difference_moments','entropy','lac_r_2','lac_r_4','lac_r_6','lac_b_2','lac_b_4','lac_b_6','lac_g_2','lac_g_4','lac_g_6','lac_gr_2','lac_gr_4','lac_gr_6'
            ]
    df = pd.DataFrame([], columns=names)
    for file in img_files:
        imgpath = ds_path + "\\" + file
        main_img = cv2.imread(imgpath)
        
#         Preprocessing
        img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
        gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gs, (25,25),0)
        ret_otsu,im_bw_otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((50,50),np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        
        #Shape features
        image, contours, _ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #plt.imshow(image)
        cnt = contours[0]
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        _,(w,h),_ = cv2.minAreaRect(cnt)     
        slimness = float(h)/w
        rectangularity = w*h/area
        roundness = (4*np.pi*area)/((perimeter)**2)
        
        #PFT
        
        _,maxrad = cv2.minEnclosingCircle(cnt)
#         fd = pft(image,M,max_rad)
        
        ff = np.fft.fft2(image)
        ff = np.abs(ff)
        ff.shape
        fd = np.zeros(ff.shape[0]*ff.shape[0]+ff.shape[1])
        for i in range(ff.shape[0]):
            for j in range(ff.shape[1]):
                if(i==0)and(j==0):
                    fd[0]=ff[i][j]/(2*3.14*(maxrad**2))
                                
                else:
                    fd[i*ff.shape[0]+j]=(ff[i][j])/ff[0][0]
    
        
        
        ##vein
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        A1 = cv2.morphologyEx(im_bw_otsu,cv2.MORPH_OPEN,kernel1)
        _, contvein1, _ = cv2.findContours(A1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cntvein1 = contvein1[0]
        a1 = cv2.contourArea(cntvein1)        
        A2 = cv2.morphologyEx(im_bw_otsu,cv2.MORPH_OPEN,kernel2)
        _, contvein2, _ = cv2.findContours(A2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cntvein2 = contvein2[0]
        a2 = cv2.contourArea(cntvein2)      
        
        A3 = cv2.morphologyEx(im_bw_otsu,cv2.MORPH_OPEN,kernel3)
        _, contvein3, _ = cv2.findContours(A3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cntvein3 = contvein3[0]
        a3 = cv2.contourArea(cntvein3)        
        
        V1 = float(a1)/area
        V2 = float(a2)/area
        V3 = float(a3)/area
        
        #Color features
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0
        
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)
        
        red_skew = skewness(red_channel,red_mean,red_std)
        green_skew = skewness(green_channel,green_mean,green_std)
        blue_skew = skewness(blue_channel,blue_mean,blue_std)
        
        red_kurtosis = kurtosis(red_channel,red_mean,red_std)
        green_kurtosis = kurtosis(green_channel,green_mean,green_std)
        blue_kurtosis = kurtosis(blue_channel,blue_mean,blue_std)
        
        #Texture features
        textures = mt.features.haralick(gs)
        ht_mean = textures.mean(axis=0)
        contrast = ht_mean[1]
        correlation = ht_mean[2]
        inverse_diff_moments = ht_mean[4]
        entropy = ht_mean[8]
        
        #####lacunarity
        lac_r_2 = lacunarity(red_channel,2)
        lac_r_4 = lacunarity(red_channel,4)
        lac_r_6 = lacunarity(red_channel,6)
        lac_b_2 = lacunarity(blue_channel,2)
        lac_b_4 = lacunarity(blue_channel,4)
        lac_b_6 = lacunarity(blue_channel,6)
        lac_g_2 = lacunarity(green_channel,2)
        lac_g_4 = lacunarity(green_channel,4)
        lac_g_6 = lacunarity(green_channel,6)
        lac_gr_2 = lacunarity(gs,2)
        lac_gr_4 = lacunarity(gs,4)
        lac_gr_6 = lacunarity(gs,6)
        
        
        vector = [area,perimeter,w,h,slimness,roundness,fd,                  red_mean,green_mean,blue_mean,red_std,green_std,blue_std,red_skew,green_skew,blue_skew,red_kurtosis,green_kurtosis,blue_kurtosis,                  contrast,correlation,inverse_diff_moments,entropy,lac_r_2,lac_r_4,lac_r_6,lac_b_2,lac_b_4,lac_b_6,lac_g_2,lac_g_4,lac_g_6,lac_gr_2,lac_gr_4,lac_gr_6
                 ]
        
        df_temp = pd.DataFrame([vector],columns=names)
        df = df.append(df_temp)
        print(file)
    return df


# In[ ]:

dataset = create_dataset()


# In[26]:

dataset.shape


# In[27]:

dataset.head


# In[28]:

dataset.to_csv("features.csv")


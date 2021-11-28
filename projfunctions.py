# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:47:00 2020

@author: marcs
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn import cluster
import pandas
from scipy import integrate
from scipy import ndimage
import sklearn
import skimage
import scipy.optimize as optimization
 


def nfloat (imagearray): #This function normalizes an array of floats so they belong to [0,1]
    M=np.max(imagearray)
    if M !=0:
        return imagearray/M
    return 0;


def CR(Border,portions):
    Splitted=np.array_split(Border,portions,1)
    RegionAverage=[np.mean(item,(0,1)) for item in Splitted]
    ColorRegions=[np.ones(Splitted[i].shape)*RegionAverage[i] for i in range(len(RegionAverage))]
    ColorRegions=np.concatenate(ColorRegions,1)
    return nfloat(ColorRegions)

def IC(Border,portions):
    
    Splitted=np.array_split(Border,portions,1)
    
    increment=[np.abs(
            np.mean(Splitted[i],(0,1))-np.mean(Splitted[i+1],(0,1))
            ) for i in range(len(Splitted)-1)]
    
    return np.stack(increment);

def ICHSV(Border,portions):
    BorderHSV=col.rgb_to_hsv(Border)
    Splitted=np.array_split(BorderHSV,portions,1)
    
    increment=[np.abs(
            np.mean(Splitted[i],(0,1))-np.mean(Splitted[i+1],(0,1))
        ) for i in range(len(Splitted)-1)]
    
    return np.stack(increment);


def HorizontalFinger(Border,portions,weight, th=0):
    increment=IC(Border,portions)
    incrementHSV=ICHSV(Border,portions)
    WeightedIncrement = increment@weight[:3] + incrementHSV@weight[3:6]
    firstMaxIndex=np.argmax(WeightedIncrement)
    detection= (WeightedIncrement[firstMaxIndex]>th)
    WeightedIncrement[firstMaxIndex]=0
    secondMaxIndex=np.argmax(WeightedIncrement)
    
    twoMax=[np.uint32((firstMaxIndex+1.5)*Border.shape[1]/portions), np.uint32((secondMaxIndex+1.5)*Border.shape[1]/portions)]
    
    
    if (twoMax[0]==twoMax[1]):
        return[twoMax[0]-1,twoMax[0]+1], False
    elif(twoMax[0]<twoMax[1]):
        return twoMax, detection;
    else:
        return [twoMax[1],twoMax[0]], detection;
    
    
    
    
def VerticalFinger(VerticalRegion,portions,weight):
    Trans=cv2.transpose(VerticalRegion)
    increment=IC(Trans,portions)
    WeightedIncrement = increment@weight
    Max=np.argmax(WeightedIncrement)
    return (Max+1.5)*Trans.shape[1]/portions;
    

def yfitScreenToPaper(x, a,b,c):#inverse of the calibration verison
#    if (b==x.all()):
#        return 0;
#    else:
    return a -c/(x-b); 


def xfitScreenToPaper(x, a,b,c):#inverse of the calibration verison
    return a +(x-b)/c;

def ycalibratefitScreenToPaper(x, a,b,c):#Function to be fitted
    return b -c/(x-a);    #definition of your function 

def xcalibratefitScreenToPaper(x, a,b,c):#Function to be fitted
    return b+c*(x-a);



def yCal(yCalibrationPointCM,yCalibrationPointPX,init=[-6.1, 0,0]):
    xdata = np.array(yCalibrationPointCM)
    ydata = np.array(yCalibrationPointPX)   
    initial = np.array(init) #Avoid x==a. Try to fit with x+0.1 or x-0.1 if it doesnt work
    sigma = 0.1*np.ones(xdata.shape)
    return optimization.curve_fit(ycalibratefitScreenToPaper, xdata, ydata,initial,sigma);


def xCal(xCalibrationPointCM,xCalibrationPointPX):
    xdata = np.array(xCalibrationPointCM)
    ydata = np.array(xCalibrationPointPX)   
    initial = np.array([0, 0,0])
    sigma = np.ones(xdata.shape)
    return  optimization.curve_fit(xcalibratefitScreenToPaper, xdata, ydata,initial,sigma);


def xCalParFunction(data,a,b,c,d):
    (x,y)=data
    return a+b*x+c*x*y+d*y;

def yCalParFunction(data,a,b,c,d):
    (x,y)=data
    return a+b*y+c*y**2+d*x*y;

def isFinger(Pix,fingerColor,treshold):
    Pixel=np.concatenate((Pix,col.rgb_to_hsv(Pix)),axis=2)
    accept=np.ones(Pixel.shape[:2])
    difference=np.abs(Pixel-fingerColor)
    for i in range(Pixel.shape[-1]):
        accept *= (difference[:,:,i]<=treshold[i])
    return accept;

def PositionFinger2(img,th=5):
    k=0
    lastlayerindex=0
    #a=[np.sum(VerticalFingerRegion[i]) for i in range(VerticalFingerRegion.shape[0])]
    for i in range(img.shape[0]):
       
        if np.sum(img[i])>0:#We check if layer i has any pixels different to 0 (here we can potentially apply an other threshold corresponding to how many bright pixels we need in the layer in order to count)
            lastlayerindex=i
            k=0
        else:
            k+=1
            
        if k >= th:#after some layers we stop looking for the tip of the finger
            break
    
    
    lastlayer=img[lastlayerindex]
    HorizontalIndexes=[j for j in range(lastlayer.shape[0]) if np.sum(lastlayer[j])!=0]
    if len(HorizontalIndexes)==0:
        midPixel=np.uint32(lastlayer.shape[0]/2)
    else:
        midPixel=np.uint32((HorizontalIndexes[0]+HorizontalIndexes[-1])/2)
    
    return(midPixel,lastlayerindex);
    
    
    
def Segmentation(img,A,B,hBorder,width,hportions,hweight,twoMax,th,thDetection=0.05,marginpart=0.4):
    Board= img[hBorder:,A[0]:B[0]]
    Border=Board[:width,:]  
    twoMax , detection=HorizontalFinger(Border,hportions,hweight,thDetection)
    VerticalFingerRegion=Board[:,twoMax[0]:twoMax[1],:]
    marge=np.uint32((twoMax[1]-twoMax[0])*marginpart)
    
    HSVBorder=col.rgb_to_hsv(Border)
    fingerColorHSV=np.concatenate( (Border[:,twoMax[0]+marge:twoMax[1]-marge,:].mean(axis=(0,1)) ,HSVBorder[:,twoMax[0]+marge:twoMax[1]-marge,:].mean(axis=(0,1))) ,axis=0)
    FingerBinary=isFinger(VerticalFingerRegion,fingerColorHSV,th)
    
    
    return Board, Border,twoMax,detection,VerticalFingerRegion,marge,fingerColorHSV[:3],FingerBinary;


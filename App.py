# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:58:48 2020

@author: marcs
"""


import tkinter as tk
import cv2
#from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows, imdecode, line
import projfunctions as own
import PIL.Image, PIL.ImageTk
import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.colors as col


capture=cv2.VideoCapture(0)

ret, frame = capture.read()
x,y,xCM,yCM,xPX,yPX= 0, 0,0 ,0 ,0,0
blackboard=np.zeros((frame.shape[0],frame.shape[0],frame.shape[-1]), dtype='uint8')



yCalibrationPointCM=[0.0, -2.0, -2.0, 2.0, 5.0, 5.0, -5.0, -5.0]   #Here you can save the value of your previous calibration process
yCalibrationPointPX=[195, 217, 219, 177, 157, 155, 269, 267]
xCalibrationPointCM=[0.0, -2.0, 2.0, -2.0, 5.0, -5.0, -3.0, 3.0]
xCalibrationPointPX=[303, 218, 384, 229, 441, 162, 143, 457]

xparam, devx = optimization.curve_fit(own.xCalParFunction, (xCalibrationPointPX,yCalibrationPointPX), xCalibrationPointCM)
yparam, devy = optimization.curve_fit(own.yCalParFunction, (xCalibrationPointPX,yCalibrationPointPX), yCalibrationPointCM)

xCMexp, yCMexp = 0,0

width=20
hportions=40
A , B= (np.uint32(frame.shape[1]*0.25),np.uint32(frame.shape[0]*0.25)),(np.uint32(frame.shape[1]*0.75),np.uint32(frame.shape[0]*0.25))
hBorder=np.min([A[1],B[1]])
twoMax=[A[0],B[0]]


root=tk.Tk()
    
def leftclick(event):
    global A,B,hBorder
    P=(event.x, event.y)
    
    if P[0]<B[0]-10:
        A=P 
        
    hBorder=np.uint32(0.5*(A[1]+B[1]))

def rightclick(event):
    global A,B,hBorder
    P=(event.x, event.y)
    if P[0]>A[0]+10:
        B=P
    hBorder=np.uint32(0.5*(A[1]+B[1]))
    
def erase(event):
    global blackboard
    blackboard*=0

def ColorCalibration(event):   
    increment=own.IC(Border,hportions)
    incrementHSV=own.ICHSV(Border,hportions)
    WeightedIncrement = increment@hweight[:3]+incrementHSV@hweight[3:6]
    


    x=np.linspace(A[0],B[0],len(WeightedIncrement))
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.plot(x, -1000*increment[:,0], label='$RED$',color='red')
    ax.plot(x, -1000*increment[:,1], label='$GREEN$',color='green')
    ax.plot(x, -1000*increment[:,2], label='$BLUE$',color='blue')
    ax.plot(x, -1000*incrementHSV[:,1], label='$Saturation$',color='orange')
    ax.plot(x, -1000*incrementHSV[:,2], label='$Value$',color='purple')
    ax.plot(x, -1000*WeightedIncrement, label='$WEIGHT$',color='black',linestyle='none',marker='+')
    fig.legend()
    fig.show()

    VFRHSV=col.rgb_to_hsv(VFRCopy)

    fig, ax = plt.subplots(1,8)
    ax[0].imshow(FingerBinary,cmap='gray')
    ax[1].imshow(VFRCopy)
    ax[2].imshow(VFRHSV[:,:,0],cmap='gray')
    ax[3].imshow(VFRHSV[:,:,1],cmap='gray')
    ax[4].imshow(VFRHSV[:,:,2],cmap='gray')
    ax[5].imshow(VFRCopy[:,:,0],cmap='gray')
    ax[6].imshow(VFRCopy[:,:,1],cmap='gray')
    ax[7].imshow(VFRCopy[:,:,2],cmap='gray')
    fig.show()


    
def xyData():
    global xparam,yparam,xCalibrationPointPX,yCalibrationPointPX, xCalibrationPointCM,yCalibrationPointCM,xPX,yPX
    a=np.double(xentry.get())
    b=np.double(yentry.get())
    if xentry.get():
        xCalibrationPointPX+=[xPX]
        yCalibrationPointPX+=[yPX]
        xCalibrationPointCM+=[a]
        yCalibrationPointCM+=[b]
    
    if(len(xCalibrationPointPX)>=5):
        xparam, devx = optimization.curve_fit(own.xCalParFunction, (xCalibrationPointPX,yCalibrationPointPX), xCalibrationPointCM)
        yparam, devy = optimization.curve_fit(own.yCalParFunction, (xCalibrationPointPX,yCalibrationPointPX), yCalibrationPointCM)

        
def Reset():
    global xCalibrationPointPX,yCalibrationPointPX, xCalibrationPointCM,yCalibrationPointCM
    print(xCalibrationPointCM,yCalibrationPointCM)
    yCalibrationPointCM=[]
    yCalibrationPointPX=[]
    xCalibrationPointCM=[]
    xCalibrationPointPX=[]
    
    
    
def getXYB(xcm,ycm):
    global xCalibrationPointCM,yCalibrationPointCM,xparam,yparam
    
    if (len(xCalibrationPointPX)>=5):
        maxX=max(xCalibrationPointCM)
        minX=min(xCalibrationPointCM)
        maxY=max(yCalibrationPointCM)
        minY=min(yCalibrationPointCM)
        L=max(maxX-minX,maxY-minY)
        px=(xcm-minX)/L
        py=(ycm-minY)/L
    else:
        px=0
        py=0
    return min(1,max(0,px)),min(1,max(0,py));    


# Then, you will define the size of the window in width(312) and height(324) using the 'geometry' method
root.geometry(str(2*frame.shape[1])+"x"+str(50+frame.shape[1]))

# In order to prevent the window from getting resized you will call 'resizable' method on the window

#root.resizable(0, 0)

   
textPX = tk.StringVar()
textCM = tk.StringVar()
coord_frame = tk.Frame(master=root, width = 312, height = 50, bd = 0, highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
coord_frame.pack(side = tk.TOP)
left_field = tk.Entry(master=coord_frame, font = ('arial', 18, 'bold'), textvariable = textCM, width = 25, bg = "#eee", bd = 0, justify = tk.LEFT)
left_field.pack(side=tk.LEFT, ipady = 10) # 'ipady' is an internal padding to increase the height of input field
right_field = tk.Entry(master=coord_frame, font = ('arial', 18, 'bold'), textvariable = textPX, width = 25, bg = "#eee", bd = 0, justify = tk.RIGHT)
right_field.pack(side=tk.RIGHT,ipady = 10)




canv=tk.Canvas(root,width=2*frame.shape[1],height=frame.shape[0])
canv.pack()
canvimage=canv.create_image(0, 0, image=PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame)), anchor=tk.NW)
canvimage2=canv.create_image(frame.shape[1], 0, image=PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(blackboard)) , anchor=tk.NW)

canv.bind('<B1-Motion>',leftclick)
canv.bind('<B3-Motion>',rightclick)




TH_frame= tk.Frame(master=root, width = 312, height = 50, bd = 0, highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
TH_frame.pack(side = tk.LEFT)
th0=tk.DoubleVar(value=0.1)
th1=tk.DoubleVar(value=0.1)
th2=tk.DoubleVar(value=0.1)
thh=tk.DoubleVar(value=0.1)
ths=tk.DoubleVar(value=0.1)
thv=tk.DoubleVar(value=0.1)
tk.Label(TH_frame,text='TH=[R,G,B,S,V]').pack()
tk.Scale(TH_frame, from_=0, to=1,variable=th0,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(TH_frame, from_=0, to=1,variable=th1,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(TH_frame, from_=0, to=1,variable=th2,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(TH_frame, from_=0, to=1,variable=ths,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(TH_frame, from_=0, to=1,variable=thv,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)

weight_frame= tk.Frame(master=root, width = 312, height = 50, bd = 0, highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
weight_frame.pack(side = tk.LEFT)
rw=tk.DoubleVar(value=0.5)
gw=tk.DoubleVar(value=0.4)
bw=tk.DoubleVar(value=0.4)
sw=tk.DoubleVar(value=0.8)
vw=tk.DoubleVar(value=0.6)
tk.Label(weight_frame,text='weight=[R,G,B,S,V]').pack()
tk.Scale(weight_frame, from_=0, to=1,variable=rw,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(weight_frame, from_=0, to=1,variable=gw,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(weight_frame, from_=0, to=1,variable=bw,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(weight_frame, from_=0, to=1,variable=sw,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)
tk.Scale(weight_frame, from_=0, to=1,variable=vw,resolution=0.05, orient=tk.VERTICAL).pack(side=tk.LEFT)

detection_frame= tk.Frame(master=root, width = 312, height = 50, bd = 0, highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
detection_frame.pack(side = tk.LEFT)
thDetection=tk.DoubleVar(value=0.1)
tk.Label(detection_frame,text='Detection TH').pack()
tk.Scale(detection_frame, from_=0, to=0.4,variable=thDetection,resolution=0.005, orient=tk.VERTICAL).pack(side=tk.LEFT)


cal_frame=tk.Frame(master=root, width = 312, height = 50, bd = 0, highlightbackground = "black", highlightcolor = "black", highlightthickness = 1)
cal_frame.pack(side = tk.LEFT)
tk.Label(master=cal_frame,text='Calibration').grid(row=0)
tk.Label(master=cal_frame,text='Where is your finger?').grid(row=0,column=1)
tk.Label(master=cal_frame,text=' (5 points needed)').grid(row=0,column=2)
tk.Button(master=cal_frame, text='Calibrate',command=xyData).grid(row=1)
tk.Button(master=cal_frame, text='Reset',command=Reset).grid(row=2)
xentry=tk.Entry(master=cal_frame)
xentry.grid(row=1, column=1)
yentry=tk.Entry(master=cal_frame)
yentry.grid(row=2, column=1)
tk.Label(master=cal_frame,text='X[CM]').grid(row=1,column=2)
tk.Label(master=cal_frame,text='Y[CM]').grid(row=2,column=2)

root.bind('<BackSpace>',erase)
root.bind('<Return>',ColorCalibration)


#%%
while True:
    #Obtaining image and refreshing interface
    root.update_idletasks()
    root.update()  
    ret, frame = capture.read()
    img=own.nfloat(np.flip(frame,2))
    
    
    #updating parameters
    
    th=[th0.get(),th1.get(),th2.get(),1,ths.get(),thv.get()]
    hweight=[rw.get(),gw.get(),bw.get(),0,sw.get(),vw.get()]
    
    #%%SEGMENTATION
    
    Board, Border,twoMax,detection,VerticalFingerRegion,marge,fingerColor,FingerBinary = own.Segmentation(img,A,B,hBorder,width,hportions,hweight,twoMax,th,thDetection.get(),0.4)
    
    
    VFRCopy=np.copy(VerticalFingerRegion)
    
    if detection:
        VerticalFingerRegion *=FingerBinary[:,:,None] #This line is just to visualize the threshold method in live
#    
    
    #%%Calculate coordinates
    xold,yold=xPX,yPX
    xPX ,yPX = own.PositionFinger2(FingerBinary)
    xPX,yPX=xPX+twoMax[0]+A[0] ,yPX+hBorder
    
    ax,bx,cx,dx = xparam
    ay,by,cy,dy = yparam
    
    xCM=own.xCalParFunction((xPX,yPX),ax,bx,cx,dx)
    yCM=own.yCalParFunction((xPX,yPX),ay,by,cy,dy)
    
    expo=0.1
    xCMexp, yCMexp =xCMexp*(1-expo)+xCM*expo,yCMexp*(1-expo)+yCM*expo #we apply an exponential average in order to avoid big oscillations in coordinates
    
    
    xB,yB=getXYB(xCM,-yCM) #One should change the sign of yB,xB depending on the orientation of the camera 
#    xB,yB=0.5,0.5
    xB,yB=np.uint32(xB*blackboard.shape[1]),np.uint32(yB*blackboard.shape[0]) 
    
    
   
    textPX.set('x, y [PX]='+str(xPX)+'   '+str(yPX))
    textCM.set('x, y [CM]='+str(round(xCMexp,2)) +'   '+ str(round(yCMexp,2) ) )
    
    #%%Display
        
    img = cv2.circle(img, A, radius=10, color=(0, 0, 1), thickness=-1)
    img = cv2.circle(img, B, radius=10, color=(0, 0, 1), thickness=-1)
    
    if detection:
        cv2.line(img,(0,yPX),(img.shape[1],yPX),(1,1,1),2)
        cv2.line(img,(xPX,0),(xPX,img.shape[0]),(1,1,1),2)
        cv2.line(Board,(twoMax[0],0),(twoMax[0],width),(1,0,0),4)
        cv2.line(Board,(twoMax[1],0),(twoMax[1],width),(1,0,0),4)
    
    
    
    #%%Draw on the blackboard
    if detection:
        blackboard=cv2.circle(blackboard, (xB,yB), radius=5
                             , color=255*fingerColor , thickness=-1)
    
    #%%actualize interface
    
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.uint8(255*img)))
    canv.itemconfig(canvimage,image = photo)
    black=PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(blackboard))
    canv.itemconfig(canvimage2,image = black)
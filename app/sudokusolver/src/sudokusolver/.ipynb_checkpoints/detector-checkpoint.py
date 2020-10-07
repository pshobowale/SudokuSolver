'''Detector

    Detector for searching a sudokugrid in an image using local variance 
'''

import cv2 as cv
import numpy as np
from numba import jit

class detector:
    def __init__(self,image_src=None):
        '''Initizialize detector

        Args:
            image_src: can be a string with a path to an image or an np.array
        '''

        #Load Image
        self.Org=None

        #print(image_src)
        if image_src is None:
            return

        if isinstance(image_src,str):
            self.Org=cv.imread(image_src)
                       
        else:
            self.Org=image_src
        print(image_src)
        
        #Rescale and convert color
        self.Org_Color=cv.resize(self.Org,(400,round(self.Org.shape[0]/self.Org.shape[1]*400))) #Scaling to 400px width
        self.Org=cv.cvtColor(self.Org_Color,cv.COLOR_RGB2GRAY)                                  #Convert to black and white
        self.Org=(self.Org-np.min(self.Org))/(np.max(self.Org)-np.min(self.Org))*255            #Normalization

        
        
    def variance_filter(self,I=None, n=5):
        if I is None:
            I=self.Org
        I=np.array(I,dtype=np.float)
        
        
        
        mean=np.array(cv.GaussianBlur(I,(n,n),1,1),dtype=np.float)
        
        return self.variance_filter_jit(I,n,mean)
        
    @jit#(nopython=True)
    def variance_filter_jit(self,I,n,mean):  
        
        h,w=I.shape
        result=np.zeros(I.shape,dtype=np.float)

        print(h,w)
        for y in range(h):
            for x in range(w):
                
                sum=0
                for i in range(-n//2,n//2+1):
                    for j in range(-n//2,n//2+1):
                        
                        xi,yi=x+i,y+j   
                        sum+=(I[yi,xi]-result[y,x])*(I[yi,xi]-result[y,x])\
                              if xi>0 and yi>0 and yi<h and xi<w else 0
                
                result[y,x]=sum/(n**2-1)           
        return result
    
    def findGrid(self,I=None,n=5):
        '''findGrid(I=None,n=5)
        Args: 

            I: grayscale Image 
            n: size of the kernel for the local variance filter

        '''

        #Filtering out Text and Noise
        if I is None:
            if self.Org is not None:
                I=np.copy(self.Org)
            
            else:
                return None
        
        contours=-(I-np.max(I))*self.variance_filter(I)                               #Weight Pixel by their local variance
        contours=contours/np.max(contours)*255                                        #Normalization
        contours=np.array(contours,dtype=np.uint8)                                    #Conversion from float to int
        contours = cv.threshold(contours,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]    #Threshold

        
        #Searching for the Grid
        contours, hierarchy = cv.findContours(contours,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)       
        areas = [cv.contourArea(c) for c in contours]                               # Find the index of the largest contour
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        epsilon = 0.1*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)                                  #approx a bounding box
        
        return approx

    def getROI(self,I=None,contour=None):
        
        if I is None:
            if self.Org is not None:
                I=np.copy(self.Org)
                
        else :
            return None
        
        if contour is None:
            contour=self.findGrid(I)
            
        #print(contour)
        
        ROI=np.copy(I)
        pts_dst = np.float32([[0,0],[0,180],[180, 0],[180, 180]])
        pts_src=contour.flatten()

        if len(pts_src)!=8:
            return None

        pts_src=np.float32([[pts_src[0],pts_src[1]],[pts_src[2],pts_src[3]],[pts_src[6], pts_src[7]],[pts_src[4], pts_src[5]]])
        trans = cv.getPerspectiveTransform(pts_src, pts_dst)
        ROI = cv.warpPerspective(ROI, trans, (180,180))

        
        return (ROI,pts_src)

    def getImage(self):
        return self.Org_Color

    def newImage(self,image_src):
        if image_src is None:
            return

        if isinstance(image_src,str):
            self.Org=cv.imread(image_src)
                       
        else:
            self.Org=image_src
        print(image_src)
        
        #Rescale and convert color
        self.Org_Color=cv.resize(self.Org,(400,round(self.Org.shape[0]/self.Org.shape[1]*400))) #Scaling to 400px width
        self.Org=cv.cvtColor(self.Org_Color,cv.COLOR_RGB2GRAY)                                  #Convert to black and white
        self.Org=(self.Org-np.min(self.Org))/(np.max(self.Org)-np.min(self.Org))*255   

    def getDigits(self,I=None):
        if I is None:
            if self.Org is not None:
                I=self.Org
            else:
                return None
        possible_digits=[]
if __name__=="__main__":
    import os as os
    import matplotlib
    import matplotlib.pyplot as plt

    img_path=[]
    img_dir="/home/seun/Documents/Programme/Python/SudokuSolver/test/Samples/"
    img_files=os.listdir(img_dir)
    #print(img_files)
    
    for f in img_files:
        if ".jpg" in f:
            img_path.append(f)

    print(img_dir+img_path[1])
    detector=detector()
    for f in img_path[:]:
        detector.newImage(img_dir+f)
        I=detector.getROI()
        
        if I is not None:
            I=I[0]
            plt.subplot(221)
            plt.imshow(I,cmap="gray")
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(detector.getImage())
            plt.colorbar()
            plt.show()
        




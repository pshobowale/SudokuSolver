'''Detector

    Detector for searching a sudokugrid in an image using local variance 
'''

import cv2 as cv
import numpy as np
from numba import jit
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; #To supress TF messages
from tensorflow.keras.models import Sequential, save_model, load_model



class detector:
    def __init__(self,image_src=None,load_keras_model=True):
        '''Initizialize detector

        Args:
            image_src: can be a string with a path to an image or an np.array
        '''
        #Load Image
        self.Org=None
        self.ROI=None
        self.possible_digits=None
        self.cordROI=None
        

        if(load_keras_model):
            print("Loading Kerasmodell from:",os.path.abspath("./src/sudokusolver/resources/DigitClassifier.keras"))
            self.model=load_model(os.path.abspath("./src/sudokusolver/resources/DigitClassifier.keras"))

        if image_src is None:
            return

        if isinstance(image_src,str):
            self.Org_Color=cv.imread(image_src)
                       
        else:
            self.Org_Color=image_src
        
        #Rescale and convert color
        self.Org=cv.resize(self.Org_Color[0],(600,round(self.Org_Color.shape[0]/self.Org_Color.shape[1]*600))) #Scaling to 400px width
        self.Org=cv.cvtColor(self.Org,cv.COLOR_RGB2GRAY)                                  #Convert to black and white
        self.Org=(self.Org-np.min(self.Org))/(np.max(self.Org)-np.min(self.Org))*255            #Normalization
        
    def loadModel(path):
        if path is None:
            print("Loading Kerasmodell from: ",os.path.abspath(path))
            path="./resources/DigitClassifier.keras"
        self.model=load_model(path)       
  
    def variance_filter(self,I=None, n=5):
        if I is None:
            I=self.Org
        
        I=np.array(I,dtype=np.float)
        
        
        
        mean=np.array(cv.GaussianBlur(I,(n,n),1,1),dtype=np.float)
        
        return self.variance_filter_jit(I,n,mean)

        
    @staticmethod
    @jit(nopython=True)
    def variance_filter_jit(I,n,mean):  
        
        h,w=I.shape
        result=np.zeros(I.shape,dtype=np.int32)

        for y in range(h):
            for x in range(w):
                
                sum=0
                for i in range(-n//2,n//2+1):
                    for j in range(-n//2,n//2+1):
                        
                        xi,yi=x+i,y+j   
                        sum+=(I[yi,xi]-mean[y,x])*(I[yi,xi]-mean[y,x])\
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
        
        contours=-(I-np.max(I))*self.variance_filter(I)#                               #Weight Pixel by their local variance
        contours=contours/np.max(contours)*255                                        #Normalization
        contours=np.sqrt(contours)
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

    def getROI(self,I=None,contour=None,ignore=False):
        
        
        usingOrg=False

        if I is None:
            if self.ROI is not None:
                if self.cordROI is None:
                    self.cordROI=self.findGrid()
                    return(self.ROI,self.cordROI)
                else:
                    return (self.ROI,self.cordROI)
            elif self.Org is not None:
                I=np.copy(self.Org)
                
            else :
                return None

            usingOrg=True
            
        
        if contour is None:
            contour=self.findGrid(I)
       

        #Rescaling found Points to Orignial Size    
        if usingOrg:
            ROI=np.copy(self.Org_Color)
            ROI=cv.cvtColor(ROI,cv.COLOR_RGB2GRAY)
            h,w=ROI.shape
            h_old,w_old=self.Org.shape
            pts_src=contour.flatten()

            for i in range(len(pts_src)):
                if i%2:
                    pts_src[i]=pts_src[i]*h/h_old
                else:
                    pts_src[i]=pts_src[i]*w/w_old

        else:
            ROI=np.copy(I)
            pts_src=contour.flatten()
        
        pts_dst = np.float32([[0,0],[0,180],[180, 0],[180, 180]])

        


        if len(pts_src)!=8:
            return None

        pts_src=np.float32([[pts_src[0],pts_src[1]],[pts_src[2],pts_src[3]],[pts_src[6], pts_src[7]],[pts_src[4], pts_src[5]]])
        trans = cv.getPerspectiveTransform(pts_src, pts_dst)

        ROI = cv.warpPerspective(ROI, trans, (180,180))


        return (ROI,pts_src)

    def getImage(self):
        if self.Org_Color is not None:
            return self.Org_Color

    def newImage(self,image_src):
        if image_src is None:
            return

        if isinstance(image_src,str):
            self.Org_Color=cv.imread(image_src)
                       
        else:
            self.Org_Color=image_src
        
        #Rescale and convert color
        self.Org=cv.resize(self.Org_Color,(400,round(self.Org_Color.shape[0]/self.Org_Color.shape[1]*400))) #Scaling to 400px width
        self.Org=cv.cvtColor(self.Org,cv.COLOR_RGB2GRAY)                                  #Convert to black and white
        self.Org=(self.Org-np.min(self.Org))/(np.max(self.Org)-np.min(self.Org))*255   
        self.ROI=None
        self.cordROI=None
        self.possible_digits=None
        

    def getDigits(self,ROI=None):
        save2self=True
        if ROI is None:
            if self.possible_digits is not None:
                return possible_digits
            if self.ROI is not None:
                ROI=self.ROI
            elif self.Org is not None:
                ROI=self.getROI()[0]
                self.ROI=ROI
            else:
                return None
        else:
            save2self=True
        
        ROI=ROI-np.min(ROI)
        ROI=ROI/np.max(ROI)*255
        
        possible_digits=[]
        digits=ROI\
                -cv.dilate(ROI,cv.getStructuringElement(cv.MORPH_RECT,(11,1)))\
                -cv.dilate(ROI,cv.getStructuringElement(cv.MORPH_RECT,(1,11)))
        digits=-(digits-np.max(digits))
        digits=digits/np.max(digits)*255

        kernel=np.zeros((20,20))
        kernel[2:17,4:15]=cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,15))
        kernel*=np.dot(cv.getGaussianKernel(20,4),cv.getGaussianKernel(20,3).T)
        kernel/=np.sum(kernel)

        kernel2=np.dot(cv.getGaussianKernel(20,5),cv.getGaussianKernel(20,4).T)
        kernel2[2:17,4:15]=0
        kernel2/=np.sum(kernel2)
        kernel2-=np.ones((20,20))
        kernel2/=np.sum(abs(kernel2))
        kernel=kernel+kernel2
        img=cv.filter2D(digits,ddepth=cv.CV_32F,kernel=kernel,anchor=(0,0))
        max=cv.dilate(img,np.ones((20,20)))
        max=img==max
        thr=img>15
        max=thr*max
        
        h,w=ROI.shape
        for i in range(h):
            for j in range(w):
                if max[i,j]:
                    possible_digits.append((ROI[i:i+20,j:j+20],(i,j)))
                    

        if save2self:
            self.possible_digits=possible_digits

        return possible_digits

    def saveDigits(self,dst,possible_digits=None,name=None):
        
        if possible_digits is None:
            if self.possible_digits is not None:
                possible_digits=self.possible_digits
            elif self.Org is not None:
                possible_digits=self.getDigits()
                self.possible_digits=possible_digits
            else:
                return

        if name is None:
            name=datetime.utcnow().strftime("%Y%m%d%h%m%s")

        for I in possible_digits:
            if I[0].shape[0]==20 and I[0].shape[1]==20:
                plt.imsave(dst+"/"+name+str(I[1])+".bmp",I[0],cmap="gray")
            
    def saveImage(I=None,path=None):
        if I is None or path is None:
            return 
        else:
            plt.imsave(path,I,cmap="gray")

    def classifyDigits(self,possible_digits=None):
        if possible_digits is None:
            if self.possible_digits is not None:
                possible_digits=self.possible_digits
            else:
                possible_digits=self.getDigits()
                
                if possible_digits is None:
                    return None
                else:
                    if len(possible_digits)>0:
                        self.possible_digits=possible_digits
                    else:
                        return None
        

        digit=[]
        pos=[]
        for d in possible_digits:
            if d[0].shape[0]==20 and d[0].shape[1]==20:
                digit.append(d[0]/255)
                pos.append(d[1])

        digitm=np.array(digit,dtype=np.float)
        digitm=np.expand_dims(digitm, -1)

        y=self.model.predict(digitm)
        prediction=[] 
        for p in y:
            prediction.append(np.argmax(p))

        sudoku=np.zeros((9,9),dtype=np.uint8)
        for i in range(len(pos)):
            p=pos[i]
            sudoku[(p[0]+10)//20,(p[1]+10)//20]=prediction[i]
        
        return (sudoku,zip(digit*255,pos,prediction))


if __name__=="__main__":
    import os as os

    img_path=[]

    img_dir="/home/seun/Documents/Programme/Python/SudokuSolver/test/Samples/"
    #img_dir="/home/seun/Documents/Programme/Python/SudokuSolver/test/Samples/wichtounet/v2_test/"
    img_files=os.listdir(img_dir)
    print(img_files)
    
    for f in img_files:
        if ".jpg" in f:
            img_path.append(f)

    print(img_dir+img_path[1])
    detector=detector()
    
    detected=0
    for f in img_path:
        detector.newImage(img_dir+f)
        I=detector.getROI()
        
        
        if I is not None:
            Org=detector.getImage()
            pd=detector.getDigits(I[0])
            digits=detector.classifyDigits(pd)
            sudoku=digits[0]
            print(sudoku)
            plt.imshow(Org)
            plt.show()
                
            #detector.saveDigits("/home/seun/Documents/Programme/Python/SudokuSolver/test/Dataset/Train",name=f)

    print("detected ", detected,"of ",len(img_path))
        
        




#! /usr/bin/python
import os
import matplotlib.pyplot as plt

def main():
    
    dataset_dir=os.path.abspath("./Dataset")
        
    img_files=os.listdir(dataset_dir)
    
    img_path=[]
    img_name=[]
    for f in img_files:
        if ".jpg" in f:
            img_path.append(dataset_dir+"/"+f)
            img_name.append(f)

    print("Bilderanzahl: ",len(img_path))
    for f in img_path:
        I=plt.imread(f)
        if I.shape[0]==20 and I.shape[1]==20:
            plt.imshow(I)
            plt.show()

if __name__=="__main__":main()
#! /usr/bin/python
import sys
import os
import getopt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../app/sudokusolver/src/sudokusolver")
import detector as d

def main(argv):    

    if len(argv)== 3:
        img_dir=os.path.abspath(argv[1])
        dataset_dir=os.path.abspath(argv[2])
    else:
        img_dir=os.path.abspath("./Samples/")
        dataset_dir=os.path.abspath("./Dataset")
        
    img_files=os.listdir(img_dir)
    print(img_files)
    
    img_path=[]
    img_name=[]
    for f in img_files:
        if ".jpg" in f:
            img_path.append(img_dir+"/"+f)
            img_name.append(f)


    print("example filepath: ",img_path[1])

    

    detector=d.detector(load_keras_model=False)
    
    detected=0
    for i,f in enumerate(img_path):
        detector.newImage(f)
        I=detector.getROI()
        
        
        if I is not None:  
            detector.saveDigits(dataset_dir,name=img_name[i])
            detected+=1

    print("Image directory:",img_dir)
    print("Dataset directory:",dataset_dir)
    print("detected ", detected,"of ",len(img_path))

if __name__ == "__main__":
    main(sys.argv)
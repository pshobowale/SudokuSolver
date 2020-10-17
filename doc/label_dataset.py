#! /usr/bin/python
import os
import matplotlib.pyplot as plt
import pickle

label = None
dataset=[]

def press(event):
    global label
    label = event.key

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
    for f in img_path[:10]:
        I=plt.imread(f)
        if I.shape[0]==20 and I.shape[1]==20:
            plt.imshow(I)
            plt.draw()
            plt.gcf().canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress(0)
            print(label)
            plt.close()
            dataset.append((label,I))

    with open("Dataset.dat", 'wb') as output:  # Overwrites any existing file.
        pickle.dump(dataset, output)
if __name__=="__main__":main()
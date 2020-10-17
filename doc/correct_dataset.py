#! /usr/bin/python
import pickle
import matplotlib.pyplot as plt
import numpy as np

label = None
dataset_final=[]

def press(event):
    global label
    label = event.key

def main():
    file=open("./Dataset1.dat","rb")
    dataset1=pickle.load(file)
    file.close()

    file=open("./Dataset2.dat","rb")
    dataset2=pickle.load(file)
    file.close()


    diffs=0

    for i in range(len(dataset1)):
        if dataset1[i][0]!=dataset2[i][0] or np.sum(dataset1[i][1]-dataset2[i][1])!=0:
            diffs+=1
            print(dataset1[i][0],"!=",dataset2[i][0])
            plt.subplot(121)
            plt.imshow(dataset1[i][1], cmap="gray")
            plt.subplot(122)
            plt.imshow(dataset2[i][1])
            plt.draw()
            plt.gcf().canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress(0)
            print("corrected label: ",label)
            plt.close()
            dataset_final.append((label,dataset1[i][1]))
        else:
            dataset_final.append(dataset1[i])

    print("Diffs:", diffs)
    with open("Dataset_Final.dat", 'wb') as output:  # Overwrites any existing file.
        pickle.dump(dataset_final, output)
        output.close()

if __name__ == "__main__": main()
    

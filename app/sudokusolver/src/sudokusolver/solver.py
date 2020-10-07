import numpy as np

class solver:
    def __init__(self):
        self.grid= np.ones((9,9,9))
        self.isSolved=False
        self.Errors=False
    
    def setField(self,x,y,value):
        if value>0:
            self.grid[x,y,:]=0
            self.grid[x,y,value-1]=1
        else:
            self.grid[x,y,:]=1

        #print("xyv",x,y,value)
        #print(self.grid[:3,5:])

    

    def getGrid(self,S=None):
        R=np.zeros((9,9))

        if S is None:
            S=self.grid

        for x in range(9):
            for y in range(9):
                sum=0
                v=0
                for i in range(9):
                    sum+=S[x,y,i]
                    if(S[x,y,i]):
                        v=int(i)
                if sum==1:
                    R[x,y]=v+1
                if sum==0:
                    self.Errors=True

        for x in range(3):
            for y in range(3):
                nbh=S[3*x:3*x+3,3*y:3*y+3]
                for i in range(9):
                    sum=np.sum(nbh[:,:,i])
                    if sum==1:
                        #print(x,y,":\n",nbh[:,:,i])
                        for j in range(3):
                            for k in range(3):
                                if nbh[j,k,i]:
                                    R[3*x+j,3*y+k]=i+1
                    if sum==0:
                        self.Erros=True

        return R

    
    def getPossibilitys(self):
        R=np.zeros((9,9))

        for x in range(9):
            for y in range(9):
                sum=0
                for i in range(9):
                    sum+=self.grid[x,y,i]
                R[x,y]=sum
        return R

    def printGrid(self,S=None):
        if S is None:
            S=self.grid
            

        R=self.getGrid(S)
        for y in range(9):
            for x in range(9):
                if(R[x,y]):
                    print(int(R[x,y]),"\t",end='')
                else:
                    print(" \t",end='')
                if(x%3==2):
                    print("|",end='')
            
            print()
            if(y%3==2):
                print("-\t-\t-\t-\t-\t-\t-\t-\t")

        

    def Change(self,A,B):
        return np.sum(A-B)

    def getClue(self,S=None):
        if S is None:
            S=np.copy(S)
        else:
            S=np.copy(S)
        R=self.getGrid(S)

        for x in range(9):
            for y in range(9):
                v=R[x,y]
                if v!=0:
                    #print(v,R[x,:])
                    S[x,:,int(v-1)]=0
                    #print(v,R[:,y])
                    S[:,y,int(v-1)]=0


                    xk,yk=x//3,y//3
                    #print(x,y,"|",xk,yk)
                    #print(R[xk*3:xk*3+3,yk*3:yk*3+3])
                    S[xk*3:xk*3+3,yk*3:yk*3+3,int(v-1)]=0
                    S[x,y,int(v-1)]=1
        return S

    def getSolution(self):
        S=np.copy(self.grid)
        New=self.getClue(S)

        
        for i in range(100):
            S=New
            New=self.getClue(S)
            

            if (self.Change(New,S)==0.0):
                print("Iterations:",i)
                break
            
        if np.sum==9*np.math.factorial(9):
            self.isSolved=True
        self.grid=New

        print("target: \n",self.grid[:3,:3,0])
        return (self.isSolved,New)


if __name__=="__main__":
    solver=solver()
    solver.setField(0,0,1)
    solver.setField(1,0,8)
    solver.setField(0,1,5)
    solver.setField(1,1,2)
    solver.setField(2,1,3)

    solver.setField(3,2,1)
    solver.setField(4,1,4)
    solver.setField(5,0,9)

    solver.setField(7,0,4)
    solver.setField(8,0,5)
    solver.setField(7,1,8)
    solver.setField(8,1,1)

    solver.setField(0,4,2)
    solver.setField(2,5,7)

    solver.setField(4,4,7)
    solver.setField(5,3,5)
    solver.setField(3,5,8)

    solver.setField(8,3,9)
    solver.setField(8,4,6)
    solver.setField(6,5,1)

    solver.setField(0,6,3)
    solver.setField(0,8,8)
    solver.setField(2,8,2)
    solver.setField(2,7,5)
    solver.setField(2,6,1)

    solver.setField(3,6,4)
    solver.setField(3,8,5)
    solver.setField(5,8,7)

    solver.setField(8,7,8)
    solver.setField(6,6,5)
    solver.setField(7,6,6)

    #Hinzufügen für komplett lösbares Sudoku
   # solver.setField(1,4,5)
   # solver.setField(1,5,3)
   # solver.setField(4,7,1)
   # solver.setField(5,5,4)
   # solver.setField(3,3,2)
   # solver.setField(2,3,8)
   # solver.setField(0,7,7)


    #'Test'
    solver.printGrid()
    solver.getSolution()
    # print("\n\n Solution")
    # solver.printGrid()
    # solver.getSolution()
    # print("\n\n Solution")
    # solver.printGrid()
    # solver.getSolution()

    print("\n\n Solution")
    #solver.print(solver.getClue)
    solver.printGrid()

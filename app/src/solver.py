import numpy as np

class solver:
    def __init__(self):
        self.grid= np.ones((9,9,9),dtype=np.uint8)
        self.Org=None
        self.isSolved=False
        self.Error=False
    
    def setField(self,x,y,value):
        if value>0:
            self.grid[x,y,:]=0
            self.grid[x,y,value-1]=1
        else:
            self.grid[x,y,:]=1

        self.Org=np.copy(self.grid)
        #print("xyv",x,y,value)
        #print(self.grid[:3,5:])

    def clearGrid(self):
        for i in range(9):
            for j in range(9):
                self.setField(i,j,0) 

    def getGrid(self,S=None,ignoreNewPossibilites=False):
        R=np.zeros((9,9),dtype=np.uint8)

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
                    self.Error=True
                    
        if ignoreNewPossibilites:
            return R

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
                        self.Error=True

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

    def printGrid(self,S=None,R=None):
        if S is None:
            S=self.grid
            if self.Error:
                print("Sudoko not solvable")
            
        if R is None:
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

        

    def printOrg(self):
        if self.Org is not None:
            self.printGrid(R=self.getGrid(self.Org,ignoreNewPossibilites=True))

    def printPossi(self):
        self.printGrid(R=self.getPossibilitys())

    def diff2Org(self,g=None):
        if self.Org is None:
            return None
        if g is None and self.grid is not None:
            g=self.getGrid(self.grid)
        else:
            return None
            
        diff=g-self.getGrid(self.Org,ignoreNewPossibilites=True)
        self.printGrid(R=diff)
        
        return diff

        

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

    def getObvious(self):
        S=np.copy(self.grid)
        New=self.getClue(S)

        
        for i in range(100):
            S=New
            New=self.getClue(S)
            
            #print("\n\n")
            #self.printPossi()

            if (self.Change(New,S)==0.0):
                print("Iterations:",i)
                break
            
        if np.sum==9*np.math.factorial(9):
            self.isSolved=True
        self.grid=New

        print("target: \n",self.grid[:3,:3,0])
        return (self.isSolved,New)


    def getSolution(self,grid=None):
        if grid is None:
            if self.Org is None:
                return
            else:
                grid=self.getGrid(self.Org,ignoreNewPossibilites=True)

        def findNextCellToFill(grid, i, j):
                for x in range(i,9):
                        for y in range(j,9):
                                if grid[x][y] == 0:
                                        return x,y
                for x in range(0,9):
                        for y in range(0,9):
                                if grid[x][y] == 0:
                                        return x,y
                return -1,-1

        def isValid(grid, i, j, e):
                rowOk = all([e != grid[i][x] for x in range(9)])
                if rowOk:
                        columnOk = all([e != grid[x][j] for x in range(9)])
                        if columnOk:
                                # finding the top left x,y co-ordinates of the section containing the i,j cell
                                secTopX, secTopY = 3 *(i//3), 3 *(j//3) #floored quotient should be used here. 
                                for x in range(secTopX, secTopX+3):
                                        for y in range(secTopY, secTopY+3):
                                                if grid[x][y] == e:
                                                        return False
                                return True
                return False

        def solveSudoku(grid, i=0, j=0):
                i,j = findNextCellToFill(grid, i, j)
                if i == -1:
                        return True
                for e in range(1,10):
                        if isValid(grid,i,j,e):
                                grid[i][j] = e
                                if solveSudoku(grid, i, j):
                                        return True
                                # Undo the current cell for backtracking
                                grid[i][j] = 0
        solveSudoku(grid)
        return grid

        

if __name__=="__main__":
    solver=solver()

    def case1():
        solver.setField(0,0,1)
        solver.setField(1,0,8)
        solver.setField(0,1,5)
        #solver.setField(1,1,2)
        solver.setField(2,1,3)
        solver.setField(3,2,1)
        solver.setField(4,1,4)
        solver.setField(5,0,9)
        solver.setField(7,0,4)
        solver.setField(8,0,5)
        solver.setField(7,1,8)
        solver.setField(8,1,1)
        solver.setField(0,4,2)
        #solver.setField(2,5,7)
        solver.setField(4,4,7)
        #solver.setField(5,3,5)
        solver.setField(3,5,8)
        solver.setField(8,3,9)
        solver.setField(8,4,6)
        #solver.setField(6,5,1)
        solver.setField(0,6,3)
        solver.setField(0,8,8)
        solver.setField(2,8,2)
        solver.setField(2,7,5)
        solver.setField(2,6,1)
        solver.setField(3,6,4)
        solver.setField(3,8,5)
        solver.setField(5,8,7)
        solver.setField(8,7,8)
        #solver.setField(6,6,5)
        solver.setField(7,6,6)
 
        #Hinzufügen für komplett lösbares Sudoku
        solver.setField(1,4,5)
        solver.setField(1,5,3)
        solver.setField(4,7,1)
        solver.setField(5,5,4)
        solver.setField(3,3,2)
        solver.setField(2,3,8)
        solver.setField(0,7,7)

    def case2():
        #test find with row const
        solver.setField(0,1,1)
        solver.setField(0,2,2)
        solver.setField(0,3,3)
        solver.setField(0,4,4)
        solver.setField(0,5,5)
        solver.setField(0,6,6)
        solver.setField(0,7,7)
        solver.setField(0,8,8)
    def case3():
        #test find with row and column
        solver.setField(0,0,1)
        solver.setField(8,1,9)

        solver.setField(0,2,3)
        solver.setField(0,3,4)
        solver.setField(0,4,5)
        solver.setField(0,5,6)
        solver.setField(0,6,7)
        solver.setField(0,7,8)

    def case4():
        #Test find with row col const
        solver.setField(0,0,9)
        solver.setField(2,3,9)
        solver.setField(3,6,9)
        solver.setField(7,7,9)
    

    def case5():
        #Test find with row and nbh const
        solver.setField(0,0,9)
        solver.setField(2,3,9)
        solver.setField(1,6,2)
        solver.setField(1,7,5)

    def case6():
        #Test find with nbh const
        solver.setField(0,0,1)
        solver.setField(0,1,2)
        solver.setField(0,2,3)
        solver.setField(1,0,4)
        solver.setField(1,1,5)
        solver.setField(1,2,6)
        solver.setField(2,0,7)
        solver.setField(2,1,8)

    
    def case7():
        solver.setField(0,0,1)
        solver.setField(0,1,2)
        solver.setField(0,2,3)
        solver.setField(1,0,4)
        solver.setField(1,1,5)
        solver.setField(1,2,6)
        solver.setField(3,0,7)
        solver.setField(3,1,8)
        solver.setField(4,0,9)

    #Evaluation

    
    case1()

    G=solver.getSolution()

    print("\n\n Org")
    solver.printOrg()
    print("\n\n Diff")
    solver.diff2Org()
    print("\n\n Possiblitys")
    solver.printPossi()
    print("\n\n Solution")
    solver.printGrid(R=G)
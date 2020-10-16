"""
solves sudokus
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from sudokusolver.solver import solver
from sudokusolver.detector import detector



class SudokuSolver(toga.App):
    def startup(self):
        main_box=toga.Box(id="mainbox",style=Pack(direction=COLUMN,padding=5))
        self.solver=solver()
        self.detector=detector()
        self.ImagePath=""
        self.Preview=toga.ImageView(image=toga.Image("resources/blank_sudoku.jpg"))
        self.grid=[]
        self.hideROI=True
        self.imgLoaded=False

        for i in range(9):
            self.grid.append([])
            row=toga.Box(style=Pack(direction=ROW, padding=0))
            for j in range(9):
                self.grid[i].append(toga.Button(label=" ",id='{}{}'.format(i,j),on_press=self.pressedButton,style=Pack(flex=1)))
                row.add(self.grid[i][j])
                if (j+1)%3==0 and j<8:
                    row.add(toga.Label(style=Pack(padding=0),text="\t"))
            main_box.add(row)
            if (i+1)%3==0 and i<8:
                main_box.add(toga.Label(text="\t"))
        
        #adding solvebutton
        main_box.add(toga.Label(text="\t"))
        main_box.add(toga.Button(label="Solve",on_press=self.solveButton))

        image_box=toga.Box(id="imagebox",style=Pack(direction=ROW,padding=2))
        image_box.add(toga.Button(label="Open",on_press=self.openButton))
        image_box.add(toga.Button(label="ROI",on_press=self.switchButton,style=Pack(direction=)))
        image_box.add(self.Preview)
        
        
        
        option=toga.OptionContainer()
        option.add("Image",image_box)
        option.add("Sudoku",main_box)
        
        #init window
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = option
        
        self.main_window.show()
        
        
        
        
    #callback for buttonpress
    #increases number on buttonlabel and update solver
    def pressedButton(self,widget):
        #print(widget.id)
        if widget.label == " ": 
            widget.label="1" 
            self.solver.setField(int(widget.id[1]),int(widget.id[0]),1) #set Field[x,y] to 1
            

        elif int(widget.label)==9:
            widget.label=" "
            self.solver.setField(int(widget.id[1]),int(widget.id[0]),0) #clear Field[x,y]
            
        else:
            widget.label=str(int(widget.label)+1)
            self.solver.setField(int(widget.id[1]),int(widget.id[0]),int(widget.label)) #set Field[x,y]
    
    def solveButton(self,widget):
        self.solver.printGrid()
        #self.solver.getSolution()
        #self.solver.getSolution()
        s,G=self.solver.getSolution()
        self.solver.printGrid(G)
        #print(G[-3:,-3:,:])

    def openButton(self,widget):
        path=self.main_window.open_file_dialog("Select an Image","~",["jpg","jpeg","bmp"])
        self.detector.newImage(path)
        sudoku=self.detector.classifyDigits()[0]
        if sudoku is not None:
            for i in range(9):
                for j in range(9):
                    if sudoku[i,j]:
                        self.grid[i][j].label=str(sudoku[i][j])
                        self.solver.setField(j,i,sudoku[i,j])
                    else:
                        self.grid[i][j].label=" "
                        self.solver.setField(j,i,0)
            self.ImagePath=path            
            self.Preview.image=toga.Image(path)
            self.Preview.refresh()
            self.imgLoaded=True

        
        #with matplotlib.pyplot as plt:
            #I=self.detector.getROI()[0]
            #plt.imshow(I)
            #plt.show()

    def switchButton(self,widget):
        if self.imgLoaded==False:
            return
        elif self.hideROI==False:
            self.Preview.image=toga.Image(path)
            widget.Label="Original"
            self.hideROI=True
            self.Preview.refresh()
        else:
            self.Preview.image=toga.Image(path)
            widget.Label="ROI"
            self.hideROI=False
            self.Preview.refresh()


    
    #Hide menuebar
    def _create_impl(self):
        factory_app = self.factory.App
        factory_app.create_menus = lambda _: None
        return factory_app(interface=self)



def main():
    return SudokuSolver()


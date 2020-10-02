"""
solves sudokus
"""
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from sudokusolver.solver import solver


class SudokuSolver(toga.App):
    def startup(self):
        main_box=toga.Box(id="mainbox",style=Pack(direction=COLUMN,padding=5))
        self.solver=solver()

        #create interactive sudokugrid
        #self.sudokurows=[]
        #for i in range(9):
        #    self.sudokurows.append(toga.Box(style=Pack(direction=ROW, padding=0)))
        #    for j in range(9):
        #        self.sudokurows[i].add(toga.Button(label=" ",id='{}{}'.format(i,j),on_press=self.pressedButton,style=Pack(flex=1)))#padding=0,width=30,height=30)))
        #       if (j+1)%3==0 and j<8:
        #            self.sudokurows[i].add(toga.Label(style=Pack(padding=0),text="\t"))
        
        self.grid=[]

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
        
        #init window
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
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
        



    
    #Hide menuebar
    def _create_impl(self):
        factory_app = self.factory.App
        factory_app.create_menus = lambda _: None
        return factory_app(interface=self)



def main():
    return SudokuSolver()

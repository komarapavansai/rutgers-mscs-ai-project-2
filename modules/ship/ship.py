import numpy as np;
import random;
from ..constants import *;

class Ship:

    def __init__(self,grid_size=5):
        self.grid_size=int(grid_size)
        # 0 indicates the blocked cells. Intilialize ship with all blocked cells.
        maze=np.full(fill_value=BLOCKED, dtype= int, shape=(self.grid_size,self.grid_size))
        # Set a random cell to Open Cell
        maze[random.randint(1, self.grid_size - 2)][random.randint(1, self.grid_size - 2)]= OPENED # 1 indicates an Open cell.
        self.maze= maze
        print(f"Initial grid :\n {self.maze}\n")

    def getOpenNeighboursCount(self,row,col):
        count=0;
        neighbours=[]

        if ( col-1 >=0 and self.maze[row][col-1]==OPENED) :
            count=count+1;neighbours.append([row,col-1])
        if (col+1 < self.grid_size and self.maze[row][col+1]==OPENED) :
            count=count+1;neighbours.append([row,col+1])
        if (row-1 >=0 and self.maze[row-1][col]==OPENED) :
            count=count+1;neighbours.append([row-1,col])
        if (row+1 < self.grid_size and self.maze[row+1][col]==OPENED) :
            count=count+1;neighbours.append([row+1,col])
        return (count,neighbours);

    def getBlockedNeighboursCount(self,row,col):
        count=0;
        neighbours=[]

        if ( col-1 >=0 and self.maze[row][col-1]==BLOCKED) :
            count=count+1;neighbours.append([row,col-1])
        if (col+1 < self.grid_size and self.maze[row][col+1]==BLOCKED) :
            count=count+1;neighbours.append([row,col+1])
        if (row-1 >=0 and self.maze[row-1][col]==BLOCKED) :
            count=count+1;neighbours.append([row-1,col])
        if (row+1 < self.grid_size and self.maze[row+1][col]==BLOCKED) :
            count=count+1;neighbours.append([row+1,col])
        return (count,neighbours);

    def getBlockedCells(self, arr):
         return np.argwhere(arr==BLOCKED)

    def ifExpansionPossbile(self):
        blocked_cells=np.argwhere(self.maze == BLOCKED)
        for each_blocked_cell in blocked_cells:
            [x,y]=[each_blocked_cell[0],each_blocked_cell[1]];
            if self.getOpenNeighboursCount(x,y)[0]==1:
                return True;
        return False;

    def getDeadCells(self):
        deadCells=[];
        open_cells=np.argwhere(self.maze == OPENED)
        for each_open_cell in open_cells:
            [x,y]=[each_open_cell[0],each_open_cell[1]];
            if self.getOpenNeighboursCount(x,y)[0]==1:
                deadCells.append([x,y])
        return deadCells;

    def ifEdgeCell(self, row, col):
        if ( col-1 < 0) :
            return True;
        if (col+1 >= self.grid_size) :
            return True;
        if (row-1 < 0 ) :
            return True;
        if (row+1 >= self.grid_size ) :
            return True;
        return False;

    def blockEdgeCells(self):
        for i in range(self.grid_size):
            self.maze[0][i]=BLOCKED
            self.maze[self.grid_size-1][i]=BLOCKED

            if i != 0 and i != self.grid_size-1:
                self.maze[i][0]=BLOCKED
                self.maze[i][self.grid_size-1]=BLOCKED    

    def desigShipLayout(self):
        original_maze= self.maze.copy();
        trimmed_maze = np.array([row[1:self.grid_size-1] for row in self.maze[1:self.grid_size-1]]);
        # print(trimmed_maze.shape)
        self.maze= trimmed_maze; self.grid_size=self.grid_size-2;
        while ( self.ifExpansionPossbile()):
            blockedCells = self.getBlockedCells(self.maze);
            random_cell = blockedCells[np.random.choice(blockedCells.shape[0])];
            neighbourCount,neighbours= self.getOpenNeighboursCount(random_cell[0],random_cell[1])
            if neighbourCount == 1:
                self.maze[random_cell[0]][random_cell[1]]=OPENED;
        deadCells=self.getDeadCells();
        random.shuffle(deadCells);
        count=0;dead_cells_count=(len(deadCells)//2);
        while(count < dead_cells_count):
            eachCell=deadCells.pop();
            x=eachCell[0];y=eachCell[1];
            if self.getOpenNeighboursCount(x,y)[0]==1:
                _, closedNeighbours= self.getBlockedNeighboursCount(x,y);
                [row,col]=closedNeighbours[random.randint(0, len(closedNeighbours) - 1)];
                self.maze[row][col]=OPENED;
                count=count+1;
        # self.blockEdgeCells();
        self.grid_size=self.grid_size+2;
        for i in range(1, self.grid_size-1):
            original_maze[i][1:self.grid_size-1] = self.maze[i-1]
        self.maze=original_maze.copy();
        print(f"The percent of open cells: {(len(np.argwhere(self.maze == OPENED))/(self.grid_size*self.grid_size))*100}%")

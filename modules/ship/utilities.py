from ..constants import *;

def get_open_neighbours(maze,row,col):
    count=0;
    grid_size=maze.shape[0]
    neighbours=[]

    if ( col-1 >=0 and maze[row][col-1] in {OPENED, PATH, RAT}) :
        count=count+1;neighbours.append((row,col-1))
    if (col+1 < grid_size and maze[row][col+1] in {OPENED, PATH, RAT}) :
        count=count+1;neighbours.append((row,col+1))
    if (row-1 >=0 and maze[row-1][col] in {OPENED, PATH, RAT}) :
        count=count+1;neighbours.append((row-1,col))
    if (row+1 < grid_size and maze[row+1][col] in {OPENED, PATH, RAT}) :
        count=count+1;neighbours.append((row+1,col))
    return (count,neighbours);
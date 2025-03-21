import heapq as PriorityQueue;
from ..ship.utilities import get_open_neighbours;
from ..constants import *;
import math;
import numpy as np;
import random;
import csv;
import matplotlib.pyplot as plt;

class Bot1:
    def __init__(self,maze,alpha=0.1):
        self.prev={};
        self.start=(None, None)
        self.end=(None, None)
        self.path=[];
        self.maze = maze
        self.grid_size = maze.shape[0];
        self.bot_postion=self.identifyBotPostion();
        self.alpha=alpha;
        self.rat_detector_action_count=0;
    
    def identifyBotPostion(self):
        botKnowledgeBase= np.argwhere((self.maze == OPENED)| (self.maze == BOT));
        blockedCellSensingActions=0;
        previousMovedDirection=None;
        while(True):
            sensedBlockedCells= self.senseBlockedCells()[0]; # Sense the blocked cells around bot
            print(f"No.of sensed blocked cells: {sensedBlockedCells}");
            print(f"Total possibilites to compute: {len(botKnowledgeBase)}");
            blockedCellSensingActions+=1;
            # Eliminate the cells that doesn't satisfy the conditions
            botKnowledgeBase= np.array(list(filter(lambda x: sensedBlockedCells==self.senseBlockedCells(x[0],x[1])[0],botKnowledgeBase)));
            # Move the bot to direction which is commonly open to further eliminate the possibilities.
            # if previousMovedDirection is None:
            #     direction=self.findCommonOpenDirection(botKnowledgeBase);
            direction=random.sample([(0, -1), (0, 1), (-1, 0), (1, 0)],1)[0];
            print(f"common open direction: {direction}")
            botMovementStatus=self.ifDirectionMovable(direction=direction);
            botKnowledgeBase=np.array(list(filter(lambda x: botMovementStatus==self.ifDirectionMovable(direction,position=x),botKnowledgeBase)));
            if(botMovementStatus and botKnowledgeBase.size > 0):
                print(f"Bot moved to direction: {direction}");
                botKnowledgeBase=botKnowledgeBase + np.array(direction);
                self.moveBot(direction);
                previousMovedDirection=direction;
            if(not botMovementStatus):
                print(f"Bot sensed blocked in common direction")
                previousMovedDirection=None;
            if(len(botKnowledgeBase)==1):
                break;
        
        print(f"Bot position is identified at {botKnowledgeBase[0]}");
        self.blockedCellSensingActions=blockedCellSensingActions;
        print(f"Total blocked cells sensing actions made by bot: {self.blockedCellSensingActions}");
        return (botKnowledgeBase[0][0],botKnowledgeBase[0][1]);

    def moveBot(self,direction):
        botPosition=np.argwhere(self.maze == BOT)[0];
        newBotPosition= botPosition + np.array(direction);
        print(f"Bot new position: {newBotPosition}")
        self.maze[botPosition[0]][botPosition[1]]=OPENED; # Change the bot previous position to Open
        self.maze[newBotPosition[0]][newBotPosition[1]]=BOT; # Update bot's new position

    def ifDirectionMovable(self,direction,position=None):
        if position is None:
            (position_x,position_y)=np.argwhere(self.maze == BOT)[0];
        else:
            (position_x,position_y)=(position[0],position[1]);
        direction_x=direction[0];direction_y=direction[1];
        newpostion_x=position_x+direction_x;
        newpostion_y=position_y+direction_y;
        if (self.maze[newpostion_x][newpostion_y]==BLOCKED):
            return False;
        else:
            return True;

    def findCommonOpenDirection(self,botKnowledgeBase):
        commonDirection=dict();
        commonDirection['left']=0
        commonDirection['right']=0
        commonDirection['up']=0
        commonDirection['down']=0

        directions = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}
        
        for cell in botKnowledgeBase:
            row=cell[0];col=cell[1];
            if ( col-1 >=0 and self.maze[row][col-1] in {OPENED}) :
                commonDirection['left']+=1;
            if (col+1 < self.grid_size and self.maze[row][col+1] in {OPENED}) :
                commonDirection['right']+=1;
            if (row-1 >=0 and self.maze[row-1][col] in {OPENED}) :
                commonDirection['up']+=1;
            if (row+1 < self.grid_size and self.maze[row+1][col] in {OPENED}) :
                commonDirection['down']+=1;
        
        # previousDirectionKey = None
        # for key, value in directions.items():
        #     if value == previousMovedDirection:
        #         previousDirectionKey = key
        #         break

        # if previousDirectionKey in commonDirection:
        #     del commonDirection[previousDirectionKey]
        
        maxCommonDirectionValue=max(commonDirection.values());
        commonOpenDirection = [key for key,value in commonDirection.items() if value==maxCommonDirectionValue];
        # Pick one direction if mutiple directions are have same number of common open cells
        print(f"Directions-> {commonDirection}")
        maxCommonDirection=random.sample(commonOpenDirection,1)[0];
        return directions.get(maxCommonDirection);

    def get_open_neighbours(self,row,col):
        count=0;
        grid_size=self.grid_size
        neighbours=[]

        if ( col-1 >=0 and self.maze[row][col-1] in {OPENED}) :
            count=count+1;neighbours.append((row,col-1))
        if (col+1 < grid_size and self.maze[row][col+1] in {OPENED}) :
            count=count+1;neighbours.append((row,col+1))
        if (row-1 >=0 and self.maze[row-1][col] in {OPENED}) :
            count=count+1;neighbours.append((row-1,col))
        if (row+1 < grid_size and self.maze[row+1][col] in {OPENED}) :
            count=count+1;neighbours.append((row+1,col))
        return (count,neighbours);
    
    def senseBlockedCells(self,x=None,y=None):
        if(x == None or y == None):
            (x,y)=np.argwhere(self.maze == BOT)[0];
        count = 0
        blocked_neighbors = []
    
        if (y-1 >= 0 and self.maze[x][y-1] == BLOCKED):  # Left
            count += 1
            blocked_neighbors.append([x, y-1])

        if (y+1 < self.grid_size and self.maze[x][y+1] == BLOCKED):  # Right
            count += 1
            blocked_neighbors.append([x, y+1])

        if (x-1 >= 0 and self.maze[x-1][y] == BLOCKED):  # Top
            count += 1
            blocked_neighbors.append([x-1, y])

        if (x+1 < self.grid_size and self.maze[x+1][y] == BLOCKED):  # Bottom
            count += 1
            blocked_neighbors.append([x+1, y])

        if (x-1 >= 0 and y-1 >= 0 and self.maze[x-1][y-1] == BLOCKED):  # Top-left
            count += 1
            blocked_neighbors.append([x-1, y-1])

        if (x-1 >= 0 and y+1 < self.grid_size and self.maze[x-1][y+1] == BLOCKED):  # Top-right
            count += 1
            blocked_neighbors.append([x-1, y+1])

        if (x+1 < self.grid_size and y-1 >= 0 and self.maze[x+1][y-1] == BLOCKED):  # Bottom-left
            count += 1
            blocked_neighbors.append([x+1, y-1])

        if (x+1 < self.grid_size and y+1 < self.grid_size and self.maze[x+1][y+1] == BLOCKED):  # Bottom-right
            count += 1
            blocked_neighbors.append([x+1, y+1])

        return count, blocked_neighbors;

    def getPingProbability(self,cell1,cell2=None):
        if cell2 is None:
            (position_x,position_y)=np.argwhere(self.maze == RAT)[0];
        else:
            (position_x,position_y)=(cell2[0],cell2[1]);
        manhattan_distance=abs(position_x-cell1[0])+abs(position_y-cell1[1])
        ping_probability=round(math.exp(-(self.alpha) * (manhattan_distance - 1)),2);
        # print(f"Probaility computed for {cell1},{cell2} -> {ping_probability}");
        self.rat_detector_action_count+=1;
        return ping_probability;

    def set_path(self):
        path=[];
        curr= tuple(self.end);
        while curr is not None:
            path.append((curr));
            curr=self.prev[tuple(curr)];
        path.reverse();
        self.path=path;

    def move_and_get_position(self):
        return self.path.pop(0)

    def execute_strategy(self,maze,start,end):
        print(f"start and end : {start},{end}")
        fringe=[(0,start)];
        self.start=start
        self.end=end
        PriorityQueue.heapify(fringe);
        totalCosts={start:0}
        prev={start:None}
        while bool(fringe):
            # print(f"in while {fringe}")
            curr=PriorityQueue.heappop(fringe)[1]
            if curr == end:
                self.prev=prev
                return True,prev,totalCosts;
            for child in get_open_neighbours(maze,curr[0],curr[1])[1]:
                # print(f"processing child: {child}")
                cost = totalCosts[tuple(curr)] + 1;
                if tuple(child) not in totalCosts:
                    prev[tuple(child)]=curr;
                    totalCosts[tuple(child)]=cost;
                    # print(f"Adding {child} to the fringe")
                    PriorityQueue.heappush(fringe,(cost,child));
                if cost < totalCosts[tuple(child)] :
                    prev[tuple(child)]=curr;
                    totalCosts[tuple(child)]=cost;
                    #replace item in fringe
                    print(f"Updating {child} to the fringe")
                    fringe= list(filter(lambda x: not np.array_equal(x[1], child),fringe))
                    PriorityQueue.heappush(fringe,(cost,child));
        self.prev=prev;
        print('failed')
        return False,prev,totalCosts;

    def is_selected_cell_in_knowledge_base(self,cell,ratKnowledgeBase):
        kb_set = set(map(tuple, ratKnowledgeBase));
        (pos_x,pos_y)= (cell[0],cell[1]);
        if (pos_x,pos_y) in kb_set:
            return True;
        else:
            return False;

    def run_simulation(self,num_time_steps=math.inf,show_animation=True):
        t=0;
        ## At time t = 0, bot starts searching for Space Rat
        (x,y)=self.bot_postion;
        # Get ping probability of Rat
        # ping_probability=self.getPingProbability((x,y));
        ratKnowledgeBase= np.argwhere((self.maze == OPENED) | (self.maze == RAT));
        bot_movements_count=0;
        # if ping_probability > 1:
        #     return [1,bot_movements_count, self.rat_detector_action_count];
        simulation_status=False;
        while t < num_time_steps:
            print(f"At timestep t={t}");
            if (t%2 == 0):
                ping_probability=self.getPingProbability((x,y)); # Get ping probability of Rat
                # print(f"Probability computed to Rat: {ping_probability}");
                ratKnowledgeBase=np.array(list(filter(lambda cell: ping_probability==self.getPingProbability((cell[0],cell[1]),(x,y)) ,ratKnowledgeBase)));
                print(f"Total possibilites to compute: {len(ratKnowledgeBase)}")
            if (t%2 == 1):
                if((len(self.path)==0) or (not self.is_selected_cell_in_knowledge_base(self.path[-1],ratKnowledgeBase))):
                    print(f"Calculating the path")
                    random_cell=random.sample(list(map(tuple, ratKnowledgeBase)),1)[0];
                    self.execute_strategy(self.maze,(x,y),(random_cell[0],random_cell[1]));
                    self.set_path();
                (x,y)=self.move_and_get_position();
                bot_movements_count+=1;
                print(f"curr postion -> {(x,y)}")
                if (self.getPingProbability((x,y)) > 1):  # Check if Bot has reached the Rat
                    print("Bot found the Space Rat.SUCCESS!")
                    simulation_status=True;
                    break;
                self.maze[x][y]=BOT;
                prev_cell=self.prev[(x,y)];
                if prev_cell is not None: self.maze[prev_cell[0]][prev_cell[1]]=PATH;
            # else:            
            #     self.maze[x][y]=PATH
            yield self.maze;
            t=t+1
        data=[int(simulation_status),bot_movements_count, self.rat_detector_action_count];
        print(f"data -> {data}")
        return data;

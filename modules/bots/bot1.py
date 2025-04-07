import heapq as PriorityQueue;
from ..ship.utilities import get_open_neighbours;
from ..constants import *;
import math;
import numpy as np;
import random;
import csv;
import matplotlib.pyplot as plt;
from modules.ship.visualizer import generate_grid;

class Bot1:
    def __init__(self,maze,alpha=0.1):
        self.prev={};
        self.start=(None, None)
        self.end=(None, None)
        self.path=[];
        self.maze = maze
        self.grid_size = maze.shape[0];
        self.bot_postion=self.identifyBotPosition();
        self.alpha=alpha;
        self.rat_detector_action_count=0;
        self.init_rat_probability();
    
    def init_rat_probability(self):
        open_cells = np.argwhere(self.maze==OPENED);
        self.rat_probability = np.zeros((self.grid_size, self.grid_size))
        self.rat_probability[tuple(zip(*open_cells))] = 1 / len(open_cells)  
        # Initialize belief structure (used for Bayes' updates after sensing)
        self.belief = np.copy(self.rat_probability)  # Start with same prior
    
    def getBotCurrentPosition(self):
        bot_pos = np.argwhere(self.maze == BOT)
        if bot_pos.size == 0:
            open_cells = np.argwhere(self.maze == OPENED)
            if open_cells.size == 0:
                raise Exception("No open cells available to place the BOT.")
            random_cell = random.choice(open_cells)
            x, y = random_cell
            self.maze[x][y] = BOT
            print(f"BOT was not found — randomly placed at: ({x}, {y})")
            return (x, y)
        return tuple(bot_pos[0])
    
    def identifyBotPosition(self):
        botKnowledgeBase = np.argwhere((self.maze == OPENED) | (self.maze == BOT))
        previousMovedDirection = None
        recent_positions = []
        wiggle_threshold = 5

        while True:
            sensedBlockedCells = self.senseBlockedCells()[0]
            print(f"Sensed blocked cells: {sensedBlockedCells}")
            print(f"Knowledge base size before filtering: {len(botKnowledgeBase)}")
            self.blockedCellSensingActions += 1

            botKnowledgeBase = np.array(list(filter(
                lambda x: sensedBlockedCells == self.senseBlockedCells(x[0], x[1])[0],
                botKnowledgeBase
            )))

            bot_current_position = self.getBotCurrentPosition()
            recent_positions.append(bot_current_position)
            if len(recent_positions) > wiggle_threshold:
                recent_positions.pop(0)

            is_wiggling = self.isBotWiggling(recent_positions, wiggle_threshold)

            if previousMovedDirection is None or is_wiggling:
                if is_wiggling:
                    print("Wiggling detected! Choosing random direction to break loop.")
                direction = random.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
            else:
                direction = self.findCommonOpenDirection(botKnowledgeBase)

            print(f"Trying to move in direction: {direction}")
            botMovementStatus = self.ifDirectionMovable(direction=direction)

            botKnowledgeBase = np.array(list(filter(
                lambda x: botMovementStatus == self.ifDirectionMovable(direction, position=x),
                botKnowledgeBase
            )))

            if botMovementStatus and botKnowledgeBase.size > 0:
                self.moveBot(direction)
                botKnowledgeBase = botKnowledgeBase + np.array(direction)
                previousMovedDirection = direction
            else:
                previousMovedDirection = None

            if len(botKnowledgeBase) == 1:
                print("Bot position successfully identified!")
                break

        print(f"Final bot position: {botKnowledgeBase[0]}")
        print(f"Total blocked cell sensing actions: {self.blockedCellSensingActions}")
        return (botKnowledgeBase[0][0], botKnowledgeBase[0][1])

    def isBotWiggling(self, positions, threshold):
        if len(positions) < threshold:
            return False
        return positions.count(positions[-1]) >= 3 or len(set(positions[-threshold:])) <= 2


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
        # ping_probability=round(math.exp(-(self.alpha) * (manhattan_distance - 1)),2);
        ping_probability=math.exp(-(self.alpha) * (manhattan_distance - 1));
        # print(f"Probaility computed for {cell1},{cell2} -> {ping_probability}");
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

    def find_path(self,maze,start,end):
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

    def check_if_rat_sensed(self,curr_cell):
        (rat_x,rat_y)=np.argwhere(self.maze == RAT)[0];
        if (rat_x,rat_y)==curr_cell:
            return True;
        else:
            return False;

    def update_rat_probability(self, pos):
        (x,y)=pos;
        remaining_prob = 1 - self.rat_probability[x,y]  # Probability of rat being elsewhere
        self.rat_probability[x,y] = 0  # Rat is definitely NOT here
        ## add explanation here
        if remaining_prob > 0:
            self.rat_probability /= remaining_prob  # Redistribute probability

    def update_belief(self, pos, if_beep_heard):
        """
        Update belief P(rat at (i,j) | beep) using Bayes' Rule.
        Belief is different from rat_probability.
        """
        new_belief = np.zeros_like(self.belief)
        if if_beep_heard:
            print("Bot heard the beep");
        else:
            print("Bot not heard the beep");
        
        # Compute P(beep)
        p_beep = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                likelihood = self.getPingProbability(pos, (i, j))
                if not if_beep_heard:
                    likelihood=1-likelihood;
                p_beep += self.rat_probability[i, j] * likelihood  # Marginalization

        # Compute posterior belief
        if p_beep > 0:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    likelihood = self.getPingProbability(pos, (i, j))
                    if if_beep_heard:  # If bot hears a beep
                        new_belief[i, j] = self.rat_probability[i, j] * likelihood / p_beep
                    else:  # If no beep, reduce probability of nearby cells
                        # new_belief[i, j] = self.rat_probability[i, j] * (1 - likelihood) / (1 - p_beep)
                        new_belief[i, j] = self.rat_probability[i, j] * (1 - likelihood) / (p_beep)

        self.belief=new_belief.copy();

    def print_belief_grid(self):
        print(f"Sum -> {np.sum(self.belief)}")
        print("\nBelief Grid:")
        for i in range(self.belief.shape[0]):
            row_str = " | ".join([f"({i},{j}): {self.belief[i,j]:.4f}" for j in range(self.belief.shape[1])])
            print(row_str)
        print("\n" + "="*50)

    def move_rat(self):
        #Move Rat to random open direction.
        (position_x,position_y)=np.argwhere(self.maze == RAT)[0];
        open_neighbours=get_open_neighbours(self.maze,position_x,position_y)[1];
        (new_pos_x,new_rat_y)=random.sample(open_neighbours, 1)[0];
        self.maze[new_pos_x][new_rat_y]=RAT;
        self.maze[position_x][position_y]=OPENED;
        print(f"Rat's new position: {(new_pos_x,new_rat_y)}");

    def predict_rat_movement(self):
        """
        Predicts the rat's new position based on movement probabilities.
        when rat moves randomly to any adjacent open cell.
        """
        # Define movement directions (Up, Down, Left, Right)
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        new_rat_probability = np.zeros_like(self.rat_probability)
        
        for (i, j), prob in np.ndenumerate(self.rat_probability):
            # Only consider nonzero probability cells, because zero means that Rat is definitely not in that cell. 
            # So, it not required to estimate the movement from that cell.
            if prob > 0: 
                valid_moves = []
                for move in MOVES:
                    ni, nj = i + move[0], j + move[1]
                    if 0 <= ni < self.maze.shape[0] and 0 <= nj < self.maze.shape[1] and self.maze[ni, nj] == OPENED:
                        valid_moves.append((ni, nj))
                
                if valid_moves:
                    prob_per_move = 1 / len(valid_moves)  # Probability split equally among valid moves
                    for ni, nj in valid_moves:
                        # prob means p(rat was in cell x)
                        new_rat_probability[ni, nj] += prob * prob_per_move

        self.rat_probability = new_rat_probability.copy()  # Update rat probability with predicted movement

    def run_simulation(self,num_time_steps=math.inf,rat_movement=False):
        t=0;
        ## At time t = 0, bot starts searching for Space Rat
        open_cells = np.argwhere(self.maze == OPENED) 
        random_cells = random.sample(list(open_cells), 1)
        initial_values = [RAT];
        for (x, y) in random_cells:
            self.maze[x][y] = initial_values.pop(0)
            print(f"Initial postion of Rat at {(x,y)}")
        (x,y)=self.bot_postion;
        bot_movements_count=0;
        # if ping_probability > 1:
        #     return [1,bot_movements_count, self.rat_detector_action_count];
        simulation_status=False;
        useRatSensor=True;
        destination=None;
        while t < num_time_steps:
            print(f"At timestep t={t}");
            
            if(rat_movement):
                # Move Rat in random direction
                print("Rat moving in some random direction");
                self.move_rat();
                # Predict the Rat's 
                print("Predict Rat's movement and update the Rat's KB");
                self.predict_rat_movement();

            # Check if Rat and Bot in same in cell
            if(self.check_if_rat_sensed((x,y))):
                print("Bot found the Space Rat.SUCCESS!")
                simulation_status=True;
                break;
            
            if (useRatSensor):
                print('Starting to use Rat sensor');
                print(f"{(x,y)}")
                self.update_rat_probability((x,y))
                if_beep_heard = np.random.rand() < self.getPingProbability((x,y))
                self.rat_detector_action_count+=1;
                self.update_belief((x,y), if_beep_heard);
                print(f"Calculated Probability: {self.getPingProbability((x,y))}");
                # self.print_belief_grid();
                print(f"{np.argwhere(self.belief == np.max(self.belief))}")
                print(f"Total possibilites to compute: {len(np.argwhere(self.rat_probability != 0))}")
                useRatSensor=False;
                t=t+1;
                continue;
            if (not useRatSensor):
                max_values=np.argwhere(self.belief == np.max(self.belief));
                # destination=np.unravel_index(np.argmax(self.belief), self.belief.shape);
                destination=tuple(random.choice(max_values));
                print(f"Destination is {destination}");
                if(not self.path):
                    print(f"Calculating the path")
                    status=self.find_path(self.maze,(x,y),destination);
                    if(not status[0]):
                        print(f"No short path found :(");
                        return;
                    else:
                        self.set_path();
                (x,y)=self.move_and_get_position();
                bot_movements_count+=1;
                print(f"curr postion -> {(x,y)}")
                # Check if Rat and Bot in same in cell
                if(self.check_if_rat_sensed((x,y))):
                    print("Bot found the Space Rat.SUCCESS!")
                    simulation_status=True;
                    break;
                else:
                    self.rat_probability[x,y] = 0;
                if (len(self.path) == 0):
                    print("Rat not present in picked the destionation. Sense for Rat again");
                    useRatSensor=True;
                self.maze[x][y]=BOT;
                prev_cell=self.prev[(x,y)];
                if prev_cell is not None: self.maze[prev_cell[0]][prev_cell[1]]=PATH;
            # else:            
            #     self.maze[x][y]=PATH
            # yield self.maze;
            t=t+1
        data=[int(simulation_status),bot_movements_count, self.rat_detector_action_count,self.alpha,f"rat_movement={rat_movement}"];
        print(f"data -> {data}")
        # with open('bot1.csv',mode='a',newline='') as file:
        #     writer=csv.writer(file);
        #     writer.writerow(data);
        return data;

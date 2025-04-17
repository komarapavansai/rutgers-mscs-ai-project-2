import heapq as PriorityQueue;
from ..ship.utilities import get_open_neighbours;
from ..constants import *;
import math;
import numpy as np;
import random;
import csv;
import matplotlib.pyplot as plt;
from modules.ship.visualizer import generate_grid,generate_heatmap;
import time;
from scipy.special import comb;

class Bot2:
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
        self.init_rat_probability();
        self.rat_probability = self.mask_blocked_and_edge_cells(self.rat_probability)
        self.belief = self.mask_blocked_and_edge_cells(self.belief)
    
    def init_rat_probability(self):
        open_cells = np.argwhere(self.maze==OPENED);
        self.rat_probability = np.zeros((self.grid_size, self.grid_size));
        # Exclude edges
        valid_cells = [(i, j) for i, j in open_cells if 0 < i < self.grid_size - 1 and 0 < j < self.grid_size - 1]
        
        for i, j in valid_cells:
            self.rat_probability[i, j] = 1 / len(valid_cells) 
        # Initialize belief structure (used for Bayes' updates after sensing)
        self.belief = np.copy(self.rat_probability)  # Start with same prior

    def identifyBotPostion(self):
        open_cells = np.argwhere(self.maze == OPENED) 
        random_cells = random.sample(list(open_cells), 1)
        initial_values = [BOT]

        for (x, y) in random_cells:
            self.maze[x][y] = initial_values.pop(0)
            print(f"Initial postion of BOT at {(x,y)}")

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
        ## When the Rat is not identified in certain cell, then we update the Rat's knowledge base in the following way.
        ## Detailed explanation is given in the project report.
        if remaining_prob > 0:
            self.rat_probability /= remaining_prob  # Redistribute probability

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
                    if 0 <= ni < self.maze.shape[0] and 0 <= nj < self.maze.shape[1] and self.maze[ni, nj] in {OPENED, BOT}:
                        valid_moves.append((ni, nj))
                
                if valid_moves:
                    prob_per_move = 1 / len(valid_moves)  # Probability split equally among valid moves
                    for ni, nj in valid_moves:
                        # prob means p(rat was in cell x)
                        new_rat_probability[ni, nj] += prob * prob_per_move

        self.rat_probability = new_rat_probability.copy()  # Update rat probability with predicted movement

    def mask_blocked_and_edge_cells(self, grid):
        masked = grid.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if self.maze[i, j] == BLOCKED or i == 0 or j == 0 or i == grid.shape[0]-1 or j == grid.shape[1]-1:
                    masked[i, j] = 0
        return masked

    def generate_ring_likelihood(self,grid_shape, bot_pos, num_beeps, num_sensing_iterations, alpha):
        likelihood = np.zeros(grid_shape)

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if self.maze[i, j] == BLOCKED or i == 0 or j == 0 or i == grid_shape[0]-1 or j == grid_shape[1]-1:
                    continue;
                d = abs(bot_pos[0] - i) + abs(bot_pos[1] - j)
                if d == 0:
                    continue  # Skip bot's current location

                p = math.exp(-alpha * (d - 1))
                likelihood[i, j] = comb(num_sensing_iterations, num_beeps) * \
                                   (p ** num_beeps) * ((1 - p) ** (num_sensing_iterations - num_beeps))

        return likelihood;

    def run_simulation(self,num_time_steps=math.inf,rat_movement=False):
        t=0;
        open_cells = np.argwhere(self.maze == OPENED) 
        random_cells = random.sample(list(open_cells), 1)
        initial_values = [RAT];
        for (x, y) in random_cells:
            self.maze[x][y] = initial_values.pop(0)
            print(f"Initial postion of Rat at {(x,y)}")
        (x,y)=self.bot_postion;
        bot_movements_count=0;

        valid_cells = [(i, j) for i, j in open_cells if 0 < i < self.grid_size - 1 and 0 < j < self.grid_size - 1]
        for (i, j) in valid_cells:
            self.rat_probability[i, j] = 1 / len(valid_cells)
        self.belief = self.rat_probability.copy()

        simulation_status=False;
        useRatSensor=True;
        num_sensing_iterations=20;
        sensing_counter = 0;
        print("Selected center sensing")
        destination=None;
        beep_count = 0;
        self.triangulation_belief=None
        botMoving=False;
        visited_locations = set()

        while t < num_time_steps:
            print(f"At timestep t={t}");

            if(rat_movement):
                print("Rat moving in some random direction");
                self.move_rat();
                print("Predict Rat's movement and update the Rat's KB");
                self.predict_rat_movement();

            if(self.check_if_rat_sensed((x,y))):
                print("Bot found the Space Rat.SUCCESS!")
                simulation_status=True;
                break;

            if (useRatSensor):
                print(f"Sensing from {(x,y)}")
                self.update_rat_probability((x,y))

                sensing_counter = 0
                beep_count = 0

                while sensing_counter < num_sensing_iterations:
                    # print(f"log: {sensing_counter}")
                    if_beep_heard = np.random.rand() < self.getPingProbability((x,y))
                    self.rat_detector_action_count+=1;
                    if if_beep_heard:
                        beep_count += 1
                    sensing_counter += 1

                print(f"Total beeps heard from {(x, y)}: {beep_count}")
                ring_likelihood = self.generate_ring_likelihood(
                    grid_shape=self.belief.shape,
                    bot_pos=(x, y),
                    num_beeps=beep_count,
                    num_sensing_iterations=num_sensing_iterations,
                    alpha=self.alpha
                )
                ring_likelihood += 1e-9
                ring_likelihood = self.mask_blocked_and_edge_cells(ring_likelihood)

                posterior = ring_likelihood * self.rat_probability
                posterior = self.mask_blocked_and_edge_cells(posterior)
                posterior /= np.sum(posterior)

                if self.triangulation_belief is None:
                    self.triangulation_belief = posterior
                else:
                    self.triangulation_belief *= posterior

                self.triangulation_belief = self.mask_blocked_and_edge_cells(self.triangulation_belief)
                self.belief = self.triangulation_belief
                self.belief /= np.sum(self.belief)

                # Optional Heatmap visual: 
                # generate_heatmap(self.belief / np.max(self.belief),pos=(x,y))

                max_values = np.argwhere(self.belief == np.max(self.belief))
                destination = tuple(random.choice([cell for cell in max_values if tuple(cell) not in visited_locations]))
                visited_locations.add(destination)
                useRatSensor = False
                botMoving = True

                t += 1
                continue

            if (not useRatSensor):
                if not self.path:
                    print(f"Calculating the path to {destination}")
                    status = self.find_path(self.maze, (x,y), destination);
                    if not status[0]:
                        print(f"No short path found :(")
                        print(f"neibours: {get_open_neighbours(self.maze,x,y)}")
                        return;
                    else:
                        self.set_path();
                        botMoving = True

                (x,y) = self.move_and_get_position();
                bot_movements_count += 1;
                print(f"curr postion -> {(x,y)}")

                if(self.check_if_rat_sensed((x,y))):
                    print("Bot found the Space Rat.SUCCESS!")
                    simulation_status = True;
                    break;
                else:
                    self.rat_probability[x,y] = 0;

                if len(self.path) == 0:
                    botMoving = False
                    print("Rat not present in picked the destination. Sense for Rat again");
                    self.rat_probability[x,y] = 0
                    self.belief[x,y] = 0
                    useRatSensor = True

                self.maze[x][y] = BOT;
                prev_cell = self.prev[(x,y)];
                if prev_cell is not None: self.maze[prev_cell[0]][prev_cell[1]] = PATH;
            # yield self.maze;
            t += 1

        data = [int(simulation_status), bot_movements_count, self.rat_detector_action_count, self.alpha, f"rat_movement={rat_movement}"];
        print(f"data -> {data}")
        return data;

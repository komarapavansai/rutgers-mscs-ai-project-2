from modules.ship.ship import Ship;
from modules.ship.visualizer import generate_grid,animation_function;
from modules.constants import *;
from modules.bots.bot1 import Bot1;
import numpy as np;
import random;

def main():
    ship = Ship(30);
    ship.desigShipLayout();
    # print(ship.maze)
    open_cells = np.argwhere(ship.maze == OPENED) 
    random_cells = random.sample(list(open_cells), 2)
    initial_values = [BOT, RAT]#, BUTTON]

    for (x, y) in random_cells:
        ship.maze[x][y] = initial_values.pop(0)
        print(f"Initial postion of BOT at {(x,y)}")
    # generate_grid(ship.maze);
    bot=Bot1(maze=ship.maze)
    animation_function(bot.run_simulation());


if __name__ == "__main__":
    main()
from modules.ship.ship import Ship;
from modules.ship.visualizer import generate_grid,animation_function;
from modules.constants import *;
from modules.bots.bot1 import Bot1;
from modules.bots.bot2 import Bot2;
import numpy as np;
import random;

def main():
    ship = Ship(30);
    ship.desigShipLayout();
    bot=Bot2(maze=ship.maze.copy())
    # animation_function(bot.run_simulation(rat_movement=True));
    bot.run_simulation(rat_movement=False);


if __name__ == "__main__":
    main()
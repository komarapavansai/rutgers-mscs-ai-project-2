import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np;
from ..constants import *;
from matplotlib.animation import FuncAnimation
from functools import partial  

def generate_grid(grid):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.clear()
    grid_size=grid.shape[0];    

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    for i in range(grid_size):
        for j in range(grid_size):
            # if grid[i, j] > 0 and grid[i,j] < 0.45:
            #     color = '#00BFFF'
            # elif grid[i, j] >=0.45 and grid[i,j] < 0.8:
            #     color = '#BA55D3'
            # elif grid[i, j] >=0.8 and grid[i,j] < 1:
            #     color = '#32CD32'
            if grid[i, j] > 0.6 and grid[i, j] < 1:
                color = '#32CD32'  # Very confident
            elif grid[i, j] > 0.2 and grid[i, j] <= 0.6 :
                color = '#BA55D3'  # Strong confidence
            elif grid[i, j] > 0 and grid[i, j] <=0.2 :
                color = '#00BFFF'  # Moderate confidence
            # if grid[i, j] == OPENED:
            #     color = 'white'
            # elif grid[i, j] == RAT:
            #     color = 'red'
            # elif grid[i, j] == BOT:
            #     color = 'green'
            # elif grid[i, j] == START:
            #     color = 'orange'
            # elif grid[i, j] == PATH:
            #     color = 'blue'
            else:
                color = 'gray'
            rect=patches.Rectangle((j, grid_size-i-1), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

    ax.set_xticks(np.arange(0, grid_size+1, 1))
    ax.set_yticks(np.arange(0, grid_size+1, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    plt.draw();plt.show();

def generate_heatmap(grid, cmap=plt.cm.Reds, pos=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.clear()
    grid_size = grid.shape[0]

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    norm = plt.Normalize(vmin=np.min(grid), vmax=np.max(grid))  # normalize values relative to grid

    for i in range(grid_size):
        for j in range(grid_size):
            value = grid[i, j]
            color = cmap(norm(value)) if value > 0 else 'white'  # white for 0/blocked cells
            if pos == (i,j):
                color = 'blue' 
            rect = patches.Rectangle((j, grid_size - i - 1), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    plt.draw()
    plt.show()

fig, ax = plt.subplots(figsize=(8, 8))

def update_grid(frame, grid_generator):
    ax.clear()
    try:
        grid = frame
        grid_size = grid.shape[0]
        ax.set_xlim(0,grid_size)
        ax.set_ylim(grid_size,0)

        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == OPENED:
                    color = 'white'
                elif grid[i, j] == RAT:
                    color = 'red'
                elif grid[i, j] == BOT:
                    color = 'green'
                elif grid[i, j] == START:
                    color = 'orange'
                elif grid[i, j] == PATH:
                    color = 'blue'
                else:
                    color = 'gray'

                rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)

        ax.set_xticks(np.arange(0, grid_size+1, 1))
        ax.set_yticks(np.arange(0, grid_size+1, 1))
        ax.grid(which='both', color='black', linestyle='-', linewidth=1)
        plt.draw()
    except StopIteration:
        # If StopIteration occurs, it means the generator has completed
        print("Generator has finished. Animation complete!")
        plt.close()  # Close the plot when the generator finishes

def animation_function(grid_generator):
    generator= grid_generator;
    animation=FuncAnimation(fig, partial(update_grid, grid_generator=generator), frames=generator, interval=1, repeat=False)
    plt.show()

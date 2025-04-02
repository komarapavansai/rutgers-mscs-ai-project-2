import numpy as np
import time
import csv
import datetime
from concurrent.futures import ProcessPoolExecutor
from modules.ship.ship import Ship
from modules.constants import *
from modules.bots.bot1 import Bot1
from modules.bots.bot2 import Bot2

# Configuration
NUM_LAYOUTS = 10
RUNS_PER_LAYOUT = 100
ALPHAS = np.round(np.arange(0.1, 2.1, 0.1), 2)

# Timestamped output filename
timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_file = f"results/parallel_results_{timestamp}.csv"

# Simulation function to run per process
def run_bot_simulation(args):
    layout_id, run_id, alpha, bot_label, maze = args

    try:
        bot_class = Bot2 if bot_label == 'bot2' else Bot1
        bot = bot_class(maze=maze.copy(), alpha=alpha)

        start_time = time.time()
        success, bot_moves, sensor_acts, *_ = bot.run_simulation(rat_movement=False)
        runtime = round(time.time() - start_time, 4)

        return [layout_id, run_id, bot_label, alpha, success, bot_moves, sensor_acts, False, runtime]

    except Exception as e:
        print(f"[ERROR] {bot_label} failed at α={alpha}, layout={layout_id}, run={run_id}: {e}")
        return [layout_id, run_id, bot_label, alpha, 0, -1, -1, False, -1]  # Error marker

# Generate job list
jobs = []

for layout_id in range(1, NUM_LAYOUTS + 1):
    ship = Ship(30)
    ship.desigShipLayout()
    maze = ship.maze.copy()

    for run_id in range(1, RUNS_PER_LAYOUT + 1):
        for alpha in ALPHAS:
            for bot_label in ['bot2', 'bot1']:
                jobs.append((layout_id, run_id, alpha, bot_label, maze))

# Run jobs in parallel
print(f"Starting {len(jobs)} parallel simulations...")
with ProcessPoolExecutor() as executor:
    results = list(executor.map(run_bot_simulation, jobs))

# Write results to CSV
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow([
        'layout_id', 'run_id', 'bot', 'alpha',
        'success', 'bot_movements', 'sensor_actions',
        'rat_movement', 'runtime_seconds'
    ])
    
    writer.writerows(results)

print(f"All simulations completed. Results saved to: {output_file}")

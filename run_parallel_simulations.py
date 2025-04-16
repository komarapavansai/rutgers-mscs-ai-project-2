import numpy as np
import time
import csv
import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.ship.ship import Ship
from modules.constants import *
from modules.bots.bot1 import Bot1
from modules.bots.bot2 import Bot2

# Configuration
NUM_LAYOUTS = 1
RUNS_PER_LAYOUT = 500
# ALPHAS = np.round(np.arange(0.1, 2.1, 0.1), 2)
ALPHAS = np.round(np.concatenate([
    # np.arange(0.01, 0.02, 0.01),
    # np.arange(0.01, 0.11, 0.02),    # [0.01, 0.03, ..., 0.09]
    # np.arange(0.1, 0.4, 0.1)      # [0.1, 0.2, ..., 2.0]
    np.arange(0.7, 1.1, 0.1)        # [0.1, 0.2, ..., 1.0]
]), 2)

# Worker function to simulate one job
def run_bot_simulation(args):
    layout_id, run_id, alpha, bot_label, maze = args

    try:
        bot_class = Bot2 if bot_label == 'bot2' else Bot1
        bot = bot_class(maze=maze.copy(), alpha=alpha)

        start_time = time.time()
        success, bot_moves, sensor_acts, *_ = bot.run_simulation(rat_movement=True)
        runtime = round(time.time() - start_time, 4)

        return [layout_id, run_id, bot_label, alpha, success, bot_moves, sensor_acts, True, runtime]

    except Exception as e:
        print(f"[ERROR] {bot_label} failed at α={alpha}, layout={layout_id}, run={run_id}: {e}")
        return [layout_id, run_id, bot_label, alpha, 0, -1, -1, True, -1]  # Error marker

def main():
    # Create results folder if missing
    os.makedirs("results", exist_ok=True)

    # Timestamped file
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file = f"results/parallel_results_{timestamp}.csv"

    # Prepare all jobs
    jobs = []
    for layout_id in range(1, NUM_LAYOUTS + 1):
        ship = Ship(30)
        ship.desigShipLayout()
        maze = ship.maze.copy()

        for run_id in range(1, RUNS_PER_LAYOUT + 1):
            for alpha in ALPHAS:
                # for bot_label in ['bot2', 'bot1']:
                for bot_label in ['bot1']:
                    jobs.append((layout_id, run_id, alpha, bot_label, maze.copy()))

    print(f"Submitting {len(jobs)} jobs...")

    # Write and flush results in real-time
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'layout_id', 'run_id', 'bot', 'alpha',
            'success', 'bot_movements', 'sensor_actions',
            'rat_movement', 'runtime_seconds'
        ])

        with ProcessPoolExecutor() as executor:
            future_to_job = {executor.submit(run_bot_simulation, job): job for job in jobs}

            for idx, future in enumerate(as_completed(future_to_job), 1):
                result = future.result()
                writer.writerow(result)

                if idx % 100 == 0:
                    print(f"Completed {idx} / {len(jobs)} simulations")

    print(f"All simulations finished. Results saved to: {output_file}")

# Safe entry point
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

import numpy as np
import csv
import time
from modules.ship.ship import Ship
from modules.constants import *
from modules.bots.bot1 import Bot1
from modules.bots.bot2 import Bot2

# Configs
num_layouts = 10
runs_per_layout = 100
alpha_values = np.arange(0.1, 2.1, 0.1)
output_csv = "simulation_log.csv"

# Write CSV header
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'layout_id', 'run_id', 'bot', 'alpha',
        'success', 'bot_movements', 'sensor_actions',
        'rat_movement', 'runtime_seconds'
    ])

# Run experiments
for layout_id in range(1, num_layouts + 1):
    ship = Ship(30)
    ship.desigShipLayout()
    maze = ship.maze.copy()

    for run_id in range(1, runs_per_layout + 1):
        for alpha in alpha_values:
            alpha = round(float(alpha), 2)

            for bot_class, bot_label in [(Bot2, 'bot2'), (Bot1, 'bot1')]:
                try:
                    print(f"\n[Layout {layout_id}] Run {run_id} | {bot_label.upper()} | α={alpha:.2f}")

                    bot = bot_class(maze=maze.copy(), alpha=alpha)
                    start_time = time.time()
                    success, bot_moves, sensor_acts, *_ = bot.run_simulation(rat_movement=False)
                    runtime = round(time.time() - start_time, 4)

                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            layout_id, run_id, bot_label, alpha,
                            success, bot_moves, sensor_acts,
                            False, runtime
                        ])

                except Exception as e:
                    print(f"[ERROR] {bot_label} failed on α={alpha:.2f}, run {run_id}: {e}")

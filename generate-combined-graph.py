import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

# Bot files and colors
bot_labels = ['bot1', 'bot2']
bot_files = ['bot1.csv', 'bot2.csv']
colors = ['blue', 'green']

# Store total actions grouped by alpha
total_actions = {bot: defaultdict(list) for bot in bot_labels}
rat_movement = True

# Read CSV files and compute total actions = bot_moves + sensor_actions
for bot_label, file_name in zip(bot_labels, bot_files):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5 and str(rat_movement).lower() in row[4].lower():
                alpha = round(float(row[3]), 2)
                bot_moves = int(row[1])
                sensor_uses = int(row[2])
                total = bot_moves + sensor_uses
                total_actions[bot_label][alpha].append(total)

# Plotting
plt.figure(figsize=(10, 6))
for i, bot_label in enumerate(bot_labels):
    alphas = sorted(total_actions[bot_label].keys())
    avg_total_actions = [np.mean(total_actions[bot_label][alpha]) for alpha in alphas]
    plt.plot(alphas, avg_total_actions, linestyle='-', color=colors[i], linewidth=2, label=bot_label)

# Graph formatting
plt.xlabel('Alpha (Sensor Sensitivity)', fontsize=14)
plt.ylabel('Average Total Actions (Moves + Senses)', fontsize=14)
plt.title('Total Actions vs Alpha (Rat Stationary)' if not rat_movement else 'Total Actions vs Alpha (Moving Rat)', fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(np.round(np.arange(0.0, 1.1, 0.1), 2))
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

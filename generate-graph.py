import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

# Bot files and colors
bot_labels = ['bot1', 'bot2']
bot_files = ['bot1.csv', 'bot2.csv']
colors = ['blue', 'green']

# Dictionaries to store grouped data
bot_movements = {bot: defaultdict(list) for bot in bot_labels}
sensor_actions = {bot: defaultdict(list) for bot in bot_labels}

# Read CSV files and extract data per alpha (rat_movement = False only)
for bot_label, file_name in zip(bot_labels, bot_files):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5 and 'false' in row[4].lower():
                alpha = round(float(row[3]), 2)
                bot_moves = int(row[1])
                sensor_uses = int(row[2])
                bot_movements[bot_label][alpha].append(bot_moves)
                sensor_actions[bot_label][alpha].append(sensor_uses)

# Plotting: dual subplot
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

for i, bot_label in enumerate(bot_labels):
    alphas = sorted(bot_movements[bot_label].keys())
    avg_moves = [np.mean(bot_movements[bot_label][alpha]) for alpha in alphas]
    avg_sensors = [np.mean(sensor_actions[bot_label][alpha]) for alpha in alphas]

    axs[0].plot(alphas, avg_moves, linestyle='-', color=colors[i], linewidth=2, label=bot_label)
    axs[1].plot(alphas, avg_sensors, linestyle='--', color=colors[i], linewidth=2, label=bot_label)

# Subplot 1 - Bot Movements
axs[0].set_ylabel('Avg Bot Movements', fontsize=14)
axs[0].set_title('Bot Movements and Sensor Usage vs Alpha (Rat Stationary)', fontsize=16)
axs[0].grid(True)
axs[0].legend()
axs[0].set_ylim(0, 4000)

# Subplot 2 - Sensor Usage
axs[1].set_xlabel('Alpha (Sensor Sensitivity)', fontsize=14)
axs[1].set_ylabel('Avg Sensor Uses', fontsize=14)
axs[1].grid(True)
axs[1].legend()
axs[1].set_ylim(0)  # auto upper

# X-axis range
plt.xticks(np.round(np.arange(0.1, 2.1, 0.1), 2))

# Layout fix
plt.tight_layout()
plt.show()

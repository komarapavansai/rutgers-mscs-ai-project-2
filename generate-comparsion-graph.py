import matplotlib.pyplot as plt
import csv
import numpy as np
from collections import defaultdict

# Config
bot_labels = ['bot1', 'bot2']
bot_files = ['bot1.csv', 'bot2.csv']
colors = ['blue', 'green']
alphas_to_plot = np.round(np.arange(0.0, 1.1, 0.1), 2)

# Initialize dictionaries
comparison_data = {
    'bot1': {'stationary': defaultdict(list), 'moving': defaultdict(list)},
    'bot2': {'stationary': defaultdict(list), 'moving': defaultdict(list)},
}

# Read CSVs and fill data
for bot_label, file_name in zip(bot_labels, bot_files):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 5:
                alpha = round(float(row[3]), 2)
                bot_moves = int(row[1])
                sensor_uses = int(row[2])
                total = bot_moves + sensor_uses

                rat_move_flag = 'moving' if 'true' in row[4].lower() else 'stationary'
                comparison_data[bot_label][rat_move_flag][alpha].append(total)

# Plotting
plt.figure(figsize=(10, 6))

for i, bot_label in enumerate(bot_labels):
    for mode, linestyle in [('stationary', '-'), ('moving', '--')]:
        alpha_keys = sorted(comparison_data[bot_label][mode].keys())
        y_vals = [np.mean(comparison_data[bot_label][mode][a]) for a in alpha_keys]
        label = f"{bot_label} ({mode})"
        plt.plot(alpha_keys, y_vals, linestyle=linestyle, color=colors[i], linewidth=2, label=label)

# Labels and style
plt.xlabel('Alpha (Sensor Sensitivity)', fontsize=14)
plt.ylabel('Average Total Actions (Moves + Senses)', fontsize=14)
plt.title('Bot Performance Comparison: Stationary vs Moving Rat', fontsize=16)
plt.grid(True)
plt.xticks(alphas_to_plot)
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

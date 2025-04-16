import csv
import glob

# Find all result files
input_files = glob.glob('results/parallel_results_*.csv')
bot1_file = 'bot1.csv'
bot2_file = 'bot2.csv'

# Prepare writers for bot files (append mode)
with open(bot1_file, mode='a', newline='') as b1file, \
     open(bot2_file, mode='a', newline='') as b2file:

    writer_bot1 = csv.writer(b1file)
    writer_bot2 = csv.writer(b2file)

    for input_file in input_files:
        print(f"Processing: {input_file}")
        with open(input_file, mode='r', newline='') as infile:
            reader = csv.DictReader(infile)
            rows_bot1 = []
            rows_bot2 = []

            for row in reader:
                out_row = [
                    row['success'],
                    row['bot_movements'],
                    row['sensor_actions'],
                    row['alpha'],
                    f"rat_movement={row['rat_movement']}"
                ]

                if row['bot'] == 'bot1':
                    rows_bot1.append(out_row)
                elif row['bot'] == 'bot2':
                    rows_bot2.append(out_row)

        # Append to each bot file
        writer_bot1.writerows(rows_bot1)
        writer_bot2.writerows(rows_bot2)

        # Clear the result file after processing
        with open(input_file, mode='w', newline='') as cleared_file:
            writer = csv.writer(cleared_file)
            writer.writerow([
                'layout_id', 'run_id', 'bot', 'alpha',
                'success', 'bot_movements', 'sensor_actions',
                'rat_movement', 'runtime_seconds'
            ])

print("All results moved to bot1.csv and bot2.csv, result files cleared.")

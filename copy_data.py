import csv
import glob

# Match all result files
input_files = glob.glob('results/parallel_results_*.csv')

# Output files
bot1_file = 'bot1.csv'
bot2_file = 'bot2.csv'

# Open output files in append mode
with open(bot1_file, mode='a', newline='') as b1file, \
     open(bot2_file, mode='a', newline='') as b2file:

    writer_bot1 = csv.writer(b1file)
    writer_bot2 = csv.writer(b2file)

    for input_file in input_files:
        print(f"Processing: {input_file}")

        with open(input_file, mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                # Format rat_movement to "rat_movement=False" or "rat_movement=True"
                rat_movement_str = f"rat_movement={row['rat_movement'].strip()}"

                out_row = [
                    row['success'],
                    row['bot_movements'],
                    row['sensor_actions'],
                    row['alpha'],
                    rat_movement_str
                ]

                if row['bot'] == 'bot1':
                    writer_bot1.writerow(out_row)
                elif row['bot'] == 'bot2':
                    writer_bot2.writerow(out_row)

print("All rows copied to bot1.csv and bot2.csv with formatted rat_movement column")

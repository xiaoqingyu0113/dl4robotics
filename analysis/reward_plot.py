import numpy as np
import matplotlib.pyplot as plt
import csv


def load_csv_to_dict(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Get the header and create a dictionary with header names as keys
        header = next(reader)
        data_dict = {key: [] for key in header}
        
        # Iterate over each row, and append the data to the corresponding key in the dictionary
        for row in reader:
            for key, value in zip(header, row):
                data_dict[key].append(float(value))
                
        # Convert lists to numpy arrays
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])
            
        return data_dict



PPO_3 = load_csv_to_dict('PPO_3.csv')
PPO_13 = load_csv_to_dict('PPO_13.csv')


fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)

ax.plot(PPO_3['Step'],PPO_3['Value'],label='no ball landing reward')
ax.plot(PPO_13['Step'],PPO_13['Value'],label='add ball landing reward')
ax.legend()
ax.grid(True)
plt.savefig('isaac_reward.png',dpi=300)
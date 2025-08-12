import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the data
df = pd.read_csv('results/metrics_2024-12-03_12-39.csv')

# Function to parse strings into Python objects
def parse_string(s):
    try:
        return ast.literal_eval(s)
    except:
        return s

# Parse the 'reward' and 'total_time' columns
df['reward'] = df['reward'].apply(parse_string)
df['total_time'] = df['total_time'].apply(parse_string)
df['mines'] = df['mines_encountered'].apply(parse_string)

# Convert to numeric and handle errors
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
df['mines'] = pd.to_numeric(df['mines'], errors='coerce')

# Drop rows with missing values in 'reward' or 'total_time'
df = df.dropna(subset=['reward', 'total_time', 'mines'])

# Plot Rewards Over Time
plt.figure(figsize=(10, 5))
plt.plot(df['run'], df['reward'], marker='o')
plt.title('Rewards Over Time')
plt.xlabel('Run')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('results/plots/reward.png')

# Plot Total Time Over Time
plt.figure(figsize=(10, 5))
plt.plot(df['run'], df['total_time'], marker='o', color='orange')
plt.title('Total Time Over Time')
plt.xlabel('Run')
plt.ylabel('Total Time')
plt.grid(True)
plt.savefig('results/plots/time.png')


# Plot Total Time Over Time
plt.figure(figsize=(10, 5))
plt.plot(df['run'], df['mines'], marker='o', color='red')
plt.title('Total Time Over Time')
plt.xlabel('Run')
plt.ylabel('Total Time')
plt.grid(True)
plt.savefig('results/plots/mines.png')

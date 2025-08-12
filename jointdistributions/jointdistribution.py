import numpy as np
import pandas as pd
import random

# Number of samples to generate
num_samples = 10

# Define mean and covariance for continuous variables
mean_values = [0.1025, 5, 4.25, 65.75]  # [ visibility, precipitation, wind_speed, temperature]
cov_matrix = [
    [1, -0.7, 0.3, 0.5], #visibility
    [-0.7, 1, 0.5, -0.4], #precipitation
    [0.3, 0.5, 1, 0.6], #wind_speed
    [0.5, -0.4, 0.6, 1]  # temperature
]

# Generate continuous variable samples, take absolute value to avoid getting negative values
data = np.abs(np.random.multivariate_normal(mean_values, cov_matrix, num_samples))

# Create a DataFrame
df = pd.DataFrame(data, columns=["visibility", "precipitation", "wind_speed", "temperature"])

# Generate random samples
data = np.random.multivariate_normal(mean_values, cov_matrix, num_samples)

#Generate accuracy from uniform 0.3, 0.95
accuracies = np.random.uniform(0.3, 0.95, num_samples) #Make this a function

#Add accuracy column
df['accuracy'] = np.random.choice(accuracies, size=num_samples)


#Generate a random surface
surfaces = ["grassy", "rocky", "wooded", "swampy", "sandy"]

#Add surface column
df['surface'] = np.random.choice(surfaces, size=num_samples)

#Generate a random time
times = ["0000", "0100","0200","0300","0400","0500","0600","0700","0800","0900","1000","1100","1200","1300","1400","1500","1600","1700","1800","1900","2000","2100","2200","2300"]

#Add times colum
time = np.random.choice(times, size=1)

# Display time
print(time)

# Display the first few samples
print(df)

#Assign one row of these values to an edge in the simulation.

# accuracy = f(time, surface, wind_speed, temperature, precip, vis)
#Black box spits out an estimate, user knows metadata and the resulting estimate from the metadata

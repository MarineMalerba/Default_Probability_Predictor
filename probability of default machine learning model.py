import numpy as np
import pandas as pd
import os
from itertools import combinations
from sklearn.metrics import mean_squared_error

# File path to your CSV file
file_path = '/Users/marinemalerba/Documents/CS/PYTHON/JPMorgan Chase & Co Quant Job Sim/Loan_Data.csv'

# Function to calculate log-likelihood for a given set of bucket boundaries
def calculate_log_likelihood(boundaries, scores, defaults):
    log_likelihood = 0
    for i in range(len(boundaries) - 1):
        bucket_mask = (scores >= boundaries[i]) & (scores < boundaries[i+1])
        n_i = np.sum(bucket_mask)
        k_i = np.sum(defaults[bucket_mask])
        p_i = k_i / n_i if n_i > 0 else 0
        if p_i > 0 and p_i < 1:
            log_likelihood += k_i * np.log(p_i) + (n_i - k_i) * np.log(1 - p_i)
    return log_likelihood

try:
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        # Load data
        df = pd.read_csv(file_path)
        print("File loaded successfully")

        # Extract relevant columns
        fico_scores = df['fico_score']
        defaults = df['default']
        print("Extracted FICO scores and defaults")

        # Number of buckets
        n_buckets = 5  

        # Initial Bucketing
        min_score = fico_scores.min()
        max_score = fico_scores.max()
        bucket_boundaries = np.linspace(min_score, max_score, n_buckets + 1)
        print("Initial Bucket boundaries initialized:", bucket_boundaries)

    # Dynamic programming to find optimal buckets
    
        #Initialize variables to keep track of the best bucket boundaries and their corresponding log-likelihood
        best_boundaries = None
        best_log_likelihood = -np.inf

        # Iterate through possible bucket combinations using combinations to reduce the number of iterations
        possible_split_points = np.arange(min_score, max_score)
        for split_points in combinations(possible_split_points, n_buckets - 1):
            boundaries = np.sort(np.array([min_score] + list(split_points) + [max_score]))
            log_likelihood = calculate_log_likelihood(boundaries, fico_scores, defaults)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_boundaries = boundaries

        # Display the optimal bucket boundaries
        print("Optimal Bucket Boundaries:", best_boundaries)

except Exception as e:
    print(f"An error occurred: {e}")
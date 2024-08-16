import numpy as np
import pandas as pd
import os
from itertools import combinations

# File path to your CSV file
file_path = # path to 'Loan_Data.csv'

# Function to calculate log-likelihood for a given set of bucket boundaries
def calculate_log_likelihood(boundaries, scores, defaults):
    log_likelihood = 0
    for i in range(len(boundaries) - 1):
        bucket_mask = (scores >= boundaries[i]) & (scores < boundaries[i + 1])
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
        fico_scores = df['fico_score'].values
        defaults = df['default'].values
        print("Extracted FICO scores and defaults")

        # Number of buckets
        n_buckets = 5  

        # Reduce precision of FICO scores to reduce the number of split points
        precision = 10  # Adjust precision as needed (e.g., 10 means rounding to the nearest 10)
        fico_scores_rounded = np.round(fico_scores / precision) * precision
        min_score = fico_scores_rounded.min()
        max_score = fico_scores_rounded.max()

        # Initial Bucketing
        bucket_boundaries = np.linspace(min_score, max_score, n_buckets + 1)

    # Dynamic programming to find optimal buckets

        # Initialize variables to keep track of the best bucket boundaries and their corresponding log-likelihood
        best_boundaries = None
        best_log_likelihood = -np.inf

        # Reduce the number of possible split points for initial testing
        possible_split_points = np.unique(fico_scores_rounded)

        # Iterate through possible bucket combinations
        for count, split_points in enumerate(combinations(possible_split_points, n_buckets - 1), start=1):
            boundaries = np.sort(np.array([min_score] + list(split_points) + [max_score]))
            log_likelihood = calculate_log_likelihood(boundaries, fico_scores_rounded, defaults)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_boundaries = boundaries

        # Display the optimal bucket boundaries
        print("Optimal Bucket Boundaries:", best_boundaries)

        # Compute and display statistics for each bucket
        fico_scores_rounded = np.array(fico_scores_rounded)
        defaults = np.array(defaults)
        for i in range(n_buckets - 1):
            bucket_mask = (fico_scores_rounded >= best_boundaries[i]) & (fico_scores_rounded < best_boundaries[i + 1])
            n = np.sum(bucket_mask)
            k = np.sum(defaults[bucket_mask])
            p = k / n if n > 0 else 0
            print(f"Bucket [{best_boundaries[i]}, {best_boundaries[i + 1]}):")
            print(f"  Number of observations: {n}")
            print(f"  Number of defaults: {k}")
            print(f"  Default rate: {p:.2f}")

except Exception as e:
    print(f"An error occurred: {e}")

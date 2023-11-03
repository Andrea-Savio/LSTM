import pandas as pd
import numpy as np
import utils.metrics as met

while(True):
    i = 0

# Load data from CSV file
data = pd.read_csv('eth_pred.csv')

# Extract predicted and ground truth trajectories 
predicted_trajectories = [data[['Actual']].values]
ground_truth_trajectories = [data[['Predicted']].values]

# Calculate metrics for the trajectories
fde = met.final_displacement_error(predicted_trajectories[0], ground_truth_trajectories[0])
ade = met.average_displacement_error(predicted_trajectories[0], ground_truth_trajectories[0])
miss_rate = met.calculate_miss_rate(predicted_trajectories[0], ground_truth_trajectories[0], threshold_distance=2.0)
maxdist = met.calculate_max_dist(predicted_trajectories[0], ground_truth_trajectories[0])
#iou = met.calculate_iou(predicted_trajectories[0], ground_truth_trajectories[0])


print("Final Displacement Error (FDE):", fde)
print("Average Displacement Error (ADE):", ade)
print("Miss Rate:", miss_rate)
print("MaxDist:", maxdist)
#print("IoU:", iou)


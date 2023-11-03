import numpy as np

def calculate_iou(box1, box2):
  """
  Calculate the Intersection over Union (IoU) between two bounding boxes.

  Parameters:
  - box1: A tuple or list representing the first bounding box as (x1, y1, x2, y2),
          where (x1, y1) are the coordinates of the top-left corner, and (x2, y2)
          are the coordinates of the bottom-right corner.
  - box2: A tuple or list representing the second bounding box in the same format.

  Returns:
  - IoU (Intersection over Union): A floating-point value between 0 and 1 representing
    the degree of overlap between the two bounding boxes. Higher values indicate greater
    overlap.
  """
  x1_1, y1_1, x2_1, y2_1 = box1
  x1_2, y1_2, x2_2, y2_2 = box2

  # Calculate the coordinates of the intersection area
  x1_intersection = max(x1_1, x1_2)
  y1_intersection = max(y1_1, y1_2)
  x2_intersection = min(x2_1, x2_2)
  y2_intersection = min(y2_1, y2_2)

  # Calculate the area of intersection. If no intersection, the area is 0.
  intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(0, y2_intersection - y1_intersection + 1)

  # Calculate the area of the two bounding boxes
  box1_area = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
  box2_area = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

  # Calculate the IoU
  iou = intersection_area / float(box1_area + box2_area - intersection_area)

  return iou

def euclidean_distance(p1, p2):
  """Calculate the Euclidean distance between two points."""
  return np.linalg.norm(p1 - p2)

def final_displacement_error(predicted_trajectory, ground_truth_trajectory):
  """
  Calculate the Final Displacement Error (FDE) between the predicted and ground truth trajectories.

  Parameters:
  - predicted_trajectory: List or array of predicted positions (numpy array) at each time step.
  - ground_truth_trajectory: List or array of ground truth positions (numpy array) at each time step.

  Returns:
  - FDE (Final Displacement Error): Euclidean distance between the final predicted position and
    the final ground truth position.
  """
  p_final = predicted_trajectory[-1]
  g_final = ground_truth_trajectory[-1]
  return euclidean_distance(p_final, g_final)

def average_displacement_error(predicted_trajectory, ground_truth_trajectory):
  """
  Calculate the Average Displacement Error (ADE) between the predicted and ground truth trajectories.

  Parameters:
  - predicted_trajectory: List or array of predicted positions (numpy array) at each time step.
  - ground_truth_trajectory: List or array of ground truth positions (numpy array) at each time step.

  Returns:
  - ADE (Average Displacement Error): Average Euclidean distance between predicted and ground truth
    positions at each time step.
  """
  num_time_steps = len(predicted_trajectory)
  if num_time_steps != len(ground_truth_trajectory):
    raise ValueError("The lengths of predicted and ground truth trajectories must be the same.")
    
  total_distance = 0.0
  for i in range(num_time_steps):
    p_i = predicted_trajectory[i]
    g_i = ground_truth_trajectory[i]
    total_distance += euclidean_distance(p_i, g_i)
    
  return total_distance #/num_time_steps

def calculate_miss_rate(predicted_trajectories, ground_truth_trajectories, threshold_distance):
  """
  Calculate the Miss Rate for trajectory prediction as the number of predicted trajectories
  within a specified threshold distance of the ground truth endpoint.
  """
     
  total_predictions = len(predicted_trajectories)
  successful_predictions = 0

  for i in range(total_predictions):
    predicted_trajectory = predicted_trajectories[i]
    ground_truth_trajectory = ground_truth_trajectories[i]

    # Calculate the Euclidean distance between the final predicted position and ground truth endpoint
    final_predicted_position = predicted_trajectory[-1]
    final_ground_truth_position = ground_truth_trajectory[-1]
    distance = np.linalg.norm(final_predicted_position - final_ground_truth_position)

    # Check if the prediction is within the threshold distance
    if distance <= threshold_distance:
      successful_predictions += 1

  miss_rate = 1.0 - (successful_predictions / total_predictions)
  return miss_rate

def calculate_max_dist(predicted_trajectory, ground_truth_trajectory):
  max_dist = 0
  for i in range(len(predicted_trajectory)):
    distance = np.linalg.norm(ground_truth_trajectory[i] - predicted_trajectory[i])
    if distance > max_dist:
      max_dist = distance
    
  return max_dist
     

if __name__ == "__main__":
  # Example usage:
  predicted_trajectory = np.array([[0, 0], [1, 1], [2, 2]])
  ground_truth_trajectory = np.array([[0, 0], [1, 2], [3, 3]])

  fde = final_displacement_error(predicted_trajectory, ground_truth_trajectory)
  ade = average_displacement_error(predicted_trajectory, ground_truth_trajectory)
  maxdist = calculate_max_dist(predicted_trajectory, ground_truth_trajectory)

  print("Final Displacement Error (FDE): ", fde)
  print("Average Displacement Error (ADE): ", ade)
  print("MaxDist: ", maxdist)

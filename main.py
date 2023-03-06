from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np


ATSQA_scores = [65, 67, 75, 80, 75, 56, 40, 63, 61, 72, 76, 85, 69, 80, 55, 52, 74, 64, 79, 78, 45, 89, 80, 58, 38, 70, 97, 80, 73, 35];
TQM_scores = [2.17, 2.33, 3.0, 4.17, 2.5, 2.33, 3.33, 2.17, 2.0, 2.5, 3.17, 4.0, 4.17, 4.0, 2.5, 3.33, 3.0, 3.17, 2.67, 3.17, 2.17, 1.83, 1.67, 3.17, 1.83, 3.17, 3.0, 1.83, 2.0, 4.5];
data = list(zip(ATSQA_scores,TQM_scores));
# convert to numpy array for convenience
data = np.array(data)

# compute mean and standard deviation of each dimension
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)

# normalize the data by subtracting mean and dividing by std
normalized_data = (data - means) / stds

# Calculate the distance to the fourth-nearest neighbour for each point
k = 4
nbrs = NearestNeighbors(n_neighbors=k).fit(normalized_data)
distances, indices = nbrs.kneighbors(normalized_data)
dist_to_k = distances[:, -1]

# Sort the distances in ascending order
sorted_distances = np.sort(dist_to_k)

# Plot the sorted distances
fig = plt.figure()
plt.plot(sorted_distances)
plt.xlabel("points")
plt.ylabel("4th-distance")
plt.show()

# Run DBSCAN
# eps=0.830
dbscan = DBSCAN(eps=0.830, min_samples=4).fit(normalized_data)
labels = dbscan.labels_
n_outliers = np.sum(labels == -1)
print(f"Number of outliers: {n_outliers}");
# Plot the points coloured on the result
# Outliers: Red, Border-Points: Green, Core-Points: Blue
colors = np.array(['r', 'b', 'g'])
colors_labels = colors[labels+1]
plt.scatter(data[:, 0], data[:, 1], c=colors_labels)
plt.xlabel("ATSQA Maintainability Score")
plt.ylabel("TQM Maintainability Stars")
# plt.plot([37, 100], [2, 4.33], color='black')
plt.show()


# def plotNormalizedData
# # Extract x and y coordinates from the points
# x = [p[0] for p in normalized_data]
# y = [p[1] for p in normalized_data]
#
# # Create a scatter plot of the points
# plt.scatter(x, y)
#
# # Set the plot title and axis labels
# plt.title("Array of points")
# plt.xlabel("x")
# plt.ylabel("y")
#
# # Display the plot
# plt.show()


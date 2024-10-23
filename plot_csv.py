import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bulged_line(start, stop, bulge):
    # Midpoint
    mid_x = (start[0] + stop[0]) / 2
    mid_y = (start[1] + stop[1]) / 2
    
    # Adjust midpoint according to bulge value
    control_x = mid_x + bulge * (stop[1] - start[1])
    control_y = mid_y - bulge * (stop[0] - start[0])

    # Generate Bezier curve points
    t = np.linspace(0, 1, 100)
    x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * stop[0]
    y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * stop[1]
    
    # Plot the curve
    plt.plot(x, y, 'g-')  # 'g-' for green line

    # Plot start and end points of the bulge
    plt.plot(start[0], start[1], 'bo', markersize=8)  # 'bo' for blue start point
    plt.plot(stop[0], stop[1], 'ro', markersize=8)  # 'ro' for red end point

# Reading the CSV file
data = pd.read_csv("C:\\Users\\benny\\OneDrive\\Desktop\\code\\input.csv")

# Extracting the necessary columns
names = data['Name']
northing = data['Northing']
easting = data['Easting']
types = data['Type']
bulges = data['Bulge']

# Plotting the points and lines
plt.figure(figsize=(10, 10))

# Initialize min and max values for x and y
min_x, max_x = np.min(easting), np.max(easting)
min_y, max_y = np.min(northing), np.max(northing)

# Calculate the range and apply it to both axes
data_range = max(max_x - min_x, max_y - min_y)
mid_x = (min_x + max_x) / 2
mid_y = (min_y + max_y) / 2

plt.xlim(mid_x - data_range / 2, mid_x + data_range / 2)
plt.ylim(mid_y - data_range / 2, mid_y + data_range / 2)

i = 0
while i < len(types):
    if types[i] == 2:
        start_point = (easting[i], northing[i])
        bulge_value = bulges[i]
        
        # Find the first Type 3 after this Type 2
        for j in range(i + 1, len(types)):
            if types[j] == 3:
                stop_point = (easting[j], northing[j])
                # Plot the line or curve between the start and stop points
                if bulge_value != 0:
                    plot_bulged_line(start_point, stop_point, bulge_value)
                else:
                    plt.plot([start_point[0], stop_point[0]], [start_point[1], stop_point[1]], 'k-')
                    plt.plot([start_point[0], stop_point[0]], [start_point[1], stop_point[1]], 'ro', markersize=3)
                # Move the index to the next position after the found stop point
                i = j
                break
    
    i += 1

# Setting plot labels and title
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.title('Plot of Points with Curved Lines (Bulges)')

# Show grid
plt.grid(True)

# Display the plot
plt.show()

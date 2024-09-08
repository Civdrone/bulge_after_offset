import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


csv_file = 'C:\\Users\\benny\\OneDrive\\Desktop\\the_ultimate_offset_test.csv'
output_csv_file = "C:\\Users\\benny\\OneDrive\\Desktop\\code\\corrected_1.csv"


# Reading the first CSV file
data_1 = pd.read_csv(output_csv_file)
data_2 = pd.read_csv(csv_file)


def plot_bulged_line(start, stop, bulge, color='k'):
    """Plot a line with bulge between start and stop points."""
    # Midpoint between start and stop
    mid_x = (start[0] + stop[0]) / 2
    mid_y = (start[1] + stop[1]) / 2

    # Control point based on bulge value
    control_x = mid_x + bulge * (stop[1] - start[1])
    control_y = mid_y - bulge * (stop[0] - start[0])

    # Generate Bezier curve points
    t = np.linspace(0, 1, 100)  # Increase number of points for smoothness
    x = (1 - t)**2 * start[0] + 2 * (1 - t) * t * control_x + t**2 * stop[0]
    y = (1 - t)**2 * start[1] + 2 * (1 - t) * t * control_y + t**2 * stop[1]
    
    # Plot the curve
    plt.plot(x, y, color+'-')  # '-' for solid line
    plt.plot(x, y, color+'o', markersize=3)  # 'o' for points, with small size






# Extracting the necessary columns from the first file
names_1 = data_1['Name']
northing_1 = data_1['Northing']
easting_1 = data_1['Easting']
types_1 = data_1['Type']
bulges_1 = data_1['Bulge']

# Reading the second CSV file

# Extracting the necessary columns from the second file
names_2 = data_2['Name']
northing_2 = data_2['Northing']
easting_2 = data_2['Easting']
types_2 = data_2['Type']
bulges_2 = data_2['Bulge']

# Plotting the points and lines
plt.figure(figsize=(10, 10))

# Initialize min and max values for x and y for both datasets
min_x = min(np.min(easting_1), np.min(easting_2))
max_x = max(np.max(easting_1), np.max(easting_2))
min_y = min(np.min(northing_1), np.min(northing_2))
max_y = max(np.max(northing_1), np.max(northing_2))

# Calculate the range and apply it to both axes
data_range = max(max_x - min_x, max_y - min_y)
mid_x = (min_x + max_x) / 2
mid_y = (min_y + max_y) / 2

plt.xlim(mid_x - data_range / 2, mid_x + data_range / 2)
plt.ylim(mid_y - data_range / 2, mid_y + data_range / 2)

# Plotting the first dataset (Corrected)
i = 0
while i < len(types_1):
    if types_1[i] == 2:
        start_point = (easting_1[i], northing_1[i])
        bulge_value = bulges_1[i]
        
        # Find the first Type 3 after this Type 2
        for j in range(i + 1, len(types_1)):
            if types_1[j] == 3:
                stop_point = (easting_1[j], northing_1[j])
                # Plot the line or curve between the start and stop points
                plot_bulged_line(start_point, stop_point, bulge_value, color='k')
                # Move the index to the next position after the found stop point
                i = j
                break
    
    i += 1

# Plotting the second dataset (Offset Only)
i = 0
while i < len(types_2):
    if types_2[i] == 2:
        start_point = (easting_2[i], northing_2[i])
        bulge_value = bulges_2[i]
        
        # Find the first Type 3 after this Type 2
        for j in range(i + 1, len(types_2)):
            if types_2[j] == 3:
                stop_point = (easting_2[j], northing_2[j])
                # Plot the line or curve between the start and stop points
                plot_bulged_line(start_point, stop_point, bulge_value, color='r')
                # Move the index to the next position after the found stop point
                i = j
                break
    
    i += 1

# Setting plot labels and title
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.title('Plot of Points with Curved Lines (Bulges) from Two CSV Files')

# Manually add legend entries
plt.plot([], [], 'ko', label='Corrected')
plt.plot([], [], 'ro', label='Offset Only')

# Show grid
plt.grid(True)

# Display legend
plt.legend()

# Display the plot
plt.show()





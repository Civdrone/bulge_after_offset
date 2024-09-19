import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_delta_distances(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Ensure that 'Northing', 'Easting', and 'Type' columns are numeric
    df['Northing'] = pd.to_numeric(df['Northing'], errors='coerce')
    df['Easting'] = pd.to_numeric(df['Easting'], errors='coerce')
    df['Type'] = pd.to_numeric(df['Type'], errors='coerce')
    
    # Drop any rows with NaN in 'Northing', 'Easting', or 'Type'
    df = df.dropna(subset=['Northing', 'Easting', 'Type']).reset_index(drop=True)
    
    # Filter rows where 'Type' == 2
    df_type2 = df[df['Type'] == 2].reset_index(drop=True)
    
    # Compute delta distances between consecutive Type 2 points
    delta_northing = df_type2['Northing'].diff()
    delta_easting = df_type2['Easting'].diff()
    delta_distances = np.sqrt(delta_northing**2 + delta_easting**2)
    
    # Remove the first NaN value resulting from the diff
    delta_distances = delta_distances.iloc[1:]
    df_type2 = df_type2.iloc[1:]  # Align the DataFrame with delta_distances
    
    # Filter delta distances smaller than 1 meter
    mask = delta_distances < 100
    delta_distances_filtered = delta_distances[mask]
    df_filtered = df_type2[mask].reset_index(drop=True)
    
    # Print delta distances between Type 2 points smaller than 1 meter
    print("Delta distances between consecutive Type 2 points (less than 1 meter):")
    for i, (dist, idx) in enumerate(zip(delta_distances_filtered, df_filtered.index), start=1):
        point_name_from = df_type2.iloc[idx]['Name']
        point_name_to = df_type2.iloc[idx + 1]['Name']
        print(f"{point_name_from} to {point_name_to}: {dist}")
    
    # Plot the delta distances smaller than 1
    plt.figure(figsize=(10, 6))
    plt.plot(delta_distances_filtered.values, marker='o')
    plt.title('Delta Distances Between Consecutive Type 2 Points (less than 1 meter)')
    plt.xlabel('Point Index')
    plt.ylabel('Delta Distance (meters)')
    plt.grid(True)
    plt.show()

# Example usage:
csv_file_path = 'C:\\Users\\benny\\OneDrive\\Desktop\\code\\non_Dash_final_test.csv'

plot_delta_distances(csv_file_path)

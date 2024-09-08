import numpy as np

def calculate_circle_center_and_radius(start, end, bulge):
    """
    Calculate the center and radius of a circle given a start point, end point, and bulge.
    
    Parameters:
    start (tuple): A tuple representing the (x, y) coordinates of the start point.
    end (tuple): A tuple representing the (x, y) coordinates of the end point.
    bulge (float): The bulge factor representing the curvature of the arc.
    
    Returns:
    center (tuple): A tuple containing the (x, y) coordinates of the center of the circle.
    radius (float): The radius of the circle.
    """

    # Calculate the midpoint between the start and end points
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Calculate the length of the chord (the distance between start and end points)
    chord_length = np.linalg.norm(np.array([end[0] - start[0], end[1] - start[1]]))

    # Calculate the radius using the bulge formula
    radius = chord_length / (2 * abs(bulge))

    # Calculate the angle between the start and end points
    angle_between_points = np.arctan2(end[1] - start[1], end[0] - start[0])

    # Calculate the bulge angle
    bulge_angle = np.arctan(bulge)
    half_pi = np.pi / 2

    # Adjust the angle to get the direction of the center
    adjusted_angle = angle_between_points + (half_pi - 2 * bulge_angle)

    # Calculate the coordinates of the circle's center
    center_x = mid_x + radius * bulge * np.sin(adjusted_angle)
    center_y = mid_y - radius * bulge * np.cos(adjusted_angle)

    return (center_x, center_y), radius

# Example usage:
start = (1.0, 1.0)
end = (4.0, 5.0)
bulge = 0.5

center, radius = calculate_circle_center_and_radius(start, end, bulge)
print(f"Circle Center: {center}, Radius: {radius}")

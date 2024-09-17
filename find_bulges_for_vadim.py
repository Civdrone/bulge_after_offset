import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TODO IMPORTANT!!
# there is one bug we need to talk about. right now the algorithm craetes false bulges / lines
# when one set of points ends and a new one starts  (for example: when there are 2 different non connected lines,
# when one line ends and the new line starts)
# i talked with Dani and he tolda me to talk about it with you Vadim because maybe i dont need to fix this
# because of the way you perform the offset operation.


######### dynamic threshold gains ######
K_DISTANCE = 0.3
K_HEADING = 0.3
BULGE_THRESHOLD = 0.5
######################################

###### blocking points gain #####
K_BLOCKING = 0.3
###############################

##### minimum bulge value ###########
MIN_BULGE = 1e-5
###################################


class Point:
    # a class to store the all points data for proccesing.
    def __init__(self, index=0, northing=0, easting=0, elevation = 0 ,type = 0 , interval = 0 , length = 0 , delta_heading=0, delta_distance=0, bulge=0, name = ""):
        self.index = index
        self.northing = northing
        self.easting = easting
        self.elevation = elevation
        self.type = type
        self.interval = interval
        self.length = length
        self.bulge = bulge
        self.name = name
        self.delta_heading = delta_heading
        self.delta_distance = delta_distance


    def to_coordinates_metadata_dict(self):
        return {
            'coordinates': {
                'latitude': float(self.northing),
                'longitude': float(self.easting),
                'altitude': float(self.elevation),
            },
            'metadata': {
                'name': str(self.name),
                'marking_type': int(self.type),
                'interval': float(self.interval),
                'dash_length': float(self.length),
                'bulge': float(self.bulge),
            }
        }


    def __str__(self):
        """Return a string representation of the Point for easy reading."""
        return (f"Index: {self.index}, Northing: {self.northing}, Easting: {self.easting}, "
                f"Delta Heading: {self.delta_heading}, Delta Distance: {self.delta_distance}, Bulge: {self.bulge}")


class PathAnalyzer:

    def __init__(self, csv_file, output_csv_file):
        self.data = pd.read_csv(csv_file)
        self.output_csv_file = output_csv_file
        # To store the new data including bulges
        self.new_data = pd.DataFrame(columns=self.data.columns)
        self.points = []  # To store points as objects of Point class
        self.processed_points = []  # To store points after processing
        # To store the end index of the last detected bulge (in order to skip mid bulge points thus reducing complexity in the code)
        self.bulge_end_index = -1
        self.bulge_errors = []  # To store the errors for each bulge
        # To store points that are blocking due to delta distance condition
        self.blocking_points = []

  ################################################# math functions ###################################################

    def calculate_heading(self, point1, point2):
        """Calculate the heading (angle) between two points."""
        return np.arctan2(point2.northing - point1.northing, point2.easting - point1.easting)

    def calculate_delta_distance(self, point1, point2):
        """Calculate the delta distance between two points."""
        return np.linalg.norm(np.array([point2.easting, point2.northing]) - np.array([point1.easting, point1.northing]))

    def calculate_delta_heading(self, current_point, next_point, previous_heading=None):
        """Calculate the change in heading between two points."""

        # Calculate the current heading between the current point and the next point
        current_heading = self.calculate_heading(current_point, next_point)

        if previous_heading is not None:
            # Calculate the delta heading as the difference between the current and previous heading
            delta_heading = current_heading - previous_heading
            # Normalize the delta heading to the range -pi to pi
            delta_heading = (delta_heading + np.pi) % (2 * np.pi) - np.pi
        else:
            delta_heading = 0

        return delta_heading

    def calculate_bulge_center(self, start_point, end_point, bulge):
        """Calculate the center of the arc described by the given bulge and vertices."""
        angle_between_points = np.arctan2(
            end_point.northing - start_point.northing, end_point.easting - start_point.easting)
        half_pi = np.pi / 2
        bulge_angle = np.arctan(bulge)

        # Calculate the center of the arc
        # Calculate start point coordinates
        start_x = start_point.easting
        start_y = start_point.northing

        # Calculate angle between points and adjust using half_pi and bulge_angle
        adjusted_angle = angle_between_points + (half_pi - 2 * bulge_angle)

        # Calculate the bulge radius
        bulge_radius = self.calculate_bulge_radius(
            start_point, end_point, bulge)

        # Multiply the radius by the bulge
        adjusted_radius = bulge_radius * bulge

        # Use polar coordinates to find the center coordinates
        center_x, center_y = self.polar(
            (start_x, start_y), adjusted_angle, adjusted_radius)

        return np.array([center_x, center_y])

    def calculate_bulge_radius(self, start_point, end_point, bulge):
        """Calculate the radius of the arc described by the given bulge and vertices."""
        return np.linalg.norm([end_point.easting - start_point.easting, end_point.northing - start_point.northing]) / (2 * np.abs(bulge))

    def polar(self, p, angle, distance):
        """Utility function to calculate polar coordinates."""
        return (p[0] + distance * np.cos(angle), p[1] + distance * np.sin(angle))

    def distance_to_bulge_arc(self, point, bulge, start_index, end_index):
        """Calculate the distance from a point to the closest point on the bulge arc."""
        start_point = self.points[start_index]
        end_point = self.points[end_index]

        # Calculate the radius and center of the arc
        arc_radius = self.calculate_bulge_radius(start_point, end_point, bulge)
        arc_center = self.calculate_bulge_center(start_point, end_point, bulge)

        # Convert point to numpy array for easy calculation
        point_location = np.array([point.easting, point.northing])

        # Calculate the distance from the point to the center of the circle
        distance_to_center = np.linalg.norm(point_location - arc_center)

        # Calculate the error as the absolute difference between the point-to-center distance and the arc radius
        distance_to_arc = np.abs(distance_to_center - arc_radius)

        return distance_to_arc

################################################### End math functions ############################################################################


################################################## utility functions ########################################################


    def find_middle_index(self, start_index, end_index):
        """Find the middle index for a given start and end index.
        Handles both odd and even cases by interpolating a new middle point if necessary.
        Returns the middle index.
        """
        num_points_in_bulge = end_index - start_index + 1

        if num_points_in_bulge % 2 == 1:
            # Odd number of points, use the usual middle index calculation
            middle_index = start_index + (end_index - start_index) // 2
        else:
            # Even number of points, calculate the middle point between the two middle indices
            left_middle_index = start_index + (end_index - start_index) // 2
            right_middle_index = left_middle_index + 1
            middle_point = self.create_interpolated_point(
                self.points[left_middle_index], self.points[right_middle_index])
            # Insert the new middle point into the points list
            self.points.insert(right_middle_index, middle_point)
            # Set the middle index to the position of the inserted point
            middle_index = right_middle_index

        return middle_index

    def create_interpolated_point(self, point1, point2):
        """Create a new interpolated point between two given points - in case the number of points forming the circle is pair (no middle point)."""
        # print("creating new middle point")
        new_index = (point1.index + point2.index) / \
            2  # New index between the two points
        new_northing = (point1.northing + point2.northing) / 2
        new_easting = (point1.easting + point2.easting) / 2
        new_delta_heading = (point1.delta_heading + point2.delta_heading) / 2
        new_delta_distance = (point1.delta_distance +
                              point2.delta_distance) / 2

        new_middle_point = Point()
        new_middle_point.index = new_index
        new_middle_point.northing = new_northing
        new_middle_point.easting = new_easting
        new_middle_point.delta_heading = new_delta_heading
        new_middle_point.delta_distance = new_delta_distance

        return new_middle_point

    def is_below_threshold(self, total_sum, threshold):
        """Check if the total sum is below the threshold."""
        return total_sum < threshold

    def calculate_dynamic_threshold(self, start_index, end_index):
        """Calculate the dynamic threshold based on the number of points considered,
        the delta heading and delta distance between the inspected points.
        from experiments i saw that smaller curves (smaller delta distance and higher delta heading between points) have higher errors,
        using this dynamic threshold we adjust the threshold to best represnt the inspected curve

        IMPORTANT:
        i am pretty sure that we can increase the robustness of the system
        by adjusting the threshold gains some more """

        num_points_in_curve = end_index - start_index + 1

        # Calculate the averages
        # Absolute value for delta heading
        avg_heading = np.mean([abs(p.delta_heading)
                              for p in self.points[start_index:end_index + 1]])
        avg_distance = np.mean(
            [p.delta_distance for p in self.points[start_index:end_index + 1]])

        # Compute the threshold unit constant - the threhosld per point, afterwards we multiply this by the number of points
        # more points results in higher errors so we account for that
        threshold_unit_constant = BULGE_THRESHOLD + K_DISTANCE * \
            (1 / avg_distance) + K_HEADING * avg_heading

        # print(f'threshold unit constant = {threshold_unit_constant}')

        # Compute the total threshold
        total_threshold = num_points_in_curve * threshold_unit_constant

        return total_threshold

    def calculate_bulge(self, start_index, middle_index, end_index):
        """Calculate the bulge based on the start, middle, and end indices.
        Limit the bulge values to the range -1 to 1.
        """

        start_point = self.points[start_index]
        middle_point = self.points[middle_index]
        end_point = self.points[end_index]

        # First, calculate the bulge using the angle-based approach
        def angle(p1, p2):
            """Calculate the angle between two points."""
            return np.arctan2(p2.northing - p1.northing, p2.easting - p1.easting)

        angle1 = angle(middle_point, start_point)
        angle2 = angle(middle_point, end_point)

        # Calculate the half-angle
        a = (angle2 - angle1 + np.pi) / 2

        # Calculate the correct bulge value
        bulge = np.sin(a) / np.cos(a)

        # Now, determine the correct direction using the cross product method
        start_to_end_dist = np.linalg.norm(
            [end_point.easting - start_point.easting, end_point.northing - start_point.northing])
        start_to_middle_dist = np.linalg.norm(
            [middle_point.easting - start_point.easting, middle_point.northing - start_point.northing])
        middle_to_end_dist = np.linalg.norm(
            [end_point.easting - middle_point.easting, end_point.northing - middle_point.northing])

        if start_to_middle_dist == 0 or middle_to_end_dist == 0:
            return 0  # Avoid division by zero

        # Use cosine rule to find the angle at the middle point
        cos_theta = (start_to_middle_dist**2 + middle_to_end_dist**2 -
                     start_to_end_dist**2) / (2 * start_to_middle_dist * middle_to_end_dist)

        # Clamp cos_theta to the valid range [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        direction_bulge = np.tan(theta / 4)

        # Determine the direction of the bulge
        cross_product_z = np.cross([middle_point.easting - start_point.easting, middle_point.northing - start_point.northing],
                                   [end_point.easting - start_point.easting, end_point.northing - start_point.northing])

        if cross_product_z < 0:
            bulge = -abs(bulge)
        else:
            bulge = abs(bulge)

        # Clip the bulge to the range -1 to 1
        bulge = np.clip(bulge, -1.0, 1.0)
        
        #zero for smaller than threshold bulge
        if abs(bulge) <= MIN_BULGE:
            bulge = 0

        # Set the bulge value in the corresponding points
        self.points[start_index].bulge = bulge
        self.points[end_index].bulge = 0

        return bulge

    def normalize_segments(self, start_index, num_points):
        """Normalize delta headings and delta distances for a specified segment.
        we normalize all the potential curve points. because curve points act similar (similar delta distances and delta headings)
        the normalization output of curve points results in very small values per each point
        non curve point will output with a big value after the normalization which will result in higher total error"""
        segment = self.points[start_index:start_index + num_points]

        # Calculate the averages
        total_heading = 0
        total_distance = 0
        for p in segment:
            total_heading += p.delta_heading
            total_distance += p.delta_distance

        avg_heading = total_heading / num_points
        avg_distance = total_distance / num_points

        # Normalize the headings and distances
        normalized_headings = []
        normalized_distances = []

        for p in segment:
            heading_normal = p.delta_heading - avg_heading
            distance_normal = p.delta_distance - avg_distance
            normalized_headings.append(heading_normal)
            normalized_distances.append(distance_normal)

        return normalized_headings, normalized_distances

    def calculate_goodness_of_fit(self, start_index, end_index, bulge_value):
        """Calculate the sum of distances between points and the bulge arc - the error."""
        total_distance = 0
        # summing all the distances from the points to the arc in order to receive final erro value
        for i in range(start_index, end_index + 1):
            point = self.points[i]
            distance = self.distance_to_bulge_arc(
                point, bulge_value, start_index, end_index)
            total_distance += distance
        return total_distance

    def process_potential_curve(self, start_index, end_index):
        """Process a potential curve and return the total sum of errors. - being called after the normalization process"""
        normalized_headings, normalized_distances = self.normalize_segments(
            start_index, end_index - start_index + 1)

        sum_delta_heading = sum(abs(h) for h in normalized_headings)
        sum_delta_distance = sum(abs(d) for d in normalized_distances)
        total_sum = sum_delta_heading + sum_delta_distance

        return total_sum


###################################################### End utility functions ###########################################################################


##################################################### pre proccessing functions ##############################################


    def if_blocking_append(self, current_point, previous_point):
        """
        Check if either the current point or the previous point is a blocking point based on dynamically adjusted relative differences.
        A point is considered blocking if the relative difference between the delta distances exceeds the dynamically adjusted K_BLOCKING.
        """
        if previous_point is not None:
            # Calculate the maximum of the two delta distances to use as the denominator for relative difference
            max_delta_distance = max(current_point.delta_distance, previous_point.delta_distance)

            # Avoid division by zero
            if max_delta_distance == 0:
                return



            # Calculate the relative difference in percentage
            relative_difference = abs(current_point.delta_distance - previous_point.delta_distance) / max_delta_distance

            # Debug print to show delta distances, relative difference, and the dynamically adjusted threshold
            # print(f"Previous Point Index: {previous_point.index}, Delta Distance: {previous_point.delta_distance}")
            # print(f"Current Point Index: {current_point.index}, Delta Distance: {current_point.delta_distance}")

            # Check if the relative difference exceeds the dynamically adjusted blocking threshold
            if relative_difference > K_BLOCKING:
                # Append the point with the larger delta distance to the blocking points list
                if current_point.delta_distance > previous_point.delta_distance:
                    self.blocking_points.append(current_point)
                    print(
                        f"Blocking point detected at Index: {current_point.index}, Delta Distance: {current_point.delta_distance}")
                else:
                    self.blocking_points.append(previous_point)
                    print(
                        f"Blocking point detected at Index: {previous_point.index}, Delta Distance: {previous_point.delta_distance}")

    def check_mismatched_start_stop(self, current_start, previous_stop):
        """
        Check if the previous point is a stop point (Type == 3) and the current point is a start point (Type == 2) at different coordinates.
        If the coordinates differ, append the current start point to the blocking points list.
        """
        if previous_stop is not None and self.data.iloc[previous_stop.index]['Type'] == 3:
            # Check if the coordinates of the start point differ from the stop point
            if current_start.northing != previous_stop.northing or current_start.easting != previous_stop.easting:
                self.blocking_points.append(current_start)
                # Optionally print debug information
                print(
                    f"Mismatched start point at index {current_start.index}, added to blocking points")

    def analyze_last_point(self):
        if self.data.iloc[-1]['Type'] == 3:
            final_point = Point(
            index=len(self.data) - 1, 
            northing=self.data.iloc[-1]['Northing'], 
            easting=self.data.iloc[-1]['Easting'], 
            elevation=self.data.iloc[-1]['Elevation'], 
            type=self.data.iloc[-1]['Type'], 
            interval=self.data.iloc[-1]['Interval'], 
            length=self.data.iloc[-1]['Length'], 
            bulge=self.data.iloc[-1]['Bulge'], 
            name=self.data.iloc[-1]['Name'].split('__')[0]
            )

            # Set final point deltas to 0 since it's a stop point
            final_point.delta_distance = 0
            final_point.delta_heading = 0
            # Append the final stop point to the processed points
            self.points.append(final_point)

    def analyze_start_points(self):
        """Prepare data for processing: Calculate delta heading and delta distance for each start point."""
        previous_heading = None  # The first point has no prior point to compare
        previous_point = None  # To store the previous point for distance comparison
        # To store the previous stop point to check for mismatched start/stop points
        previous_stop_point = None

        for i in range(len(self.data) - 1):
            if self.data.iloc[i]['Type'] == 2:  # We process only the start points

                # Insert data from the CSV into the Point object:
                current_point = Point(
                    index=i, 
                    northing=self.data.iloc[i]['Northing'], 
                    easting=self.data.iloc[i]['Easting'], 
                    elevation=self.data.iloc[i]['Elevation'], 
                    type=self.data.iloc[i]['Type'], 
                    interval=self.data.iloc[i]['Interval'], 
                    length=self.data.iloc[i]['Length'], 
                    bulge=self.data.iloc[i]['Bulge'], 
                    name=self.data.iloc[i]['Name'].split('__')[0]
                )

                next_point = Point(
                    index=i + 1, 
                    northing=self.data.iloc[i + 1]['Northing'], 
                    easting=self.data.iloc[i + 1]['Easting'], 
                    elevation=self.data.iloc[i + 1]['Elevation'], 
                    type=self.data.iloc[i + 1]['Type'], 
                    interval=self.data.iloc[i + 1]['Interval'], 
                    length=self.data.iloc[i + 1]['Length'], 
                    bulge=self.data.iloc[i + 1]['Bulge'], 
                    name=self.data.iloc[i + 1]['Name']
                )
                
                # Calculate the Euclidean distance between these points
                current_point.delta_distance = self.calculate_delta_distance(current_point, next_point)

                # Calculate the delta heading between current and next points
                current_point.delta_heading = self.calculate_delta_heading(
                    current_point, next_point, previous_heading)
                # Update previous_heading for the next iteration
                previous_heading = self.calculate_heading(
                    current_point, next_point)

                # Check if the current point is a blocking point
                self.if_blocking_append(current_point, previous_point)

                previous_point = current_point  # Store the current point for the next comparison

                # Check for mismatched start and stop points and append to blocking points
                self.check_mismatched_start_stop(
                    current_point, previous_stop_point)

                # append the point with all the required data in the same order as the CSV file
                self.points.append(current_point)

            elif self.data.iloc[i]['Type'] == 3:
                # Update the previous stop point for the next iteration
                previous_stop_point = Point(
                    index=i, northing=self.data.iloc[i]['Northing'], easting=self.data.iloc[i]['Easting'])

        # a special case. generally we take into account only the start points and ignore the stop points.
        # at the final stop point there will not be also a start point so we also add the final stop point to our points list
        self.analyze_last_point()

########################################################### pre proccessing functions ##########################################################################


############################################################## main functions ######################################################


    def find_max_curve(self, start_index):
        """Find the maximum curve starting from a given index.
        This function tries to find the curve that consists of the most points.
        """

        # The condition that tells us if we need to keep looking for the max curve or not.
        max_curve_found = False
        # Adjust the final index to be within the bounds of the points list.
        final_index = len(self.points) - 1
        # A bulge will have at least 3 points (start index + 2 points).
        index_counter = 2
        # Limiting the max points based on the total number of points.
        max_points = final_index - start_index

        while index_counter <= max_points:  # Keep running until all points are processed.
            end_index = start_index + index_counter

            # Ensure that end_index is within bounds
            if end_index > final_index:
                print(
                    f"end_index {end_index} is out of bounds. Stopping curve detection.")
                break

            # Calculate a score for points in between the start and end index.
            total_sum = self.process_potential_curve(start_index, end_index)
            # Calculate the threshold to determine if the points are a bulge.
            threshold = self.calculate_dynamic_threshold(
                start_index, end_index)

            # Debugging output to trace the process
            # print(f"Checking potential curve from index {start_index} to {end_index}...")
            # print(f"Total sum: {total_sum:.6f}, Threshold: {threshold:.6f}")

            # Checking if bulge conditions are met
            if self.is_below_threshold(total_sum, threshold):
                # print(f"Extending curve at index {end_index}...")

                # Stop extending the curve if a blocking point is reached.
                if self.points[end_index] in self.blocking_points:
                    # print(f"Reached blocking point at index {end_index}. Stopping curve detection.")
                    break

                index_counter += 1
                max_curve_found = True

            else:  # The curve conditions are not met anymore
                # print(f"Stopping extension at index {end_index}, total sum {total_sum} exceeds threshold {threshold}")
                break

        # If a valid curve was found, process the curve
        if max_curve_found:
            return self.process_curve_found(start_index, end_index, final_index)
        else:
            return None


    def process_curve_found(self, start_index, end_index, final_index):
        """Process the curve found by extending the indices and calculating the bulge.
        this function returns the final start index , end index, and bulge of the curve"""

        # Go back one step to the last valid curve
        end_index -= 1

        # Extend the curve by one index backward and forward, ensuring no extension into blocking points
        # (we do this because the start and end indexes "score" are affected by the non curve points next to them and destroying their scores)
        if start_index > 0 and self.points[start_index - 1] not in self.blocking_points:
            start_index -= 1
        if end_index < final_index and self.points[end_index + 1] not in self.blocking_points:
            end_index += 1

        # finding the middle index, we need it for the bulge calculation
        middle_index = self.find_middle_index(start_index, end_index)

        # calculating the bulge based on the start middle and end indexes
        bulge_value = self.optimize_bulge(start_index, middle_index, end_index)

        # Check the goodness of fit for the optimized bulge
        goodness_of_fit = self.calculate_goodness_of_fit(
            start_index, end_index, bulge_value)
        print(
            f"Goodness of fit for optimized bulge from index {start_index} to {end_index}: {goodness_of_fit:.6f}")

        return start_index, end_index, bulge_value


    def process_point(self, start_index):
        """Process a single point or curve and update the processed points list."""
        # print(f"Processing point at index {start_index}...")  # Debugging output
        curve_info = self.find_max_curve(start_index)

        if curve_info:
            start_index, end_index, bulge_value = curve_info
            # print(f"Bulge detected from {start_index} to {end_index} with bulge value {bulge_value:.6f}")
            self.processed_points.append(self.points[start_index])
            self.processed_points.append(self.points[end_index])
            self.bulge_end_index = end_index
            # Skip all the points between the start and stop bulge points.
            return end_index + 1

        else:
            # print(f"No bulge detected at {start_index}. Moving to the next point.")
            self.processed_points.append(self.points[start_index])
            return start_index + 1  # no bulge is found from this point, moving the the point after it


    def iterate_and_normalize_all_segments(self):
        """this is the main function. goes through all the points in the list of proccesed points and detects curves
        Iterate over all segments, normalize them, and detect bulges based on the threshold."""

        start_index = 0
        final_index = len(self.points)

        while start_index < final_index:
            # Skip points already part of a detected bulge - in order to decrease complexity
            if start_index <= self.bulge_end_index:
                # print(f"Skipping point at index {start_index} because it's already part of a detected bulge.")
                start_index = self.bulge_end_index - 1
                continue

            # Check if the current index is the start of a bulge
            start_index = self.process_point(start_index)
            
        self.final_comprasion_input_output()
        self.adding_stop_points_to_list()
            
    def final_comprasion_input_output(self):
        """
        Perform a final comparison between the `points` list (original data) and the `processed_points` list
        (processed data). Ensure no points are skipped or missing in the `processed_points` list.
        """
        # Compare the first point in processed_points and points
        first_processed_point = self.processed_points[0]

        # Search for the first processed point in the points list
        found_first_index = None
        for i, point in enumerate(self.points):
            if (point.northing == first_processed_point.northing and
                point.easting == first_processed_point.easting):
                found_first_index = i
                break

        if found_first_index is not None and found_first_index > 0:
            # Add all points with lower index than the found first index from original points to processed points
            for i in range(0, found_first_index):
                self.processed_points.insert(i, self.points[i])
            # print(f"Copied missing points before index {found_first_index} to processed_points.")
        
        # Compare the last point in processed_points and points
        last_processed_point = self.processed_points[-1]

        # Search for the last processed point in the points list
        found_last_index = None
        for i, point in enumerate(self.points):
            if (point.northing == last_processed_point.northing and
                point.easting == last_processed_point.easting):
                found_last_index = i
                break

        if found_last_index is not None:
            # Add all points with higher index than the found last index from original points to processed points
            for i in range(found_last_index + 1, len(self.points)):
                self.processed_points.append(self.points[i])
            # print(f"Copied missing points after index {found_last_index} to processed_points.")
        
        # Debug print to confirm all points have been processed
        # print(f"Final check complete. Total points processed: {len(self.processed_points)}")
        
    def adding_stop_points_to_list(self):
        new_processed_points = []  # creating a new list to store the new data including stop points
        index_counter = 1  # re indexing the list 

        for i in range(len(self.processed_points) - 1):
            point = self.processed_points[i]
            next_point = self.processed_points[i + 1]

            # Rebuild the name for the start point
            point_name = f"{point.name}__{index_counter}_start"
            if point.bulge != 0:
                point_name += "_bulge"
            
            point.index = index_counter
            point.name = point_name

            
            

            if point.type == 2:  # If it's a start point
                
                new_processed_points.append(point)
                # Create a new stop point based on the next start point
                stop_point = Point(
                    index=index_counter,  # Use the same index as the start point
                    northing=next_point.northing,
                    easting=next_point.easting,
                    elevation=next_point.elevation,
                    type=3,  # Set as stop point
                    interval=next_point.interval,
                    length=next_point.length,
                    bulge=0.0,  # Stop points have zero bulge
                    name=f"{point.name.split('__')[0]}__{index_counter}_stop"
                )

                new_processed_points.append(stop_point)

                index_counter += 1

        self.processed_points = new_processed_points




                


    def optimize_bulge(self, start_index, middle_index, end_index):
        """because it looks like the offset operation is not very accurate, we start by computing the initial bulge
        using a formula taking as input the start middle and end points of the bulge and afterwards
        Optimizing the bulge value to minimize the error between the points creating the bulge and the bulge itself."""

        best_bulge = None
        start_error = None
        best_error = float('inf')
        # craeting an initial guess for the bulge
        initial_bulge = self.calculate_bulge(
            start_index, middle_index, end_index)

        # stupid iterative way of optimizing the bulge.
        # this part is simplified because i dont know if there is a js alternative to scipy.optimize
        # TODO Vadim you are welcome to give your advice. the complexity in here is not the best as you can see :)

        # Adjust the range and step size as needed
        for delta in np.linspace(-0.1, 0.1, num=200):
            bulge_candidate = initial_bulge + delta
            error = self.calculate_goodness_of_fit(
                start_index, end_index, bulge_candidate)
            if start_error == None:
                start_error = error
                print(f"Error before optimization: {error}")
            if error < best_error:
                best_error = error
                best_bulge = bulge_candidate

        return best_bulge


############################################################## End of main functions ######################################################


############################################### debug functions - no need to copy to js #####################################################################

    def generate_csv_from_points(self):
        """Generate a CSV file from the processed points."""
        for i in range(len(self.processed_points)):
            point = self.processed_points[i]

            # Add stop point before start point (except for the first point)
            if i > 0:
                stop_row = self.data.iloc[point.index].copy()
                stop_row['Name'] = f"{point.index}_stop"
                stop_row['Type'] = 3
                stop_row['Bulge'] = self.processed_points[i - 1].bulge
                self.new_data = pd.concat(
                    [self.new_data, pd.DataFrame(stop_row).T], ignore_index=True)

            # Add the start point
            start_row = self.data.iloc[point.index].copy()
            start_row['Name'] = f"{point.index}_start_bulge" if point.bulge != 0 else f"{point.index}_start"
            start_row['Type'] = 2
            start_row['Bulge'] = point.bulge
            self.new_data = pd.concat(
                [self.new_data, pd.DataFrame(start_row).T], ignore_index=True)

        # Add the last stop point for the last point
        last_point = self.processed_points[-1]
        last_point_index = last_point.index

        # Check if the last point has a corresponding "stop" point in the original data
        if last_point_index < len(self.data) - 1 and self.data.iloc[last_point_index + 1]['Type'] == 3:
            stop_row = self.data.iloc[last_point_index + 1].copy()
            stop_row['Name'] = f"{last_point_index + 1}_stop"
            stop_row['Type'] = 3
            stop_row['Bulge'] = last_point.bulge
            self.new_data = pd.concat(
                [self.new_data, pd.DataFrame(stop_row).T], ignore_index=True)
        else:
            stop_row = self.data.iloc[last_point.index].copy()
            stop_row['Name'] = f"{last_point.index}_stop"
            stop_row['Type'] = 3
            stop_row['Bulge'] = last_point.bulge
            self.new_data = pd.concat(
                [self.new_data, pd.DataFrame(stop_row).T], ignore_index=True)

        # Save the new CSV file
        self.new_data.to_csv(self.output_csv_file, index=False)
        print(f"New CSV file generated: {self.output_csv_file}")

    # def print_processed_points(self):
    #     """Print all processed points in a readable manner."""
    #     print("\nProcessed Points:")
    #     for point in self.processed_points:
    #         print(point)
    #     print("\n")

    # def plot_errors_for_each_bulge(self):
    #     """Plot the errors for each bulge on separate graphs."""
    #     for i, errors in enumerate(self.bulge_errors, 1):
    #         indexes, error_values = zip(*errors)
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(indexes, error_values, marker='o',
    #                  linestyle='-', color='b')
    #         plt.xlabel('Index')
    #         plt.ylabel('Error')
    #         plt.title(f'Error per Index for Bulge {i}')
    #         plt.grid(True)
    #         plt.show()
    #         input(f"Press Enter to continue to the next bulge...")

    # def plot_data(self, delta_distances, delta_headings):
    #     """Plot delta distances and delta headings on separate subplots."""
    #     plt.figure(figsize=(12, 8))

    #     plt.subplot(2, 1, 1)
    #     plt.plot(delta_distances, 'bo-', label='Delta Distance')
    #     plt.xlabel('Point Number')
    #     plt.ylabel('Delta Distance')
    #     plt.title('Delta Distance Between Points')
    #     plt.grid(True)

    #     plt.subplot(2, 1, 2)
    #     plt.plot(delta_headings, 'ro-', label='Delta Heading')
    #     plt.xlabel('Point Number')
    #     plt.ylabel('Delta Heading (radians)')
    #     plt.title('Delta Heading Between Points')
    #     plt.grid(True)

    #     plt.tight_layout()
    #     plt.show()

############################################### End of debug functions #####################################################################


# Example usage:
# csv_file = '/home/bennyciv/git/python_tools/offset_mission/offset_test_ofek_code.csv' "C:\Users\benny\OneDrive\Desktop\smooth_offset_for_offseting_again.csv"
# csv_file = 'C:\\Users\\benny\\OneDrive\\Desktop\\the_second_offset.csv'
# output_csv_file = "C:\\Users\\benny\\OneDrive\\Desktop\\code\\corrected_1.csv"

# csv_file = '/Users/civrobotics/dev/git/bulge_after_offset/the_ultimate_offset_test.csv'
# output_csv_file = "/Users/civrobotics/dev/git/bulge_after_offset/output.csv"


# analyzer = PathAnalyzer(csv_file, output_csv_file)
# analyzer.analyze_start_points()  # Prepare data to analyze it
# analyzer.iterate_and_normalize_all_segments()  # Process and find the bulges
# analyzer.generate_csv_from_points()  # Save the optimized bulges to the CSV file


def find_bulges(csv_file, output_csv_file):
    analyzer = PathAnalyzer(csv_file, output_csv_file)
    analyzer.analyze_start_points()  # Prepare data to analyze it
    analyzer.iterate_and_normalize_all_segments()  # Process and find the bulges
    # Save the optimized bulges to the CSV file
    analyzer.generate_csv_from_points()

    return [point.to_coordinates_metadata_dict() for point in analyzer.processed_points]


if __name__ == "__main__":
    PATH_INPUT = 1

    try:
        root_dir = os.path.dirname(os.path.abspath(__file__))

        input_csv_file = sys.argv[PATH_INPUT]
        # input_csv_file = 'C:\\Users\\benny\\OneDrive\\Desktop\\code\\input.csv'

        output_csv_file = os.path.join(root_dir, 'output.csv')

        points_data = find_bulges(input_csv_file, output_csv_file)
        res = json.dumps(points_data)
        sys.stdout.write(res)
        sys.stdout.flush()
    except IndexError:
        sys.stderr.write("Error: Missing required command-line arguments.\n")
        sys.stderr.flush()
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.flush()
        sys.exit(1)

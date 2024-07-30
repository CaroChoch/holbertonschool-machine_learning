#!/usr/bin/env python3
""" displays the number of launches per rocket """
import requests
from datetime import datetime


def number_of_launches_per_rocket():
    """Displays the number of launches per rocket"""
    # URL of the SpaceX API for launches
    launches_url = 'https://api.spacexdata.com/v4/launches'
    # Make a GET request to the SpaceX API to get the data for all launches
    launches_response = requests.get(launches_url)
    # Parse the response as JSON
    launches_data = launches_response.json()

    # Dictionary to store the count of launches per rocket name
    rockets_launch_count = {}

    # Loop through each launch in the launch data
    for launch in launches_data:
        # Get the rocket ID for the current launch
        rocket_id = launch.get('rocket')
        # Make a GET request to get the rocket details using the rocket ID
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
            rocket_id)
        rocket_response = requests.get(rocket_url)
        # Parse the response as JSON to get rocket details
        rocket_data = rocket_response.json()
        # Get the rocket name from the rocket details
        rocket_name = rocket_data.get('name')

        # Update the count of launches for this rocket name
        if rocket_name not in rockets_launch_count:
            # If this rocket name is not already in the dictionary,
            # add it with a count of 1
            rockets_launch_count[rocket_name] = 1
        else:
            # If this rocket name is already in the dictionary,
            # increment the count by 1
            rockets_launch_count[rocket_name] += 1

    # Convert rockets_launch_count dictionary to a list of tuples for sorting
    rockets_list = list(rockets_launch_count.items())

    # Sort the list of rockets by rocket name in alphabetical order
    for current_index in range(len(rockets_list)):
        for comparison_index in range(current_index + 1, len(rockets_list)):
            if rockets_list[current_index][0] > rockets_list[
                    comparison_index][0]:
                # Swap the positions if the current rocket name is
                # greater than the comparison rocket name
                rockets_list[current_index], rockets_list[comparison_index] = (
                    rockets_list[comparison_index], rockets_list[current_index]
                )

    # Sort the list of rockets by the number of launches in descending order
    for current_index in range(len(rockets_list)):
        for comparison_index in range(current_index + 1, len(rockets_list)):
            if rockets_list[current_index][1] < rockets_list[
                    comparison_index][1]:
                # Swap the positions if the current launch count is less than
                # the comparison launch count
                rockets_list[current_index], rockets_list[comparison_index] = (
                    rockets_list[comparison_index], rockets_list[current_index]
                )

    # Print the sorted list of rockets and their launch counts
    for rocket in rockets_list:
        print("{}: {}".format(rocket[0], rocket[1]))


if __name__ == '__main__':
    print(number_of_launches_per_rocket())

#!/usr/bin/env python3
""" displays the number of launches per rocket """
import requests


if __name__ == '__main__':

    # URL of the SpaceX API for launches
    launches_url = 'https://api.spacexdata.com/v4/launches'
    # Make a GET request to the SpaceX API to get the data for all launches
    launches_response = requests.get(launches_url)
    # Parse the response as JSON
    launches_data = launches_response.json()

    # URL of the SpaceX API for rockets
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    # Make a GET request to the SpaceX API to get the data for all rockets
    rockets_response = requests.get(rockets_url)
    # Parse the response as JSON
    rockets_data = rockets_response.json()

    # Dictionary to store the count of launches per rocket name
    rockets_launch_count = {}

    # Loop through each launch in the launch data
    for launch in launches_data:
        # Get the rocket ID for the current launch
        rocket_id = launch['rocket']

        # Loop through each rocket in the rocket data
        for rocket in rockets_data:
            # Find the rocket name corresponding to the rocket ID
            if rocket['id'] == rocket_id:
                rocket_name = rocket['name']

        # Update the count of launches for this rocket name
        if rocket_name not in rockets_launch_count:
            # If this rocket name is not already in the dictionary,
            # add it with a count of 1
            rockets_launch_count[rocket_name] = 1
        else:
            # If this rocket name is already in the dictionary,
            # increment the count by 1
            rockets_launch_count[rocket_name] += 1

    # Convert rockets_launch_count dictionary to a list of tuples
    # and sort it by the number of launches in descending order
    rockets = sorted(rockets_launch_count.items(),
                     key=lambda x: x[1],
                     reverse=True)

    # Print the sorted list of rockets and their launch counts
    for rocket, nb_launches in rockets:
        print('{}: {}'.format(rocket, nb_launches))

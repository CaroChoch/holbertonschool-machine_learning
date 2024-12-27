#!/usr/bin/env python3
""" Display the first launch of each rocket """
from datetime import datetime
import requests


def get_first_launch():
    """
    Displays the first launch of each rocket

    Returns: Nothing
    """

    # URL of SpaceX API endpoint for upcoming launches
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches_response = requests.get(launches_url)  # Make GET request to API
    launches = launches_response.json()  # Parse the JSON response

    # Extract the Unix dates of all upcoming launches
    first_launch = None
    for launch in launches:
        if first_launch is None or \
                launch['date_unix'] < first_launch['date_unix']:
            first_launch = launch

    # Extract relevant information from the first launch
    launch_name = first_launch['name']
    launch_date = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    # Fetch rocket details using the rocket ID
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    rocket_response = requests.get(rocket_url)
    rocket_details = rocket_response.json()
    rocket_name = rocket_details['name']

    # Fetch launchpad details using the launchpad ID
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(
        launchpad_id)
    launchpad_response = requests.get(launchpad_url)
    launchpad_details = launchpad_response.json()
    launchpad_name = launchpad_details['name']
    launchpad_location = launchpad_details['locality']

    # Construct the information string
    launch_info = "{} ({}) {} - {} ({})".format(
        launch_name, launch_date, rocket_name,
        launchpad_name, launchpad_location
    )

    # Print the launch information
    print(launch_info)


if __name__ == '__main__':
    get_first_launch()

#!/usr/bin/env python3
""" Returns a list of ships that can hold a given number of passengers """
import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can hold a given number of passengers

    Arguments:
        - passengerCount is the number of passengers

    Returns:
        - a list of ships that can hold that many passengers
    """
    # Initialize empty list to store the names of ships that meet the criteria
    ships = []
    # URL of the Star Wars API endpoint for starships
    url = "https://swapi-api.hbtn.io/api/starships"

    # Loop through the paginated results
    while url:
        response = requests.get(url)  # Make a GET request to the API
        data = response.json()  # Parse the JSON response

        # Loop through the results in the current page
        for ship in data['results']:
            # Replace commas in the 'passengers' field to handle large numbers
            nb_passengers = ship['passengers'].replace(',', '')
            # Check if the 'passengers' field is not 'n/a' or 'unknown'
            # and meets the passengerCount criterion
            if nb_passengers not in ['n/a', 'unknown'] and \
               int(nb_passengers) >= passengerCount:
                # Add the name of the ship to the list if it meets the criteria
                ships.append(ship['name'])

        # Move to the next page of results
        url = data['next']

    # Return list of ships names that can hold the given number of passengers
    return ships

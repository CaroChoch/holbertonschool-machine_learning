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
        # Loop through the ships in the current page of results
        for ship in data['results']:
            try:
                # Check if the ship can hold at least passengerCount passengers
                if int(ship['passengers']) >= passengerCount:
                    # If it can, add the name of the ship to the list
                    ships.append(ship['name'])
            except ValueError:
                pass  # Skip ships with unknown passenger capacity

        # Move to the next page of results
        url = data['next']

    # Return list of ships names that can hold the given number of passengers
    return ships

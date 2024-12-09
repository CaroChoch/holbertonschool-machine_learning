#!/usr/bin/env python3
""" returns the list of names of the home planets of all sentient species """
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all sentient species
    """
    # URL of the Star Wars API endpoint for species
    url = "https://swapi-api.hbtn.io/api/species/"

    # Initialize empty list to store the names of the home planets
    planets = []

    while url is not None:
        response = requests.get(url)  # Make a GET request to the API
        data = response.json()  # Parse the JSON response
        # Loop through the species in the current page of results
        for species in data['results']:
            # Check if the species is classified as 'sentient'
            if species['designation'] == 'sentient' or \
                    species['classification'] == 'sentient':
                # Check if the homeworld is not None
                if species['homeworld'] is not None:
                    # Make a GET request to homeworld URL to get planet details
                    planet = requests.get(species['homeworld']).json()
                    # Add the name of the home planet to the list
                    planets.append(planet['name'])
        # Move to the next page of results
        url = data['next']
    # Return the list of home planet names
    return planets

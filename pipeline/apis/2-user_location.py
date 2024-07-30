#!/usr/bin/env python3
""" Prints the locations of a specific user """
import requests
import sys
import time

if __name__ == '__main__':
    # Get the URL from the command line argument
    url = sys.argv[1]
    # Set the headers to accept GitHub's API JSON format
    headers = {"Accept": "application/vnd.github+json"}
    # Make a GET request to the URL with the specified headers
    data = requests.get(url, headers=headers)

    # If the request is successful (status code 200)
    if data.status_code == 200:
        # Print the 'location' field from the JSON response
        print(data.json()['location'])

    # If the resource is not found (status code 404)
    elif data.status_code == 404:
        print("Not found")

    # If the request is forbidden (status code 403) due to rate limiting
    elif data.status_code == 403:
        # Get the time when the rate limit will reset
        limit_time = int(data.headers['X-Ratelimit-Reset'])
        # Get the current time
        now = int(time.time())
        # Calculate the remaining time until the rate limit resets in minutes
        result = int((limit_time - now) / 60)
        # Print the time remaining until the rate limit resets
        print("Reset in {} min".format(result))

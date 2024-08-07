#!/usr/bin/env python3
""" provides some stats about Nginx logs stored in MongoDB """
from pymongo import MongoClient


def log_stats():
    """
    Function that provides some stats about Nginx logs stored in MongoDB
    Return : nothing
    """

    # Connect to the MongoDB client
    client = MongoClient('mongodb://127.0.0.1:27017')

    # Access the 'nginx' collection within the 'logs' database
    collection = client.logs.nginx

    # Print the total number of documents (logs) in the collection
    print('{} logs'.format(collection.count_documents({})))

    # Define the list of HTTP methods to check
    print('Methods:')
    methods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']

    # Loop through each method and count the number of occurrences
    for method in methods:
        method_count = collection.count_documents({'method': method})
        print('\tmethod {}: {}'.format(method, method_count))

    # Count the number of documents where method is GET and path is /status
    status_count = collection.count_documents(
        {"method": "GET", "path": "/status"}
    )

    # Print the count of the status logs
    print('{} status check'.format(status_count))


if __name__ == '__main__':
    # Execute the log_stats function if this script is run directly
    log_stats()

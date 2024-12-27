#!/usr/bin/env python3
""" function that returns the list of school having a specific topic """
import pymongo


def schools_by_topic(mongo_collection, topic):
    """
    Function that returns the list of school having a specific topic
    Return : list of schools or empty list
    """
    return list(mongo_collection.find({"topics": topic}))

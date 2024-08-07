#!/usr/bin/env python3
""" Function that inserts a new document in a collection based on kwargs """
import pymongo


def insert_school(mongo_collection, **kwargs):
    """
    Function that inserts a new document in a collection based on kwargs
    Return : the new _id
    """
    if not mongo_collection or not kwargs:
        return None
    new_id = mongo_collection.insert_one(kwargs).inserted_id
    return new_id

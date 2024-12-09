#!/usr/bin/env python3
""" Python function that lists all documents in a collection """
import pymongo


def list_all(mongo_collection):
    """
    Function that lists all documents in a collection
    Return : list of documents or empty list
    """
    return list(mongo_collection.find())

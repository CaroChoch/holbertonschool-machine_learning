#!/usr/bin/env python3
""" Display a histogram of student grades for Project A"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Display a histogram of student grades for Project A
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xticks(np.arange(0, 101, 10))

    plt.show()

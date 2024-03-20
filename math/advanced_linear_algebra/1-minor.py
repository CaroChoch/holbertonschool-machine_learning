#!/usr/bin/env python3
"""
minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculate the minor of a matrix
    Argument:
        - matrix: list of lists whose minor should be calculated
    Returns: the minor of the matrix
    """

    # Check if the matrix is a non-empty list
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")

    # Check if each element of the matrix is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check if the matrix is a non-empty square matrix
    if len(matrix) == 1 and len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    # If the matrix is a 1x1 matrix, return [[1]]
    if len(matrix) == 1:
        return [[1]]

    minors = []
    # Iterate through each element of the matrix
    for i in range(len(matrix)):
        minors.append([])
        for j in range(len(matrix[i])):
            # Calculate the sub-matrix excluding row i and column j
            sub_matrix = [row[:j] + row[j+1:] for row in (
                matrix[:i] + matrix[i+1:])]
            # Calculate the determinant of the sub-matrix
            minors[i].append(determinant(sub_matrix))
    return minors


def determinant(matrix):
    """
    Calculate the determinant of a matrix
    Argument:
        - matrix: list of lists whose determinant should be calculated
    Returns: the determinant of the matrix
    """
    det = 0  # Initialize the determinant to 0

    # Check if the input is a valid list of lists
    if not isinstance(matrix, list) or not matrix:
        raise TypeError("matrix must be a list of lists")

    # Check if each row in the matrix is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check if the input matrix is a 1x1 matrix containing an empty list
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Check if the input matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    # Handle the case of a 1x1 matrix
    if len(matrix) == 1:
        det = matrix[0][0]
        return det

    # Handle the case of a 2x2 matrix
    if len(matrix) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return det

    # Handle the case of larger matrices using recursion approach
    for i in range(len(matrix[0])):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += matrix[0][i] * (-1) ** i * determinant(sub_matrix)
    return det

#!/usr/bin/env python3
""" Determinant of a matrix """


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

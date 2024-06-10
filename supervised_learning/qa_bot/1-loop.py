#!/usr/bin/env python3
"""Create a loop that takes in user input and prints the answer"""

exit_commands = ['exit', 'quit', 'goodbye', 'bye']


def main():
    """ Run the loop to answer questions """
    while True:
        # Prompt user for a question and remove any trailing newline characters
        question = input("Q: ").rstrip('\n')

        # Check if the user input is an exit command
        if question.lower() in exit_commands:
            print("A: Goodbye")
            break

        # Print a placeholder answer
        print("A:")


# Ensure the main function is called when the script is executed
if __name__ == "__main__":
    main()

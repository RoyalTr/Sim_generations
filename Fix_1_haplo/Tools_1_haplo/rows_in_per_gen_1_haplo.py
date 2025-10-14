#!/usr/bin/env python3
"""
rows_in_per_gen_1_haplo.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import os

def count_lines_in_file(filepath):
    # Check if the file exists before trying to open it.
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            line_count = 0
            for _ in file:
                line_count += 1
        return line_count
    except Exception as e:
        # Catch other potential errors, like permission issues
        print(f"An error occurred while reading the file: {e}")
        return None

if __name__ == '__main__':
    file_to_check = 'out_1_haplo_per_gen.txt'
    number_of_lines = count_lines_in_file(file_to_check)

    if number_of_lines is not None:
        print(f"The file '{file_to_check}' has {number_of_lines:,} lines.")

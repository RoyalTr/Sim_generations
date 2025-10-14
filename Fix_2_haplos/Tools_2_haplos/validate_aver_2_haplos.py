#!/usr/bin/env python3
"""
validate_aver.py
AUTHOR: Dr. Royal Truman
VERSION: 0.9
"""
import os

file_name = 'out_2_haplos_per_gen.txt'

if os.path.exists(file_name):
    try:
        file_size = os.path.getsize(file_name)
        approx_rows = int(file_size / 190)

        print(f"File size of {file_name} = {file_size:,} bytes")
        # f'{number:,}'

        print(f"This corresponds to about {approx_rows:,} rows of data")
    except Exception as e:
        # This catches any other unexpected errors, like permission issues
        print(f"An unexpected error occurred: {e}")
else:
    print(f"Error: File '{file_name}' not found")
# my_script.py

import sys
import json

# Read input value from command-line arguments
input_value = sys.argv[1]
input_value = eval(input_value)
print("The input type is: ")
output_value = type(input_value)

# Print the output value
print(output_value)
print("\n Exiting SSH. ")


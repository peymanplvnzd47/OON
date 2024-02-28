import json
import random

# Read the JSON file
with open('not_full_network.json', 'r') as file:
    data = json.load(file)

# Iterate over the data and modify lists with size 6
for node_data in data.values():
    for matrix_data in node_data.get('switching_matrix', {}).values():
        for node_values in matrix_data.values():
            if isinstance(node_values, list) and len(node_values) == 6:
                if all(value == 0 for value in node_values):
                    node_values.extend([0] * 4)
                else:
                    node_values.extend([random.randint(0, 1) for _ in range(4)])

# Write the modified data to a new JSON file
with open('new config with 10 ch random.json', 'w') as file:
    json.dump(data, file, indent=4)

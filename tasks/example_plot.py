import matplotlib.pyplot as plt
import numpy as np

# Turn on interactive mode
plt.ion()

# Create a simple plot
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Interactive Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the plot without blocking the code execution
plt.show()

# Continue running other lines of code
other_variable = 42
print(f'This is another line of code. Value: {other_variable}')

# Example of keeping the plot window open for a while
# plt.pause(5)  # Keep the plot open for 5 seconds

# You can add more code here, and the plot window will remain open

# Turn off interactive mode when you are done
plt.ioff()





from Lab8.core.elements import *

# Example usage to create a Network and perform operations
if __name__ == '__main__':

    root = Path(__file__).parent.parent
    input_folder = root / 'resources'

    three_scenarios_network = ['nodes.json', 'nodes_full.json', 'nodes_not_full.json']

    # file_input = input_folder / 'nodes.json'
    file_input = input_folder / 'nodes.json'
    # file_input = input_folder / 'nodes_not_full.json'

    # Assuming you have already created the Network instance

    network = Network(file_input)

    # Access and print the number of amplifiers for each line
    for line_label, line in network.lines.items():
        print(f"Line {line_label}: Number of Amplifiers = {line.n_amplifiers}")

    network.draw_beta(save_image=False, image_filename='network_plot.jpg')
    # plt.show()

#    network.draw_beta_with_amplifiers(save_image=False, image_filename='network_plot.jpg')



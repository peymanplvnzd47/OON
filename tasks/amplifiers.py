from Lab9.core.elements import *

# Example usage to create a Network and perform operations
if __name__ == '__main__':

    root = Path(__file__).parent.parent
    input_folder = root / 'resources_exam'

    three_scenarios_network = ['network.json', 'full_network.json', 'not_full_network.json']

    # file_input = input_folder / 'nodes.json'
    file_input = input_folder / 'network.json'
    # file_input = input_folder / 'nodes_not_full.json'

    # Assuming you have already created the Network instance

    network = Network(file_input)

    # Access and print the number of amplifiers for each line
    for line_label, line in network.lines.items():
        print(f"Line {line_label}: Number of Amplifiers = {line.n_amplifiers}")
    # Turn on interactive mode
    plt.ion()

    network.draw_beta(save_image=False, image_filename='network_plot.jpg')
    plt.ioff()
    plt.show()



#    network.draw_beta_with_amplifiers(save_image=False, image_filename='network_plot.jpg')



from Lab9.core.elements import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


# Example usage to create a Network and perform operations
if __name__ == '__main__':
    def split_and_zip_string_list(input_list):
        return [tuple(chars) for chars in input_list]


    def count_zeros(input_array):
        return len(input_array) - np.count_nonzero(input_array)



    root = Path(__file__).parent.parent
    input_folder = root / 'resources'

    # three_scenarios_network = ['network.json', 'full_network.json', 'not_full_network.json']
    three_scenarios_network = ['nodes.json', 'nodes_full.json', 'nodes_not_full.json']


    # file_input = input_folder / 'nodes.json'
    # file_input = input_folder / 'full_network.json'
    file_input = input_folder / 'network.json'
    # file_input = input_folder / 'nodes_full.json'



    # file_input = input_folder / 'nodes.json'
    # file_input = input_folder / 'nodes_full.json'
    # file_input = input_folder / 'nodes_not_full.json'

    TRx_strategy = ['fixed-rate']
    T_tot_connection_req = 200


    #for counter, TRx_strategy_indx in enumerate(TRx_strategy):
    TRx_strategy_indx = TRx_strategy[0]

        #for path_choice in ['snr']:
    path_choice = 'snr'

   # random.seed(1)
    latencies = []
    R_b_lat = []
    R_b_snr = []
    snrs = []
    paths_visu_snr = []
    paths_visu_laten = []
    latencies_without_Nones = []
    snrs_without_zeros = []
    R_b_snr_without_zeros = []
    rej_reas = []

    network = Network(file_input)
    network.set_transceiver_for_all_nodes(TRx_strategy_indx)
    route_space_snr = network.create_route_space_dataframe()
    route_space_latency = network.create_route_space_dataframe()
    network.release_all_channels()
    network.initialize_switching_matrices()
    M_value = 15
    # Generate channel traffic matrices for each channel
    channel_matrices = network.generate_channel_traffic_matrix( M_value)
    global req_matrices, channel_states_array, edge
    req_matrices = network.generate_request_traffic_matrix( M_value, uniform= True)
    line = network.lines['AD']
    channel_states_array = np.array([channel['state'] for channel in line.channels])



    for line in network.lines.values():
        line.state = 'free'
    # print("Starting of Switching Matrices:")
    # for node_label, node in network.nodes.items():
    #     print(f"Node {node_label}:")
    #     for src_node, dest_nodes in node.get_switching_matrix().items():
    #         print(f"  {src_node}: {dest_nodes}")



    #for i in range(T_tot_connection_req):
    def update_matrix(req_matrices, channel_states_array):
        global TRx_strategy_indx, path_choice, rej_reas, edge, R_b_gbps



        # input_node_label = random.choice(list(network.nodes.keys()))
        # print('input: ', input_node_label)
        # output_node_label = random.choice(list(network.nodes.keys()))
        # positive_indices = np.where(req_matrices > 0)
        # Step 1: Find positive elements and their corresponding indices
        positive_indices = np.where(req_matrices > 0)
        positive_elements = list(zip(positive_indices[0], positive_indices[1]))


            # Step 2: Randomly select one positive element
        selected_index = np.random.randint(0, len(positive_elements))
        selected_element = positive_elements[selected_index]
        # Step 3: Map the selected positive element's indices to their corresponding nodes
        input_node_label = list(network.nodes.keys())[selected_element[0]]
        output_node_label = list(network.nodes.keys())[selected_element[1]]


        print('INPUT: ', input_node_label)
        print('OUTPUT: ', output_node_label)

        # Create a Connection instance with 1 mW signal power
        connection = Connection(input_node_label, output_node_label, 0.001)
        Stream_result = network.stream([connection], label=path_choice)
        best_path = Stream_result[0]
        best_channel = Stream_result[1]
        # total_latency, _, total_snr_db = network.calculate_path_metrics(best_path, connection.signal_power)

        print('Rejection reason is ', Stream_result[2])
        rej_reas.append(Stream_result[2])
        block_adjacent = False
        print('Selected Channel is :', best_channel)
        R_b_gbps = connection.bit_rate


        if best_path:
        #    R_b_gbps = connection.bit_rate
        #    print(i, path_choice, best_path, connection.snr, 'R_b = ', R_b_gbps, 'Gbps')
            network.update_switching_matrices(best_path, best_channel, block_adjacent)
        # Update the traffic matrix for the random channel
        #     channel_matrices[best_channel] = network.update_traffic_matrix(channel_matrices[best_channel],
        #                                                              input_node_label, output_node_label,
        #                                                              best_channel, R_b_gbps)
            req_matrices = network.update_traffic_matrix(req_matrices,
                                                         input_node_label, output_node_label, R_b_gbps)
            # Iterate over each element of req_matrices
            for i in range(len(req_matrices)):
                for j in range(len(req_matrices[i])):
                    if req_matrices[i][j] < 0:
                        req_matrices[i][j] = 0







            for node_label in best_path[1:-1]:
                node = network.nodes[node_label]
                updated_switching_matrix = node.get_switching_matrix()

                # print(f"Switching matrix for node {node_label} (updated):")
                # for src_node, dest_nodes in updated_switching_matrix.items():
                #     print(f"{src_node}: {dest_nodes}")

            lines_to_use = network.get_lines_from_path(best_path)
            print(lines_to_use)
            edge = split_and_zip_string_list(lines_to_use)

            print(edge)

            paths_visu_snr.append(best_path) if path_choice == 'snr' else paths_visu_laten.append(best_path)
            channel_state_path = np.ones(10, dtype=int)
            print("Updated Channel States:")
            for line_label in network.get_lines_from_path(best_path):
                line = network.lines[line_label]

                # Example values for Rs, df, and Nch


                # Calculate and print the optimal power
                # optimal_power = line.Optimal_power(Rs, df, Nch)
                # print(f"Optimal Power: {optimal_power} Watts")
                print(f"  Line: {line_label}")
                channel_states_array = np.array([channel['state'] for channel in line.channels])
                channel_state_path = channel_state_path * channel_states_array
                print(channel_states_array)
                for channel in range(len(line.channels)):
                    channel_state = line.get_channel_state(channel)
                    print(f"    Channel {channel}: {channel_state}")
            print('Channel state of the path', best_path, channel_state_path)

            best_path_all_lines = ['->'.join(line[i:i + 2]) for line in lines_to_use for i in
                                   range(len(line) - 1)]

            # Store latency or SNR as needed
            if path_choice == 'snr':
                for path_line in best_path_all_lines:
                    route_space_snr.loc[
                        route_space_snr['Path'].str.contains(path_line), f'Channel_{best_channel}'] = 'Occupied'
                snrs.append(connection.snr)

        # Store latency or SNR as needed
        if path_choice == 'snr':
            snrs.append(connection.snr)
            R_b_snr.append(connection.bit_rate)
            latencies.append(connection.latency)
            print('SNR is: ', connection.snr)
        return channel_states_array


    print("Final Switching Matrices:")
    for node_label, node in network.nodes.items():
        print(f"Node {node_label}:")
        for src_node, dest_nodes in node.get_switching_matrix().items():
            print(f"  {src_node}: {dest_nodes}")

    print('**************************', 'END OF SIMULATION OF ', path_choice, '**************************')



graph = network.graph


# Create a directed graph
G = nx.DiGraph()
counter = T_tot_connection_req
# Add nodes and edges to the graph
for node, data in graph.items():
    G.add_node(node, pos=data["position"])
    for neighbor in data["connected_nodes"]:
        G.add_edge(node, neighbor)

# Assign strength to each edge (initially set to 0)
edge_strength = {edge: 0 for edge in G.edges()}

# Create colormap
colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # green, yellow, red
cmap_name = 'custom_colormap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=10)

# Draw the graph with initial edge colors
pos = nx.get_node_attributes(G, 'pos')
pos = nx.kamada_kawai_layout(G)
# Create color bar
figN, axN = plt.subplots(figsize=(8, 7))
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=10))
sm.set_array([])
cbar = figN.colorbar(sm, ax=axN, orientation='vertical')
cbar.set_label('Edge Strength')

# Function to update edge strength and redraw the graph
def plot_final_traffic_matrix(req_matrices, network, M_value):
    figZ, axZ = plt.subplots()
    sns.set(font_scale=1.2)
    selected_channel_show = 0
    axZ = sns.heatmap(req_matrices, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Bit Rate (Gbps)'})
    axZ.set_xticklabels(network.nodes.keys())
    axZ.set_yticklabels(network.nodes.keys())
    plt.title(f"Updated Channel Traffic Matrix (M={M_value})")
    plt.show()
def update(frame):

    global edge_strength, counter, M_value, R_b_gbps
    ch = update_matrix(req_matrices, channel_states_array)  # Call the function to update the matrix
    # Randomly select an edge



    # Check if the edge strength is not saturated
    # if edge_strength[edge[0]] < 10:
    #     # Increase the strength of the selected edge by 1
    if all(strength == 10 for strength in edge_strength.values()):
        ani.event_source.stop()  # Stop the animation if all edge strengths are saturated
    positive_indices_ch = np.where(req_matrices > 0)
    positive_elements_ch = list(zip(positive_indices_ch[0], positive_indices_ch[1]))
    if len(positive_elements_ch) == 0:
        plot_final_traffic_matrix(req_matrices, network, M_value)
        ani.event_source.stop()
    print('edge', (ch))
    edge_strength[edge[0]]  = count_zeros(ch)
    # else:
    #     ani.event_source.stop()  # Stop the animation after the for loop finishes
    # # Clear the current plot
    axN.clear()
    # Redraw the graph with updated edge colors
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='lightblue', font_color='black', font_size=14, arrowsize=12, edge_color=[cm(edge_strength[edge]) for edge in G.edges()], ax=axN, width=5)

    # Update counter


    if counter > 0:
        counter -= 1
    else:
        figZ, axZ = plt.subplots()
        # Use Seaborn to visualize the updated channel traffic matrix
        sns.set(font_scale=1.2)
        # plt.figure(figsize=(10, 8))

        selected_channel_show = 0
        axZ = sns.heatmap(req_matrices, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Bit Rate (Gbps)'})
        axZ.set_xticklabels(network.nodes.keys())
        axZ.set_yticklabels(network.nodes.keys())
        plt.title(f"Updated Channel Traffic Matrix (M={M_value})")
        print(network.nodes.keys())





        plt.show()

        ani.event_source.stop()  # Stop the animation after the for loop finishes



# Create animation
ani = FuncAnimation(figN, update, frames=20, interval=200)
num_frames = T_tot_connection_req

# # Create animation
#ani2 = FuncAnimation(fig, update, frames=num_frames, blit=True, interval=200)

# Display the plot
plt.tight_layout()

fig, ax = plt.subplots()
# Use Seaborn to visualize the updated channel traffic matrix
sns.set(font_scale=1.2)
#plt.figure(figsize=(10, 8))

selected_channel_show = 0
ax = sns.heatmap(req_matrices, annot=True, fmt=".0f", cmap="viridis", cbar_kws={'label': 'Bit Rate (Gbps)'})
ax.set_xticklabels(network.nodes.keys())
ax.set_yticklabels(network.nodes.keys())
plt.title(f"Updated Channel Traffic Matrix (M={M_value})")
print(network.nodes.keys())


plt.show()
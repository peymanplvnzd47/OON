from Lab9.core.elements import *

# Example usage to create a Network and perform operations
if __name__ == '__main__':



    root = Path(__file__).parent.parent
    input_folder = root / 'resources'

    # three_scenarios_network = ['network.json', 'full_network.json', 'not_full_network.json']
    three_scenarios_network = ['nodes.json', 'nodes_full.json', 'nodes_not_full.json']


    # file_input = input_folder / 'nodes.json'
    # file_input = input_folder / 'full_network.json'
    file_input = input_folder / 'nodes_not_full.json'
    # file_input = input_folder / 'nodes_full.json'



    # file_input = input_folder / 'nodes.json'
    # file_input = input_folder / 'nodes_full.json'
    # file_input = input_folder / 'nodes_not_full.json'

    TRx_strategy = ['fixed-rate']
    T_tot_connection_req = 150
    # Perform the run for both latency and SNR paths
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 5))

    for counter, TRx_strategy_indx in enumerate(TRx_strategy):

        for path_choice in ['snr']:

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
            cap_sum = []

            network = Network(file_input)
            network.set_transceiver_for_all_nodes(TRx_strategy_indx)
            route_space_snr = network.create_route_space_dataframe()
            route_space_latency = network.create_route_space_dataframe()
            network.release_all_channels()
            network.initialize_switching_matrices()
            M_value = 3
            # Generate channel traffic matrices for each channel
            channel_matrices = network.generate_channel_traffic_matrix( M_value)
            req_matrices = network.generate_request_traffic_matrix( M_value)

            for line in network.lines.values():
                line.state = 'free'
            print("Starting of Switching Matrices:")
            for node_label, node in network.nodes.items():
                print(f"Node {node_label}:")
                for src_node, dest_nodes in node.get_switching_matrix().items():
                    print(f"  {src_node}: {dest_nodes}")


            for i in range(T_tot_connection_req):

                # input_node_label = random.choice(list(network.nodes.keys()))
                # print('input: ', input_node_label)
                # output_node_label = random.choice(list(network.nodes.keys()))
                # positive_indices = np.where(req_matrices > 0)
                # Step 1: Find positive elements and their corresponding indices
                positive_indices = np.where(req_matrices > 0)
                positive_elements = list(zip(positive_indices[0], positive_indices[1]))
                # Step 2: Randomly select one positive element
                # Check if there are any positive elements
                if len(positive_elements) == 0:
                    print("All the requests are allocated")
                    break

                selected_index = np.random.randint(0, len(positive_elements))
                selected_element = positive_elements[selected_index]
                # Step 3: Map the selected positive element's indices to their corresponding nodes
                input_node_label = list(network.nodes.keys())[selected_element[0]]
                output_node_label = list(network.nodes.keys())[selected_element[1]]


                print('input: ', input_node_label)
                print('input: ', output_node_label)


                # while input_node_label == output_node_label:
                #     output_node_label = random.choice(list(network.nodes.keys()))
                # print('output', output_node_label)

                # Create a Connection instance with 1 mW signal power
                connection = Connection(input_node_label, output_node_label, 0.001)
                Stream_result = network.stream([connection], label=path_choice)
                best_path = Stream_result[0]
                best_channel = Stream_result[1]
                # total_latency, _, total_snr_db = network.calculate_path_metrics(best_path, connection.signal_power)

                print('Rejection reason is ', Stream_result[2])
                rej_reas.append(Stream_result[2])
                block_adjacent = True
                print('Selected Channel is :', best_channel)
                R_b_gbps = connection.bit_rate

                print(i, path_choice, best_path, connection.snr, 'R_b = ', R_b_gbps, 'Gbps')
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
                    sum_of_matrix = sum(sum(row) for row in req_matrices)
                    cap_sum.append(sum_of_matrix)
                    # Iterate over each element of req_matrices
                    for i in range(len(req_matrices)):
                        for j in range(len(req_matrices[i])):
                            if req_matrices[i][j] < 0:
                                req_matrices[i][j] = 0







                    for node_label in best_path[1:-1]:
                        node = network.nodes[node_label]
                        updated_switching_matrix = node.get_switching_matrix()

                        print(f"Switching matrix for node {node_label} (updated):")
                        for src_node, dest_nodes in updated_switching_matrix.items():
                            print(f"{src_node}: {dest_nodes}")

                    lines_to_use = network.get_lines_from_path(best_path)

                    paths_visu_snr.append(best_path) if path_choice == 'snr' else paths_visu_laten.append(best_path)
                    channel_state_path = np.ones(10, dtype=int)
                    print("Updated Channel States:")
                    for line_label in network.get_lines_from_path(best_path):
                        line = network.lines[line_label]
                        # Example values for Rs, df, and Nch


                        # Calculate and print the optimal power
                        # optimal_power = line.Optimal_power(Rs, df, Nch)
                        # print(f"Optimal Power: {optimal_power} Watts")
                        # print(f"  Line: {line_label}")
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

            print("Final Switching Matrices:")
            for node_label, node in network.nodes.items():
                print(f"Node {node_label}:")
                for src_node, dest_nodes in node.get_switching_matrix().items():
                    print(f"  {src_node}: {dest_nodes}")

            print('**************************', 'END OF SIMULATION OF ', path_choice, '**************************')

        # Count the number of None values in snrs
        num_none = latencies.count(None)

        num_zero_snr = snrs.count(0)
        #num_none_snr_bitrate = snrs.count(-100)
        #print('number of nones due to bitrate', num_none_snr_bitrate)


        print(file_input)

        print(f"Number of None values for Latency: {num_none}")
        # route_space_latency.to_excel('Channels_occupancy_3 Final _v4.xlsx', index=False)
        print(f"Number of Zero SNR values: {num_zero_snr}")

        line_states = network.probe(['A', 'B'], 4)
        print(line_states)

        # Filter out the N
        # one values from snrs
        latencies_without_Nones = [lt for lt in latencies if lt is not None]
        snrs_without_zeros = [lt for lt in snrs if lt != 0]
        #snrs_without_zeros = [lt for lt in snrs if lt != -100]


        R_b_snr_without_zeros = [lt for lt in R_b_snr if lt != 0]



        # Set up the subplot layout
        #fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 5))

        # Plot 1: Distribution of SNRs
        sns.histplot(snrs_without_zeros, bins=20, color='blue', alpha=0.7, ax=axes[ counter, 0])
        axes[counter, 0].set(xlabel='SNR (dB)', ylabel='Occurrences', title=f'Distribution of SNRs -{TRx_strategy_indx }')
        axes[counter, 0].grid(True)

        # Plot 2: Distribution of Latencies
        sns.histplot(latencies_without_Nones, bins=20, color='red', alpha=0.7, ax=axes[counter, 1])
        axes[counter, 1].set(xlabel='Latency (s)', ylabel='Occurrences', title='Distribution of Latencies')
        axes[counter, 1].grid(True)

        # Plot 3: Distribution of Bit rate based on snr optimization
        sns.histplot(R_b_snr_without_zeros, bins=20, color='orange', alpha=0.7, ax=axes[counter, 2])
        axes[counter, 2].set(xlabel='Bitrate', ylabel='Occurrences',
                    title=f'Distribution of Bit rate ')
        axes[counter, 2].grid(True)

        # Add a common title for the entire set of subplots
        fig.suptitle(f'Distribution of SNR, latency, and bitrate for Network with best snr path choice',
                     fontsize=14,
                     fontweight='bold')


        # Adjust layout for better visualization
        plt.tight_layout()
        # Calculate and display averages in the last column
        average_snr = np.mean(snrs_without_zeros)
        average_latency = np.mean(latencies_without_Nones)
        average_bitrate = np.mean(R_b_snr_without_zeros)
        capacity = np.round(np.sum(R_b_snr_without_zeros) / 1e3, 2)

        # Set up a yellow box in the last column for textual values
        axes[counter, 3].text(0.5, 0.5,
                              f'{TRx_strategy_indx}:\n'
                              f'Average SNR: {average_snr:.4f}\n'
                              f'Average Latency: {average_latency:.4f}\n'
                              f'Average Bitrate: {average_bitrate:.3f} Gbps\n'
                              f'Total connections: {T_tot_connection_req:.4f}\n'
                              f'Failure percentage : {num_zero_snr / T_tot_connection_req * 100:.3f}\n'
                              f'Block adjacent channels: {str(block_adjacent)}\n'
                              f'Total Capacity: {str(capacity)} Tbps',
                              fontsize=9, va='center', ha='center', color='black',
                              bbox=dict(facecolor='brown', alpha=0.5))

        axes[counter, 3].axis('off')
        print(rej_reas)
        unique_values = set(rej_reas)

        # Print the number of repetitions for each unique element
        for value in unique_values:
            count = rej_reas.count(value)
            print(f"Element {value} has {count} repetitions.")

    # plt.show()


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


plt.figure(figsize=(10, 6))

# Plot using seaborn
sns.lineplot(data=cap_sum)
print(cap_sum)
# Set title and labels
plt.title('Request Capacity Allocation by Time')
plt.xlabel('Time')
plt.ylabel('Capacity')

plt.show()





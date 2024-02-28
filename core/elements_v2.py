
import json
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import erfcinv
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.special
import pandas as pd
import math
from scipy import constants
from Lab9.core.parameters import speed_of_light
from Lab9.core.math_utils import  *
import random
import numpy as np
from matplotlib.animation import FuncAnimation
# Checking branch
from pathlib import Path


class SignalInformation:
    def __init__(self, signal_power, path):
        self.signal_power = signal_power
        self.noise_power = 0.0
        self.latency = 0.0
        self.path = path

    def update_signal_power(self, increment):
        if increment >= 0:
            self.signal_power += increment

    def update_noise_power(self, increment):
        if increment >= 0:
            self.noise_power += increment

    def update_latency(self, increment):
        if increment >= 0:
            self.latency += increment

    def update_path(self, node_label):
        self.path.append(node_label)

    def get_signal_power(self):
        return self.signal_power

    def get_noise_power(self):
        return self.noise_power

    def get_latency(self):
        return self.latency

    def get_path(self):
        return self.path

    def __str__(self):
        return f"Signal Information: Signal Power = {self.signal_power}, Noise Power = {self.noise_power}, Latency = {self.latency}, Path = {self.path}"





class Lightpath(SignalInformation):
    Rs = 32e9  # Signal symbol rate in Hz
    df = 50e9  # Frequency spacing between two consecutive channels in Hz

    def __init__(self, signal_power, path, channel):
        super().__init__(signal_power, path)
        self.channel = channel

    def get_channel(self):
        return self.channel

    def set_channel(self, new_channel):
        # You can add validation or checks if needed
        self.channel = new_channel

    def __str__(self):
        return f"Lightpath: Signal Power = {self.signal_power}, Noise Power = {self.noise_power}, " \
               f"Latency = {self.latency}, Path = {self.path}, Channel = {self.channel}, " \
               f"Signal Symbol Rate (Rs) = {self.Rs / 1e9} GHz, " \
               f"Frequency Spacing (df) = {self.df / 1e9} GHz"



class Node:
    def __init__(self, node_data):
        self.label = node_data['label']
        self.position = node_data['position']
        self.connected_nodes = node_data['connected_nodes']
        self.successive = {}
        self.switching_matrix = node_data.get('switching_matrix', None)  # Include switching_matrix in initialization
        self.transceiver = node_data.get('transceiver', 'flex-rate')  # Include transceiver attribute with default value



    def __str__(self):
        return f"Node: Label = {self.label}, Position = {self.position}, Connected Nodes = {self.connected_nodes}, Transceiver = {self.transceiver}"

    def set_switching_matrix(self, switching_matrix):
        self.switching_matrix = switching_matrix

    def get_switching_matrix(self):
        return self.switching_matrix

    def set_transceiver(self, transceiver):
        self.transceiver = transceiver

    def Optimal_power(self, ASE_noise, etha_nli, N_spans, B_n):

        P_opt = (ASE_noise/(2 * etha_nli * N_spans * B_n))**(1/3)

        return P_opt


class Line:
    def __init__(self, line_data, num_channels=10):
        self.label = line_data['label']
        self.length = line_data['length']
        self.successive = {}
        self.channels = [{'state': 1} for _ in range(num_channels)]  # Initialize all channels as 1

        # Constants for amplifiers
        self.amplifier_spacing = 80e3  # Amplifier spacing in meters
        self.gain = 16  # Gain of the optical amplifier in dB
        self.noise_figure = 5.5  # Noise figure of the optical amplifier in dB

        # Fiber characteristics
        self.alpha_dB = 0.2  # Attenuation coefficient in dB/km
        self.beta2_abs = 21.3e-12  # Absolute value of dispersion parameter in ps^2/km
        self.gamma = 1.27e-3  # Nonlinear coefficient in (W m)^−1
        # Convert alpha from dB/km to 1/m using the conversion factor log10(e)/10
        self.alpha = self.alpha_dB / (20 * math.log10(math.e))

        # Calculate the number of amplifiers based on the line length
        self.n_amplifiers = self.calculate_amplifiers()
        self.n_spans = self.calculate_amplifiers() # -1
        self.signal_power_line = 0.001
        self.GSNR_dB_new =5
        self.FP = 0.7 # Filtering penalty
        self.df_l =  50e9

    def calculate_eta_nli(self, alpha, beta2_abs, gamma, length, Rs, df_l, Nch):
        # Constants
        pi = math.pi

        # Calculate effective length
        Leff = 1 / (2 * alpha * 1e-3) # In km
        # Calculate eta_nli
        eta_nli = (16 / (27 * pi)) * math.log((pi ** 2 / 2) * ((abs(beta2_abs) / alpha) * Rs ** 2 * Nch ** (2 * Rs / df_l))) \
                  * alpha / abs(beta2_abs) * gamma ** 2 * Leff ** 2 / Rs ** 3

        return eta_nli

    def fiber_characteristics(self):
        return f"Attenuation Coefficient (α): {self.alpha:.6f} 1/m, " \
               f"Dispersion Parameter (|β₂|): {self.beta2_abs:.6e} ps^2/km, " \
               f"Nonlinear Coefficient (γ): {self.gamma:.6e} (W m)^−1"

    def calculate_amplifiers(self):
        # Calculate the number of amplifiers needed based on the line length and amplifier spacing
        return int(self.length / self.amplifier_spacing)

    # Define the function to calculate GSNR
    def calculate_GSNR_change(self, GSNR_db, alpha_dB, length, n_amplifiers, gain):
        updated_gsnr_values = []
        for i in range(length):
            # Calculate the updated GSNR after each amplifier
            updated_gsnr = GSNR_db - alpha_dB / 1e3 * (i + 1) + (i + 1) * n_amplifiers * gain
            updated_gsnr_values.append(updated_gsnr)
        return updated_gsnr_values

    def ase_generation(self):
        # Calculate ASE generation based on the number of amplifiers, Plank constant, and other constants
        h = constants.Planck  # Plank constant
        f = 193.414e12  # Frequency in Hz (C-band center)
        B_n = 12.5e9  # Noise bandwidth in Hz

        ase = self.n_amplifiers * h * f * B_n * db_to_linear(self.noise_figure) * (10 ** (self.gain / 10 )- 1)
        return ase

    def nli_generation(self, signal_power):
        """
        Calculate the total amount of nonlinear interference noise in linear units.

        Parameters:
        - signal_power: Signal power in watts.

        Returns:
        - Total nonlinear interference noise in linear units.
        """
        # Calculate the total amount of nonlinear interference noise using the provided formula
        Bn = 12.5e9  # Noise bandwidth in Hz
        Rs = 32e9
        Nch = 10

        df = 50e9
        eta_nli = self.calculate_eta_nli(self.alpha, self.beta2_abs, self.gamma, self.length, Rs, df, Nch)

        nli = signal_power ** 3 * abs(eta_nli) * self.n_amplifiers * Bn
        return nli

    def latency_generation(self):
        # Calculate the latency based on the length and the speed of light in fiber
        speed_of_light = constants.speed_of_light  # Speed of light in meters per second in vacuum
        latency = (3 / 2) * (self.length / speed_of_light)
        return latency

    def noise_generation(self, signal_power):
        # Calculate total noise generation including ASE and NLI
        total_noise = self.ase_generation() + self.nli_generation(signal_power)
        return total_noise

    def Optimal_power(self, Rs, df, Nch):
        eta_nli = self.calculate_eta_nli(self.alpha, self.beta2_abs, self.gamma, self.length, Rs, df, Nch)
        B_n = 12.5e9

        P_opt = ((self.ase_generation()/(2 * abs(eta_nli)* self.n_spans * B_n)))**(1/3)
        self.signal_power_line = P_opt * 0.001

        return self.signal_power_line

    def calculate_GSNR(self, sig_pow):

        GSNR_lin =  (sig_pow/self.noise_generation(sig_pow)) * 0.5

        GSNR_db = linear_to_db(GSNR_lin)
        self.GSNR_dB_new = GSNR_db

        return GSNR_db

    def ISNR(self, GSNR_dB_new):

        ISNR = 1/db_to_linear(GSNR_dB_new)
        return(ISNR)


    def set_channel_state(self, channel, new_state):
        # Set the state of the specified channel (1 or 0)
        if 0 <= channel < len(self.channels):
            self.channels[channel]['state'] = new_state

    def get_channel_state(self, channel):
        # Get the state of the specified channel
        if 0 <= channel < len(self.channels):
            return self.channels[channel]['state']
        else:
            return None

    def get_free_channels(self):
        # Get a list of indices of free channels
        return [i for i, channel in enumerate(self.channels) if channel['state'] == 1]

    # Other methods and attributes
    def __str__(self):
        return f"Line: Label = {self.label}, Length = {self.length}, Channels = {self.channels}"
class Network:
    def __init__(self, json_file):

        self.nodes = {}
        self.lines = {}
        self.num_channels = 10
        self.channels = list(range(self.num_channels))

        self.graph = self.load_graph_from_json(json_file)

        for node_label, node_data in self.graph.items():
            self.nodes[node_label] = Node({
                "label": node_label,
                "connected_nodes": node_data["connected_nodes"],
                "position": tuple(node_data["position"]),
                "switching_matrix": node_data.get("switching_matrix", None)  # Include switching_matrix in node creation
            })



        with open(json_file) as f:
            data = json.load(f)
            for node_label, node_data in data.items():
                self.nodes[node_label] = Node({
                    "label": node_label,
                    "connected_nodes": node_data["connected_nodes"],
                    "position": tuple(node_data["position"]),
                    "switching_matrix": node_data.get("switching_matrix", None)
                    # Include switching_matrix in node creation
                })
        # Create the 'weighted_paths' attribute using the 'create_paths_dataframe' method
        self.weighted_paths = self.create_paths_dataframe()
        # Create the 'route_space' attribute
        self.route_space = self.create_route_space_dataframe()

        # Create lines and calculate their lengths
        for node_label, node in self.nodes.items():
            for neighbor_label in node.connected_nodes:
                line_label = node_label + neighbor_label
                if line_label not in self.lines and neighbor_label in self.nodes:
                    self.lines[line_label] = Line({
                        "label": line_label,
                        "length": self.calculate_line_length(node, self.nodes[neighbor_label])
                    })

        # Connect nodes and lines
        self.connect()

    def generate_channel_traffic_matrix(self, M):
        num_nodes = len(self.nodes)
        channel_traffic_matrices = []
        for channel_number in range(1, self.num_channels + 1):
            channel_traffic_matrix = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # Set non-diagonal requests
                        channel_traffic_matrix[i, j] = M * 100
            channel_traffic_matrices.append(channel_traffic_matrix)
        return channel_traffic_matrices

    # def generate_request_traffic_matrix(self, M):
    #     num_nodes = len(self.nodes)
    #
    #
    #     req_traffic_matrix = np.zeros((num_nodes, num_nodes))
    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             if i != j:
    #                 # Set non-diagonal requests
    #                 req_traffic_matrix[i, j] = M * 100
    #
    #     return req_traffic_matrix

    import numpy as np


    def generate_request_traffic_matrix(self, M, uniform=True):
        num_nodes = len(self.nodes)

        req_traffic_matrix = np.zeros((num_nodes, num_nodes))
        if uniform:
            # Generate a uniform matrix
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # Set non-diagonal requests
                        req_traffic_matrix[i, j] = M * 100
        else:
            # Generate a random matrix
            req_traffic_matrix = np.random.randint(1, M * 100, size=(num_nodes, num_nodes))
            np.fill_diagonal(req_traffic_matrix, 0)  # Ensure diagonal elements are zero

        return req_traffic_matrix

    def update_traffic_matrix(self, traffic_matrix, input_node, output_node, R_b_gbps):
        # Subtract R_b_gbps from the specified channel's matrix element
        # updated_matrix = np.copy(traffic_matrix[channel - 1])  # Copy to avoid modifying the original matrix
        node_indices = {node_label: i for i, node_label in enumerate(self.nodes.keys())}

        # Check if the updated value is non-negative
        # if traffic_matrix[node_indices[input_node], node_indices[output_node]] - R_b_gbps < 0:
        #     raise ValueError(f"Error: Saturation in Channel-{channel}. "
        #                      f"Attempting to subtract {R_b_gbps} Gbps, but result would be negative.")

        # Update the matrix
        traffic_matrix[node_indices[input_node], node_indices[output_node]] -= R_b_gbps
        return traffic_matrix

    def initialize_switching_matrices(self, num_channels=10):
        for node in self.nodes.values():
            if node.switching_matrix is None:
                switching_matrix = self.generate_switching_matrix_for_node(node, num_channels)
                node.set_switching_matrix(switching_matrix)
            else:
                switching_matrix = node.get_switching_matrix()
                node.set_switching_matrix(switching_matrix)

    def print_switching_matrices(self):
        for node_label, node in self.nodes.items():
            print(f"Switching matrix for node {node_label}:")
            if node.switching_matrix:
                for src_node, dest_nodes in node.switching_matrix.items():
                    print(f"{src_node}: {dest_nodes}")
            else:
                print("No switching matrix available for this node.")
            print("\n")

    # def initialize_switching_matrices(self, num_channels=10):
    #     for node in self.nodes.values():
    #         switching_matrix = self.generate_switching_matrix_for_node(node, num_channels)
    #         node.set_switching_matrix(switching_matrix)

    def load_graph_from_json(self, json_file):
        with open(json_file) as f:
            return json.load(f)

    # Example usage:
    def set_transceiver_for_all_nodes(self, transceiver):
        for node_label, node in self.nodes.items():
            node.set_transceiver(transceiver)

    def generate_switching_matrix_for_node(self, node, num_channels):
        connected_nodes = node.connected_nodes
        num_nodes = len(connected_nodes)

        # Initialize a dictionary for the switching matrix
        switching_matrix = {neighbor: {n: [0] * num_channels for n in connected_nodes} for neighbor in connected_nodes}

        # Populate the matrix based on connectivity
        for i, src_node in enumerate(connected_nodes):
            for j, dest_node in enumerate(connected_nodes):
                if src_node != dest_node:
                    for channel in range(num_channels):
                        switching_matrix[src_node][dest_node][channel] = 1

        return switching_matrix

    def switching_matrix_gen(self, path, selected_channel):
        node_sw = path[1:-1]
        node_pairs = [[path[i], path[i + 2]] for i in range(len(path) - 2)]
        num_ch = 10

        SW_Matrix_hist = []
        for node_swich in node_sw:
            node = self.nodes[node_swich]
            switching_matrix = self.generate_switching_matrix_for_node(node, num_ch)
            SW_Matrix_hist.append(switching_matrix)
            node.set_switching_matrix(switching_matrix)  # Set switching_matrix for each node

        for pairs, SW_Matrix in zip(node_pairs, SW_Matrix_hist):
            self.update_switching_matrix(SW_Matrix, pairs[0], pairs[1], selected_channel)

        print("Selected Channel: ", selected_channel)
        print("Whole Path", path)
        print("Switching nodes", node_sw)
        print("Switching Paths", node_pairs)

        for i in range(len(node_sw)):
            node_label = node_sw[i]
            print(f"Switching matrix for node {node_label}(updated):")
            for src_node, dest_nodes in SW_Matrix_hist[i].items():
                print(f"{src_node}: {dest_nodes}")

        return SW_Matrix_hist

    def is_path_valid(self, path, channel):
        all_nodes_valid = True  # Initialize as True

        # Exclude the first and last nodes
        node_pairs = [[path[i], path[i + 2]] for i in range(len(path) - 2)]


        for node_label,node_pair in zip(path[1:-1],node_pairs):
            node = self.nodes[node_label]
            validity_path = []

            # Get the switching matrix for the current node
            src_node_matrix = np.array(node.get_switching_matrix()[node_pair[0]][node_pair[1]][channel])
            validity_path.append(src_node_matrix)

            # Check if the switching matrix is valid (contains at least one '1')
        if not all(x == 1 for x in validity_path):
            print(f"Node {node_pair[0]} is NOT valid for channel {channel}.")
            all_nodes_valid = False

        return all_nodes_valid

    def update_switching_matrix(self, switching_matrix, source_node, destination_node, channel, block_adjacent):
        for neighbor, values in switching_matrix[source_node].items():
            if neighbor == destination_node or neighbor == source_node:

                #switching_matrix[destination_node][neighbor][channel] = 0
                switching_matrix[source_node][neighbor][channel] = 0

                # Block the adjacent channels based on the specified rules
                if block_adjacent:
                    for adjacent_channel in [channel - 1, channel + 1]:
                        if 0 <= adjacent_channel < self.num_channels:
                            switching_matrix[source_node][neighbor][adjacent_channel] = 0

    def update_switching_matrices(self, path, selected_channel, block_adjacent=True):
        node_sw = path[1:-1]
        node_pairs = [[path[i], path[i + 2]] for i in range(len(path) - 2)]

        for node_swich, pairs in zip(node_sw, node_pairs):
            node = self.nodes[node_swich]
            switching_matrix = node.get_switching_matrix()
            self.update_switching_matrix(switching_matrix, pairs[0], pairs[1], selected_channel, block_adjacent)

    def release_all_channels(self):
        for line_label, line in self.lines.items():
            for channel in range(len(line.channels)):
                line.set_channel_state(channel, 1)

    def connect(self):
        for node in self.nodes.values():
            node.successive = {label: self.nodes[label] for label in node.connected_nodes}



    def calculate_line_length(self, node1, node2):
        return math.dist(node1.position, node2.position)

    # Inside your Network class
    # Inside your Network class

            # Update other information as needed

    def update_route_space_df(self, route_space_df, path, channel_states):
        for channel, state in enumerate(channel_states):
            route_space_df.loc[route_space_df['Path'] == path, f'Channel_{channel}'] = state

    def create_route_space_dataframe(self):
        # Create a DataFrame to represent the route space
        channels = list(range(10))  # Assuming 10 channels for each line
        columns = ["Path"] + [f"Channel_{channel}" for channel in channels]
        route_space_data = []

        # Fill the DataFrame with initial values (all channels are available for all paths)
        for source_label in self.nodes:
            for destination_label in self.nodes:
                if source_label != destination_label:
                    paths = self.find_paths(source_label, destination_label)
                    for path in paths:
                        path_string = "->".join(path)
                        row_data = [path_string] + ["Free"] * len(channels)
                        route_space_data.append(row_data)

        # Create the DataFrame
        route_space_df = pd.DataFrame(route_space_data, columns=columns)
        return route_space_df

    def create_route_space_dataframe_switching_m(self):
        # Create a DataFrame to represent the route space
        channels = list(range(10))  # Assuming 10 channels for each line
        columns = ["Path"] + [f"Channel_{channel}" for channel in channels]
        route_space_data = []

        # Fill the DataFrame with initial values (all channels are available for all paths)
        for source_label in self.nodes:
            for destination_label in self.nodes:
                if source_label != destination_label:
                    paths = self.find_paths(source_label, destination_label)
                    for path in paths:
                        path_string = "->".join(path)
                        row_data = [path_string] + ["1"] * len(channels)
                        route_space_data.append(row_data)

        # Create the DataFrame
        route_space_df = pd.DataFrame(route_space_data, columns=columns)
        return route_space_df

    def calculate_bit_rate(self,  strategy, GSNR, Rs=32e9, Bn=12.5e9, BERt=1e-3):

        GSNR_lin = 10 ** (GSNR / 10)

        # Check the transceiver strategy
        if strategy == 'fixed-rate':
            # Equation (1)
            Rb = 100 if GSNR_lin > 2 * erfcinv(2 * BERt ) * Rs / Bn else 0
        elif strategy == 'flex-rate':
            # Initialize Rb
            Rb = 0
            # Equation (2)
            if GSNR_lin < 2 * erfcinv(2 * BERt) ** 2 * Rs / Bn:
                Rb = 0
            elif 2 * erfcinv(2 * BERt) ** 2 * Rs / Bn <= GSNR_lin < 14 / 3 * erfcinv(3 / 2 * BERt) ** 2 * Rs / Bn:
                Rb = 100
            elif 14 / 3 * erfcinv(3 / 2 * BERt) ** 2 * Rs / Bn <= GSNR_lin < 10 * erfcinv(
                    8 / 3 * BERt) ** 2 * Rs / Bn:
                Rb = 200
            elif GSNR_lin >= 10 * erfcinv(8 / 3 * BERt) ** 2 * Rs / Bn:
                Rb = 400
        elif strategy == 'shannon':
            # Equation (3)
            Rb = 2 * Rs * math.log2(1 + (GSNR_lin * Rs) / Bn) // 1e9

        return Rb

    def find_paths(self, source_label, destination_label):
        def dfs(current_node, path):
            if current_node.label == destination_label:
                all_paths.append(path.copy())
                return
            visited.add(current_node.label)
            for neighbor_label, neighbor_node in current_node.successive.items():
                if neighbor_label not in visited:
                    path.append(neighbor_label)
                    dfs(neighbor_node, path)
                    path.pop()
            visited.remove(current_node.label)

        source_node = self.nodes[source_label]
        destination_node = self.nodes[destination_label]
        visited = set()
        all_paths = []
        dfs(source_node, [source_label])
        return all_paths

    def propagate(self, signal_info):
        path = signal_info.get_path()
        modified_signal_info = signal_info  # Create a copy of the original signal_info to modify

        for i in range(len(path) - 1):
            current_node_label = path[i]
            next_node_label = path[i + 1]
            line_label = current_node_label + next_node_label
            line = self.lines[line_label]

            # Update the signal information's latency and noise power based on the line's characteristics
            line_latency = line.latency_generation()
            line_noise_power = line.noise_generation(modified_signal_info.get_signal_power())

            modified_signal_info.update_latency(line_latency)
            modified_signal_info.update_noise_power(line_noise_power)

        return modified_signal_info



    def draw(self, save_image=False, image_filename='network_plot.jpg'):
        G = nx.Graph()

        # Add nodes to the graph
        for node_label, node in self.nodes.items():
            G.add_node(node.label, pos=node.position)

        # Add edges (lines) to the graph
        for line_label, line in self.lines.items():
            node1, node2 = list(line.label)
            G.add_edge(node1, node2, length=line.length)

        # Get positions for plotting
        node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}

        # Draw the network using NetworkX
        # plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=(10, 10))

        nx.draw(G, pos=node_positions, with_labels=True, node_size=2000, edge_color='white',node_color='r', font_size=10,
                font_color='white')
        fig.set_facecolor('black')
        # edge_labels = nx.get_edge_attributes(G, 'length')
        # Round the edge labels to two digits after the dot
        edge_labels = {edge: f'{length:.2f}' for edge, length in nx.get_edge_attributes(G, 'length').items()}
        nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_labels)
        plt.title('Network Visualization')
        # Save the image if save_image is True
        if save_image:
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')
        plt.show()
    def draw_beta(self, save_image=False, image_filename='network_plot.jpg'):
        G = nx.Graph()

        # Add nodes to the graph
        for node_label, node in self.nodes.items():
            G.add_node(node.label, pos=node.position)

        # Add edges (lines) to the graph
        for line_label, line in self.lines.items():
            node1, node2 = list(line.label)
            G.add_edge(node1, node2, length=line.length)

        # Get positions for plotting
        node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}

        # Draw the network using NetworkX
        # plt.figure(figsize=(10, 10))
        fig = plt.figure(figsize=(10, 10))

        nx.draw(G, pos=node_positions, with_labels=True, node_size=2000, edge_color='white',node_color='r', font_size=15,
                font_color='white')
        fig.set_facecolor('black')
        # edge_labels = nx.get_edge_attributes(G, 'length')
        # Round the edge labels to two digits after the dot
        edge_labels = {edge: f'{length:.2f}' for edge, length in nx.get_edge_attributes(G, 'length').items()}
        # nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_labels)
        plt.title('Network Visualization')
        # Add triangles to the edges
        pos = nx.get_node_attributes(G, 'pos')
        for edge in G.edges():
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            distance = math.dist((x1, y1), (x2, y2))
            num_triangles = int(distance / 80e3)  #

            for i in range(1, num_triangles + 1):
                t = i / (num_triangles + 1)
                triangle_pos = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                plt.plot(triangle_pos[0], triangle_pos[1], marker='>', markersize=25, color='orange', alpha=0.7)
        # Save the image if save_image is True
        if save_image:
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')
        plt.show()

    def draw_beta_with_amplifiers(self, save_image=False, image_filename='network_plot_with_amplifiers.jpg'):
        G = nx.Graph()

        # Add nodes to the graph
        for node_label, node in self.nodes.items():
            G.add_node(node.label, pos=node.position)

        # Add edges (lines) to the graph
        for line_label, line in self.lines.items():
            node1, node2 = list(line.label)
            G.add_edge(node1, node2, length=line.length, num_amplifiers=line.n_amplifiers)

        # Get positions for plotting
        node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}

        # Create a new figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw the network using NetworkX
        nx.draw(G, pos=node_positions, with_labels=True, node_size=2000, edge_color='white', node_color='r',
                font_size=10, font_color='white', ax=ax)

        # Draw triangles for amplifier spacing on each edge
        for edge, data in G.edges(data=True):
            length = round(data['length'])
            num_amplifiers = round(data['num_amplifiers'])
            node1, node2 = edge[:2]  # Take only the first two values from the edge tuple
            positions = np.array([node_positions[node1], node_positions[node2]])
            direction = positions[1] - positions[0]
            unit_direction = direction / np.linalg.norm(direction)
            amplifier_spacing = length / num_amplifiers

            for i in range(1, num_amplifiers):
                position = positions[0] + i * unit_direction * amplifier_spacing
                ax.plot(position[0], position[1], marker='^', markersize=10, color='blue', alpha=0.7)

        plt.title('Network Visualization with Amplifiers')
        fig.set_facecolor('black')

        # Save the image if save_image is True
        if save_image:
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')

        plt.show()

    def calculate_snr_db(self, signal_power, noise_power):
        # Calculate SNR in dB using the formula 10 * log10(signal power / noise power)
        return 10 * math.log10(signal_power / noise_power)

    def calculate_path_metrics(self, path, signal_power_watts=0.001):
        total_latency = 0.0
        total_noise_power = 0.0

        for i in range(len(path) - 1):
            current_node_label = path[i]
            next_node_label = path[i + 1]
            line_label = current_node_label + next_node_label
            line = self.lines[line_label]
            total_latency += line.latency_generation()
            total_noise_power += line.noise_generation(
                signal_power_watts)  # Calculate noise power based on signal power

        total_snr_db = self.calculate_snr_db(signal_power_watts, total_noise_power)
        return total_latency, total_noise_power, total_snr_db

    def create_paths_dataframe(self, signal_power_watts=0.001, channel=0):
        all_paths_data = []

        for source_label in self.nodes:
            for destination_label in self.nodes:
                if source_label != destination_label:
                    paths = self.find_paths(source_label, destination_label)
                    for path in paths:
                        path_string = "->".join(path)

                        # Calculate metrics considering the channel occupancy
                        total_latency, total_noise_power, total_snr_db = self.calculate_path_metrics(path, channel)

                        all_paths_data.append([path_string, total_latency, total_noise_power, total_snr_db])

        columns = ["Path", "Total Latency (s)", "Total Noise Power (W)", "SNR (dB)"]
        paths_df = pd.DataFrame(all_paths_data, columns=columns)
        return paths_df



    def find_best_snr(self, input_node_label, output_node_label):
        # Find all possible paths between the input and output nodes
        paths = self.find_paths(input_node_label, output_node_label)

        best_snr = -float("inf")
        best_path = None
        best_channel = None

        for path in paths:
            # Convert the list of node labels to a list of line labels
            lines_to_use = self.get_lines_from_path(path)

            # Check if there is at least one free channel in each line of the path
            common_free_channels = set(range(len(self.lines[lines_to_use[0]].channels)))

            for line_label in lines_to_use:
                line = self.lines[line_label]
                common_free_channels &= {channel for channel in range(len(line.channels))
                                         if line.get_channel_state(channel) == 1}

            # If there are common free channels in all lines
            if common_free_channels:
                total_latency, total_noise_power, total_snr_db = self.calculate_path_metrics(path,
                                                                                             signal_power_watts=1e-3)

                # Compare SNR to find the best path
                if total_snr_db > best_snr:
                    best_snr = total_snr_db
                    best_path = path
                    best_channel = next(iter(common_free_channels))  # Use the first common free channel

        return best_path, best_channel



    def find_best_latency(self, input_node_label, output_node_label):
        # Find all possible paths between the input and output nodes
        paths = self.find_paths(input_node_label, output_node_label)

        best_latency = float('inf')  # Initialize with positive infinity
        best_path = None
        best_channel = None

        for path in paths:
            # Convert the list of node labels to a list of line labels
            lines_to_use = self.get_lines_from_path(path)

            # Check if there is at least one free channel in each line of the path
            common_free_channels = set(range(len(self.lines[lines_to_use[0]].channels)))

            for line_label in lines_to_use:
                line = self.lines[line_label]
                common_free_channels &= {channel for channel in range(len(line.channels))
                                         if line.get_channel_state(channel) == 1}

            # If there are common free channels in all lines
            if common_free_channels:
                total_latency, _, _ = self.calculate_path_metrics(path)

                # Compare latency to find the best path
                if total_latency < best_latency:
                    best_latency = total_latency
                    best_path = path
                    best_channel = next(iter(common_free_channels))  # Use the first common free channel

        # If no valid path is found, set best_path and best_channel as empty arrays
        if best_path is None:
            best_path = []
            best_channel = None

        return best_path, best_channel

    # In the Connection class


    def stream(self, connections, label='latency'):
        if label not in ['latency', 'snr']:
            raise ValueError("Invalid label. Choose 'latency' or 'snr'.")

        for connection in connections:
            input_node_label = connection.input
            output_node_label = connection.output
            rejection_reason = None

            if label == 'latency':
                best_path, best_channel = self.find_best_latency(input_node_label, output_node_label)
            else:
                best_path, best_channel = self.find_best_snr(input_node_label, output_node_label)

            if best_path:
                all_nodes_valid = True
                if len(best_path) >= 3:
                    all_nodes_valid = self.is_path_valid(best_path, best_channel)
                if all_nodes_valid:
                    # Process the original path
                    total_latency, _, total_snr_db = self.calculate_path_metrics(best_path, connection.signal_power)

                    # Calculate the bit rate using the transceiver strategy
                    transceiver_strategy = self.nodes[input_node_label].transceiver
                    calculated_bit_rate = self.calculate_bit_rate( transceiver_strategy, total_snr_db)

                    if calculated_bit_rate > 0:
                        connection.latency = total_latency
                        connection.snr = total_snr_db

                        # Set the best channel for the connection
                        connection.set_channel(best_channel)

                        # Use the lines and channel for the connection
                        lines_to_use = self.get_lines_from_path(best_path)
                        lines_to_use_objects = [self.lines[line_label] for line_label in lines_to_use]
                        connection.use_lines_channel(lines_to_use_objects, best_channel)

                        # Assign the calculated bit rate to the connection
                        connection.set_bit_rate(calculated_bit_rate)
                    else:
                        # Bit rate is zero, reject the connection
                        # connection.snr = total_snr_db
                        best_path = []
                        connection.latency = None
                        connection.snr = connection.snr
                        connection.set_channel(None)
                        connection.set_bit_rate(0)
                        # Bit rate is zero, reject the connection
                        rejection_reason = "Bit rate is zero"
                # Switching matrix check, reject the connection
                else:
                    best_path = []
                    connection.latency = None
                    connection.snr = 0
                    connection.set_channel(None)
                    connection.set_bit_rate(0)
                    # Switching matrix check, reject the connection
                    rejection_reason = "Switching matrix check failed"
            # Best path check, reject the connection
            else:
                best_path = []
                connection.latency = None
                connection.snr = 0
                connection.set_channel(None)
                connection.set_bit_rate(0)
                # Best path check, reject the connection
                rejection_reason = "Best path check failed"

        return best_path, best_channel, rejection_reason

    def create_and_manage_connections(self, traffic_matrix):
        while np.any(traffic_matrix > 0):
            # Find a random pair of source-destination nodes with non-zero request
            src_node, dest_node = np.nonzero(traffic_matrix)
            src_node = random.choice(src_node)
            dest_node = random.choice(dest_node)

            bit_rate_request = traffic_matrix[src_node, dest_node]



    # def stream(self, connections, label='latency'):
    #     if label not in ['latency', 'snr']:
    #         raise ValueError("Invalid label. Choose 'latency' or 'snr'.")
    #
    #     for connection in connections:
    #         input_node_label = connection.input
    #         output_node_label = connection.output
    #
    #         if label == 'latency':
    #             best_path, best_channel = self.find_best_latency(input_node_label, output_node_label)
    #         else:
    #             best_path, best_channel = self.find_best_snr(input_node_label, output_node_label)
    #
    #
    #         if best_path:
    #             all_nodes_valid = True
    #             if len(best_path) >= 3:
    #                 all_nodes_valid = self.is_path_valid(best_path, best_channel)
    #             if all_nodes_valid:
    #
    #                 # Process the original path
    #                 total_latency, _, total_snr_db = self.calculate_path_metrics(best_path, connection.signal_power)
    #                 connection.latency = total_latency
    #                 connection.snr = total_snr_db
    #
    #                 # Set the best channel for the connection
    #                 connection.set_channel(best_channel)
    #
    #                 # Use the lines and channel for the connection
    #                 lines_to_use = self.get_lines_from_path(best_path)
    #                 # lines_to_use += [line[::-1] for line in lines_to_use]
    #                 lines_to_use_objects = [self.lines[line_label] for line_label in lines_to_use]
    #                 connection.use_lines_channel(lines_to_use_objects, best_channel)
    #             else:
    #                 best_path = []
    #                 connection.latency = None
    #                 connection.snr = 0
    #         else:
    #             best_path = []
    #             connection.latency = None
    #             connection.snr = 0
    #
    #     return best_path, best_channel

    def get_lines_from_path(self, path):
        # Helper method to get a list of line labels from a given path
        lines = [path[i] + path[i + 1] for i in range(len(path) - 1)]
        return lines
    def find_channels(self, source_label, destination_label):
        paths = self.find_paths(source_label, destination_label)
        channel_paths = []

        for path in paths:
            channel_path = []

            for i in range(len(path) - 1):
                current_node_label = path[i]
                next_node_label = path[i + 1]
                line_label = current_node_label + next_node_label
                line = self.lines[line_label]

                free_channels = line.get_free_channels()
                if free_channels:
                    channel_path.append({line_label: free_channels})

            channel_paths.append(channel_path)

        return channel_paths

    def probe(self, path, channel):
        """
        Propagate a SignalInformation without occupying any line and report the state for each line.

        Parameters:
        - path (list): List of node labels representing the path.
        - channel (int): Channel index.

        Returns:
        - dict: A dictionary containing line labels as keys and their states as values.
        """
        line_states = {}

        for i in range(len(path) - 1):
            node_label_1, node_label_2 = path[i], path[i + 1]
            line_label = f"{node_label_1}{node_label_2}"

            if line_label in self.lines:
                line = self.lines[line_label]
                channel_state = line.get_channel_state(channel)
                line_states[line_label] = channel_state

        return line_states




class Connection:
    def __init__(self, input_node, output_node, signal_power, channel=None):
        self.input = input_node
        self.output = output_node
        self.signal_power = signal_power
        self.latency = 0.0
        self.snr = 0.0

        self.channel = channel  # New attribute to store the channel used by the connection
        self.used_lines = []  # List to keep track of lines used by this connection
        self.bit_rate = 0

    def set_channel(self, channel):
        # Set the channel used by the connection
        self.channel = channel

    def get_channel(self):
        # Get the channel used by the connection
        return self.channel

    def use_lines(self, lines):
        for line in lines:
            line.set_channel_state(self.channel, 0)

    def release_lines(self, lines):
        for line in lines:
            line.set_channel_state(self.channel, 1)  # Clear the list of used lines

    def use_lines_channel(self, lines, channels):
        # Use the specified channels of the lines
        for line in lines:
            line.set_channel_state(channels, 0)

    def release_lines_channel(self, lines, channels):
        # Use the specified channels of the lines
        for line in lines:
            line.set_channel_state(channels, 1)

    def set_bit_rate(self, bit_rate):
        # Set the assigned bit rate
        self.bit_rate = bit_rate

    def get_bit_rate(self):
        # Get the assigned bit rate
        return self.bit_rate

    # Other methods and attributes
    def __str__(self):
        return f"Connection: Input = {self.input}, Output = {self.output}, Signal Power = {self.signal_power}, " \
               f"Latency = {self.latency}, SNR = {self.snr}, Channel = {self.channel}"


#  plt.title('Network Visualization with Paths' + self.title, color='white', weight='bold')
class PathVisualizer_1:
    def __init__(self, network, paths, title):
        self.network = network
        self.paths = paths
        self.title = title
        self.paths = paths  # Initialize paths_visu_snr as an empty list

    def visualize_paths(self, save_image=False, image_filename='network_with_paths.jpg'):
        G = nx.Graph()

        # Add nodes to the graph
        for node_label, node in self.network.nodes.items():
            G.add_node(node.label, pos=node.position)

        # Add edges (lines) to the graph
        for line_label, line in self.network.lines.items():
            node1, node2 = list(line.label)
            G.add_edge(node1, node2, length=line.length)

        # Get positions for plotting
        node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}

        # Draw the network using NetworkX
        fig = plt.figure(figsize=(5, 5))
        plt.title('Network Visualization with Paths' + self.title, color='white', weight='bold')

        nx.draw(
            G,
            pos=node_positions,
            with_labels=True,
            node_size=2000,


            edge_color='white',
            node_color='g',
            width=8 ,
            font_size=18,
            font_color='white',
            arrowsize=20,
            # connectionstyle='arc3,rad=0.1',

            edgecolors='k',

        )
        # fig.set_facecolor('purple')
        fig.set_facecolor('black')
        curvatures = np.random.rand(G.number_of_edges())

        # Draw the paths on top of the network
        for paths_list in self.paths:
            for path in paths_list:
                edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
                nx.draw_networkx_edges(G, pos=node_positions, edgelist=edges, edge_color='red', width=4, arrows=True,arrowsize=20, style='dashed',connectionstyle='arc3,rad=0.08')



        # Save the image if save_image is True

        if save_image:
            plt.savefig(image_filename, dpi=300, bbox_inches='tight')
        # plt.show()

    def draw_gif(self, save_image=False, image_filename='network_plot.jpg'):
        fig, ax = plt.subplots(figsize=(7, 7))
        G = nx.Graph()

        def update(frame):
            # Add nodes to the graph
            for node_label, node in self.nodes.items():
                G.add_node(node.label, pos=node.position)

            # Add edges up to the current frame
            for i, (line_label, line) in enumerate(self.lines.items()):
                if i <= frame:
                    node1, node2 = list(line.label)
                    G.add_edge(node1, node2, length=line.length)

            # Get positions for plotting
            node_positions = {node: data['pos'] for node, data in G.nodes(data=True)}

            # Draw the network using NetworkX
            ax.clear()
            nx.draw(G, pos=node_positions, with_labels=True, node_size=2000, node_color='b', font_size=10,
                    font_color='white')
            edge_labels = nx.get_edge_attributes(G, 'length')
            plt.title('Network Visualization - Frame {}'.format(frame))

        # Set up the animation
        animation = FuncAnimation(fig, update, frames=len(self.lines), interval=1000, repeat=False)

        # Save the animation as a GIF
        if save_image:
            animation.save(image_filename, writer='imagemagick', fps=1)

        plt.show()




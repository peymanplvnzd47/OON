## OON - Open Optical Network Simulator
Welcome to OON, the Open Optical Network Simulator! This project was developed as part of the Open Optical Network course at Politecnico di Torino. The primary objective of this simulator is to replicate the behavior of an optical network with fiber and calculate the required Generalized signal-to-noise ratio (GSNR) for each connection.

# Overview
OON utilizes a JSON configuration file representing the network's layout, including nodes and edges. It also incorporates a request matrix that generates connections randomly, allowing users to control network traffic. One of the key features of this simulator is the ability to select the type of transceivers at nodes from a predefined list, which includes fixed rate, flexible rate, and Shannon types.

# Features
Replicates the behavior of an optical network with fiber.
Calculates the GSNR required for each connection.
Utilizes a JSON configuration file for network layout and connection generation.
Offers flexibility in selecting transceiver types at nodes.
Allows users to control and generate network traffic until reaching saturation.
Getting Started
To get started with OON, follow these steps:
## Code Explanation


```python
# Define File Input
file_input = input_folder / 'network.json'

# Define Transmission Strategy
TRx_strategy = ['fixed-rate'] # shannon , flex-rate

# Initialize Network
network = Network(file_input)

# Set Transceivers for All Nodes
network.set_transceiver_for_all_nodes(TRx_strategy_indx)

# Create Route Space DataFrames
route_space_snr = network.create_route_space_dataframe()
route_space_latency = network.create_route_space_dataframe()

# Release Channels and Initialize Switching Matrices
network.release_all_channels()
network.initialize_switching_matrices()

# Update Switching Matrices
block_adjacent = False
network.update_switching_matrices(best_path, best_channel, block_adjacent)

```


<img src="https://github.com/peymanplvnzd47/OON/assets/62776383/598608da-c712-4bd2-8b23-75829edfd69f" width="600" alt="live_evolution">


![image](https://github.com/peymanplvnzd47/OON/assets/62776383/cc29afe7-e0ed-49d6-839c-fdf13d777128)



# Clone the repository to your local machine.
Install the necessary dependencies (provide instructions if any).
Run the simulator with your desired configurations.
Explore the generated results and analyze network behavior.
Usage
Include instructions on how to use the simulator, specifying how to configure the network layout, set transceiver types, generate traffic, and analyze results.

# Contribution
Contributions to OON are welcome! If you'd like to contribute, please follow these guidelines (provide guidelines if any).

# License
Specify the license under which the project is distributed.

Contact
If you have any questions or suggestions regarding OON, feel free to contact [maintainer's name/email].

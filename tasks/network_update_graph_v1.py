import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import random

# Your graph data
# graph = {
#     "A": {"connected_nodes": ["F", "G", "C", "D", "E"], "position": [477860, -212480]},
#     "B": {"connected_nodes": ["F", "G", "C", "E"], "position": [125158, 216047]},
#     "C": {"connected_nodes": ["A", "B", "D", "G"], "position": [599279, 137709]},
#     "D": {"connected_nodes": ["A", "E", "C", "G"], "position": [-44897, 66017]},
#     "E": {"connected_nodes": ["A", "B", "G", "D"], "position": [261527, -24516]},
#     "F": {"connected_nodes": ["A", "B", "G"], "position": [-189005, -206552]},
#     "G": {"connected_nodes": ["B", "C", "D", "A", "E", "F"], "position": [444888, 168774]},
# }
graph = {
    "A": {"connected_nodes": ["B", "C", "D"], "position": [-350e3, 150e3]},
    "B": {"connected_nodes": ["A", "D", "F"], "position": [-100e3, 400e3]},
    "C": {"connected_nodes": ["A", "D", "E"], "position": [-200e3, -300e3]},
    "D": {"connected_nodes": ["A", "B", "C", "E", "F"], "position": [0, 0]},
    "E": {"connected_nodes": ["C", "D", "F"], "position": [150e3, -350e3]},
    "F": {"connected_nodes": ["B", "D", "E"], "position": [300e3, 250e3]},
}



# Create a directed graph
G = nx.DiGraph()

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
fig, ax = plt.subplots(figsize=(10, 6))
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=10))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label('Edge Strength')

# Function to update edge strength and redraw the graph
def update(frame):
    global edge_strength
    # Randomly select an edge
    edge = random.choice(list(G.edges()))
    print(edge)

    # Check if the edge strength is not saturated
    if edge_strength[edge] < 10:
        # Increase the strength of the selected edge by 1
        edge_strength[edge] += 1
        #print(edge_strength)
    else:
        ani.event_source.stop()  # Stop the animation after the for loop finishes
    # Clear the current plot
    ax.clear()
    # Redraw the graph with updated edge colors
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=1000, node_color='lightblue', font_color='black', font_size=14, arrowsize=12, edge_color=[cm(edge_strength[edge]) for edge in G.edges()], ax=ax, width=5)

# Create animation
ani = FuncAnimation(fig, update, frames=20, interval=15)

# Display the plot
plt.tight_layout()
plt.show()

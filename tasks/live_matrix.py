import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# Create colormap
colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # red, yellow, green
cmap_name = 'custom_colormap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Create initial matrix
matrix = np.random.randint(0, 101, size=(9, 9))

# Create figure and heatmap
fig, ax = plt.subplots()
heatmap = sns.heatmap(matrix, annot=True, fmt='d', cmap=cm, cbar=False, square=True, ax=ax)
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Random Matrix')

# Add color bar
cbar = fig.colorbar(heatmap.collections[0], ax=ax)
cbar.set_label('Value')

# Update function
def update(frame):
    global matrix
    matrix = np.random.randint(0, 101, size=(9, 9))  # Update all cells with new random values
    ax.clear()
    heatmap = sns.heatmap(matrix, annot=True, fmt='d', cmap=cm, cbar=False, square=True, ax=ax)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Random Matrix (Live Update)')
    return heatmap,

# Create animation
ani = FuncAnimation(fig, update, frames=10, blit=True, interval=200)

plt.show()

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
matrix = np.zeros((9, 9))  # Initialize with zeros

# Create figure and heatmap
fig, ax = plt.subplots()
heatmap = sns.heatmap(matrix, annot=True, fmt='.1f', cmap=cm, cbar=False, square=True,
                      ax=ax)  # Use '.1f' format for floats
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Random Matrix')

# Add color bar
cbar = fig.colorbar(heatmap.collections[0], ax=ax)
cbar.set_label('Value')

# Counter for the for loop
counter = 100


# Update function
# Update function
def update(frame):
    global matrix, counter

    # Generate new random values for all cells
    if counter > 0:
        matrix = np.random.randint(0, 101, size=(9, 9))
        counter -= 1
    else:
        ani.event_source.stop()  # Stop the animation after the for loop finishes

    ax.clear()
    heatmap = sns.heatmap(matrix, annot=True, fmt='.1f', cmap=cm, cbar=False, square=True,
                          ax=ax)  # Use '.1f' format for floats

    # Reset xlabel and ylabel
    plt.xlabel('Column')
    plt.ylabel('Row')

    # Reset title
    plt.title('Random Matrix (Live Update)')

    return heatmap,


# Create animation
ani = FuncAnimation(fig, update, frames=counter, interval=200)

plt.show()

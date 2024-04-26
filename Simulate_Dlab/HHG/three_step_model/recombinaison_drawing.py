import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, np.pi/2, 100)
x2 = np.linspace(np.pi/2, np.pi, 100)
x3 = np.linspace(np.pi, 3*np.pi/2, 100)
x4 = np.linspace(3*np.pi/2, 2*np.pi, 100)
x5 = np.linspace(2*np.pi, 5*np.pi/2, 100)
x6 = np.linspace(5*np.pi/2, 3*np.pi, 100)

y1 = np.cos(x1)
y2 = np.cos(x2)
y3 = np.cos(x3)
y4 = np.cos(x4)
y5 = np.cos(x5)
y6 = np.cos(x6)

fig, ax = plt.subplots()

# Plot the data
ax.plot(x1, y1, color='blue')
ax.plot(x2, y2, color='red')
ax.plot(x3, y3, color='blue')
ax.plot(x4, y4, color='red')
ax.plot(x5, y5, color='blue')
ax.plot(x6, y6, color='red')

# Set y-limits to make the x-axis visible
ax.set_ylim(-1.1, 1.1)

# Move the x-axis to the y=0 position by setting its position
ax.spines['bottom'].set_position('zero')
ax.tick_params(right=False, top=False, which='both', direction='out')

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set custom tick positions and labels for the x-axis
y_tick_positions = [-1,0,1]
x_tick_positions = [np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 5*np.pi/2, 3*np.pi]
x_tick_labels = [r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$', r'$\frac{5\pi}{2}$', r'$3\pi$']
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels)
ax.set_yticks(y_tick_positions)
plt.xlim([0,3*np.pi])
# Add a legend
plt.show()

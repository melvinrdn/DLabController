from matplotlib.colors import LinearSegmentedColormap

def white_turbo(segments=512):
    colors = [
        (1, 1, 1),  # white
        (0, 0, 0.5),  # dark blue
        (0, 0, 1),  # clearer blue
        (0, 1, 1),  # turquoise
        (0, 1, 0),  # green
        (1, 1, 0),  # yellow
        (1, 0.5, 0),  # orange
        (1, 0, 0),  # red
        (0.5, 0, 0)  # darker red
    ]
    return LinearSegmentedColormap.from_list('white_turbo', colors, N=segments)
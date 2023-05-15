import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np


def draw_polygon(ax, fig):
    inv = []
    inv.append(ax.transData.inverted())
    coords = []
    codes = []
    l, = plt.plot(coords, marker='o')
    endclick = [None]
    curr_pos = [0, 0]

    def resize(event):
        inv[0] = ax.transData.inverted()

    def press(event):
        curr_pos[0] = event.x
        curr_pos[1] = event.y

    def release(event):
        inv[0] = ax.transData.inverted()
        if event.dblclick:
            endclick[0] = True
            return

        x, y = event.x, event.y
        if x != curr_pos[0] or y != curr_pos[1]:
            return

        print(x, y, inv[0].transform((x, y)))
        coords.append(inv[0].transform((x, y)))
        if len(codes) == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)

        xdata = np.append(l.get_xdata(), coords[-1][0])
        ydata = np.append(l.get_ydata(), coords[-1][1])
        l.set_data(xdata, ydata)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', press)
    cid2 = fig.canvas.mpl_connect('button_release_event', release)
    cid3 = fig.canvas.mpl_connect('resize_event', resize)
    w = None
    for i in range(4):
        w = plt.waitforbuttonpress(timeout=5)

    coords.append(coords[0])
    codes.append(Path.CLOSEPOLY)

    xdata = np.append(l.get_xdata(), coords[-1][0])
    ydata = np.append(l.get_ydata(), coords[-1][1])
    l.set_data(xdata, ydata)

    path = Path(coords, codes)
    patch = patches.PathPatch(path, facecolor='orange', lw=2, alpha=0.2)
    ax.add_patch(patch)
    plt.draw()
    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(cid2)
    fig.canvas.mpl_disconnect(cid3)
    return path

#
#
# if __name__=='__main__':
#
#     fig = plt.figure()
#     ax=plt.gca()
#     l,=plt.plot([1,3,2,5,-1])
#     path=draw_polygon(ax)
#
#     print(path.contains_points(np.array(l.get_data()).T))

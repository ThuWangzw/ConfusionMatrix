from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import os

# dir_path = './data/reordering/proj_figs/'
dir_path = './backend/data/reordering/proj_figs/'

def draw_plot(points, file_name='test.png'):
    if type(points) == list:
        points = np.array(points)
    plt.plot(points[:, 0], points[:, 1], 'c*-')
    plt.show()
    plt.savefig(os.path.join(dir_path, file_name))

def draw_directed_plot(points, file_name='test.png', index=None, note=None, classes=None):
    show_class = 4
    font_size = 12
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')

    if type(points) == list:
        points = np.array(points)
    for i in range(len(points)):

        plt.scatter(points[i, 0], points[i, 1], c= plt.cm.Set3(classes[i]))
    # plt.plot(points[:, 0], points[:, 1], 'c*-')
    if index==None:
        index = list(range(len(points)))
    for i in range(len(points)-1):
        start = points[index[i]]
        end = points[index[i+1]]

        # if classes[index[i]]==show_class:
        plt.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]), arrowprops=dict(arrowstyle='->', color='grey', alpha=1))

        # if note and classes[index[i]]==show_class:
        plt.text(start[0], start[1], s=note[i], fontsize=font_size, color='grey')
        if i==len(points)-2:
            plt.text(end[0], end[1], s=note[i+1], fontsize=font_size, color='grey')

    plt.title(file_name)
    plt.savefig(os.path.join(dir_path, file_name))

if __name__ == "__main__":
    draw_plot()
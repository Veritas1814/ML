import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_voxels(voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    filled = np.argwhere(voxels > 0)
    ax.scatter(filled[:,0], filled[:,1], filled[:,2], zdir='z', c='blue', marker='s')
    plt.show()
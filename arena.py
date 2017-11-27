#!usr/bin/env

'''
This module provides an abstraction of an arena within which Antuinos will be
simulated.
'''

# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

class Arena():
    '''
    Provides an abstraction of the environment where Antuinos operate and tools
    to visualise it.
     '''

    def __init__(self, size=(100, 100)):
        self.size = size
        self.dim = len(self.size)
        self._goals = [size]
        self.directions = np.array(np.concatenate(
            (np.identity(self.dim), -1 * np.identity(self.dim))))

        for idx, goal in enumerate(self._goals):
            if goal == (-1, -1):
                self._goals[idx] = size

        # Constants
        self.GOAL_RADIUS = 10
        self.GOAL_COLOR = 'green'

    def draw_sphere(self, axes, centre=(0, 0, 0), radius=None, color='g', transparency=0.3):
        if radius is None:
            radius = self.GOAL_RADIUS
        # Make data
        u = np.linspace(0, 2 * np.pi, radius / self.GOAL_RADIUS * 20)
        v = np.linspace(0, np.pi, radius / self.GOAL_RADIUS * 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + centre[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + centre[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centre[2]

        # Plot the surface
        axes.plot_surface(x, y, z, color=color, alpha=transparency)

    def figurize_arena(self, axes):
        '''
        Draws a visual representation of the arena in the provided axes.
        '''

        if self.dim == 2:
            circles = []
            # Goal circles
            for goal in self._goals:
                circles.append(plt.Circle(goal,
                                          radius=self.GOAL_RADIUS,
                                          fc=self.GOAL_COLOR))
            for circle in circles:
                axes.add_patch(circle)

        elif self.dim == 3:
            # Draw goals
            for goal in self._goals:
                ret = self.draw_sphere(axes, centre=goal)

            # Make sure things look right
        if self.dim == 2:
            axes.set_xlim((0, self.size[0]))
            axes.set_ylim((0, self.size[1]))
        elif self.dim == 3:
            axes.set_xlim3d((0, self.size[0]))
            axes.set_ylim3d((0, self.size[1]))
            axes.set_zlim3d((0, self.size[2]))

        axes.grid(True)
        axes.set_aspect(1)


# Module-run code (used for testing)
if __name__ == '__main__':
    print('Starting')
    size = (300, 300, 300)
    arena = Arena(size=size)
    fig = plt.figure()

    if len(size) == 2:
        axes = plt.gca()
    elif len(size) == 3:
        axes = fig.add_subplot(111, projection='3d')
    arena.figurize_arena(axes)
    plt.show()

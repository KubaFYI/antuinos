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
import pdb


class Arena():
    '''
    Provides an abstraction of the environment where Antuinos operate and tools
    to visualise it.
     '''

    def __init__(self, size=(100, 100)):
        self.size = size
        self.dim = len(self.size)
        self.goals = np.array([size], dtype=np.float)
        self.directions = np.array(np.concatenate(
            (np.identity(self.dim), -1 * np.identity(self.dim))))
        self.opposite_dirs = np.empty(self.directions.shape[0], dtype=np.int)
        self.side_dirs = [[] for _ in range(self.directions.shape[0])]
        # pdb.set_trace()
        for i, dire1 in enumerate(self.directions):
            for j, dire2 in enumerate(self.directions):
                if (dire1 == -1 * dire2).all():
                    self.opposite_dirs[i] = j
                elif (dire1 != dire2).any():
                    self.side_dirs[i].append(j)
        self.side_dirs = np.array(self.side_dirs, dtype=np.int)

        for idx, goal in enumerate(self.goals):
            if len(goal) == 2 and (goal == (-1, -1)).all():
                self.goals[idx] = size

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
            for goal in self.goals:
                circles.append(plt.Circle(goal,
                                          radius=self.GOAL_RADIUS,
                                          fc=self.GOAL_COLOR))
            for circle in circles:
                axes.add_patch(circle)

        elif self.dim == 3:
            # Draw goals
            for goal in self.goals:
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

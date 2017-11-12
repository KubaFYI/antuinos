#!usr/bin/env

'''
This module provides an abstraction of an arena within which Antuinos will be
simulated.
'''

# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# Define constants (arena numpy array key)
START = -1
GOAL = -2
OBSTACLE = -3
EMPTY = 0

# Constants used in drawing
COLOR = {START: 'purple',
         GOAL: 'red',
         OBSTACLE: 'black',
         'ant': 'green'}

STRT_PNT_RD = 5
GL_PNT_RD = 5

# Module code


class Arena():
    '''
    Provides an abstraction of the environment where Antuinos operate and tools
    to visualise it.

    The arena is defined as a 2D grid of non-negative-integer-indexed cells.
    One cell must be defined as a 'starting point' (the origin being the
    default) and at least one cell must be defined as a 'goal' (the opposite
    corner of the arena for default). Each arena can contain obstacles defined
    as a list of strings containing arithmetic conditions on cell indexes
    specified as (x, y). Should a cell fulfil any of them it will count as an
    obstacle.
     '''

    def __init__(self, size=(100, 100), start_point=(0, 0), goals=[(-1, -1)],
                 obstacles=[], file=None):
        if file is not None:
            (self._size, self.start_point,
             self._goals, self._obstacles) = self._parse_arena_file(file)
        else:
            self._size = size
            self.start_point = start_point
            self._goals = goals
            for idx, goal in enumerate(self._goals):
                if goal == (-1, -1):
                    self._goals[idx] = size
            self._obstacles = obstacles

        self.narena = self._gen_np_arena()

        # Create transparent-black color map for displaying water
        self._colormap = plt.cm.binary(np.arange(plt.cm.binary.N))
        self._colormap[:, -1] = np.linspace(0, 1, plt.cm.binary.N)
        self._colormap = ListedColormap(self._colormap)

    def _gen_np_arena(self):
        '''
        Create and return a numpy array representing the arena. In its first
        dimention positive values signify occupancy, negative values indicate
        points of interest. The second dimention is intened to hlold the
        state of the Antuino-generated markers.
        '''
        narena = np.zeros((2,) + self._size)
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                if (x, y) == self.start_point:
                    narena[(0, x, y)] = START
                elif (x, y) in self._goals:
                    narena[(0, x, y)] = GOAL
                else:
                    is_obstacle = False
                    for cond in self._obstacles:
                        if eval(cond):
                            is_obstacle = True
                            break
                    narena[(0, x, y)] = OBSTACLE if is_obstacle else EMPTY
        return narena

    def figurize_arena(self, axes):
        '''
        Draws a visual representation of the arena in the provided axes.
        '''
        axes.grid(False)
        circles = []

        # Start circle
        circles.append(plt.Circle((self.start_point),
                                  radius=STRT_PNT_RD,
                                  fc=COLOR[START], zorder=1))
        # Goal circles
        for goal in self._goals:
            circles.append(plt.Circle(goal,
                                      radius=GL_PNT_RD,
                                      fc=COLOR[GOAL]))
        for circle in circles:
            axes.add_patch(circle)

        # Draw obstacles
        # axes.pcolormesh(np.flipud(np.rot90(self.narena[0,...])) == OBSTACLE)
        axes.pcolormesh((self.narena[0, ...]).T == OBSTACLE,
                        cmap=self._colormap,
                        zorder=2)

        # Make sure things look right
        axes.set_xlim((0, self._size[0]))
        axes.set_ylim((0, self._size[1]))
        axes.grid(False)
        # axes.invert_xaxis()
        # axes.invert_yaxis()
        axes.set_aspect(1)

        # Draw the marker traces
        # axes.pcolor(self.narena[1,...])

    def _parse_arena_file(self, filename):
        '''
        Parse a file describing arena geometry.
        '''
        # TODO
        raise NotImplementedError

    def are_valid_positions(self, positions):
        valid_positions = positions[:, 0] > 0
        valid_positions *= positions[:, 0] < self._size[0]
        valid_positions *= positions[:, 1] > 0
        valid_positions *= positions[:, 1] < self._size[1]
        return valid_positions


# Module-run code (used for testing)
if __name__ == '__main__':
    print('Starting')
    arena = Arena(obstacles=(['x >= 15 and x < 20 and y >= 10 and y < 30']))
    plt.figure()
    axes = plt.gca()
    arena.figurize_arena(axes)
    plt.show()

    pass

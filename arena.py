#!usr/bin/env

'''
This module provides an abstraction of an arena within which Antuinos will be
simulated.
'''

# Imports
import numpy as np
from matplotlib import pyplot as plt



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
            (self._size, self._start_point,
             self._goals, self._obstacles) = self._parse_arena_file(file)
        else:
            self._size = size
            self._start_point = start_point
            self._goals = goals
            for idx, goal in enumerate(self._goals):
                if goal == (-1, -1):
                    self._goals[idx] = size
            self._obstacles = obstacles

        self._drawing = drawing_ready
        if self._drawing:
            self._figure = plt.figure()
            self._axes = plt.gca()

        self.narena = self._gen_np_arena()

    def _gen_np_arena(self):
        '''
        Create and return a numpy array representing the arena. In its first
        dimention positive values signify occupancy, negative values indicate
        points of interest. The second dimention is intened to hlold the
        state of the Antuino-generated markers.
        '''
        narena = np.zeros((1,) + self._size)
        for x in range(self._size[0]):
            for y in range(self._size[1]):
                if (x, y) == self._start_point:
                    narena[(0, x, y)] = START
                elif (x, y) in self._goals:
                    narena[(0, x, y)] = GOAL
                else:
                    is_obstacle = False
                    for cond in self._obstacles:
                        if eval(cond):
                            print('pow!')
                            is_obstacle = True
                            break
                    narena[(0, x, y)] = OBSTACLE if is_obstacle else EMPTY
        return narena

    def _figurize_arena(self, axes):
        '''
        Draws a visual representation of the arena in the provided axes.
        '''
        axes.grid(False)
        circles = []

        # Start circle
        circles.append(plt.Circle((self._start_point),
                                  radius=STRT_PNT_RD,
                                  fc=COLOR[START]))
        # Goal circles
        for goal in self._goals:
            circles.append(plt.Circle(goal,
                                      radius=GL_PNT_RD,
                                      fc=COLOR[GOAL]))
        for circle in circles:
            axes.add_patch(circle)

        # Draw obstacles
        print((self.narena[0,...] == OBSTACLE).any())
        # axes.pcolormesh(np.flipud(np.rot90(self.narena[0,...])) == OBSTACLE)
        axes.pcolormesh((self.narena[0,...]).T == OBSTACLE, cmap='binary')

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

# Module-run code (used for testing)
if __name__ == '__main__':
    print('Starting')
    arena = Arena(obstacles=(['x >= 15 and x < 20 and y >= 10 and y < 30']))
    plt.figure()
    axes = plt.gca()
    arena._figurize_arena(axes)
    plt.show()

    pass


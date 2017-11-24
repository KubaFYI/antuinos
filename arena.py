#!usr/bin/env

'''
This module provides an abstraction of an arena within which Antuinos will be
simulated.
'''

# Imports
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pdb


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

    def _gen_coord_gird_circle(self, cntr, radius):
        '''
        Returns a list of integer coordinates whose centres lay within a circle
        of given parameters.
        '''
        retvall = []
        for x in range(radius + 2):
            for y in range(radius + 2):
                if x**2 + y**2 < radius**2 + 2:
                    retvall.append([(cntr[0] + x),
                                    (cntr[1] + y)])
                    retvall.append([(cntr[0] - x),
                                    (cntr[1] - y)])
                    retvall.append([(cntr[0] + x),
                                    (cntr[1] - y)])
                    retvall.append([(cntr[0] - x),
                                    (cntr[1] + y)])
        retvall = np.array(retvall)
        retvall[:, 0] %= self._size[0]
        retvall[:, 1] %= self._size[1]
        return retvall

    def _gen_coord_gird_rectangle(self, lower_left, upper_right):
        '''
        Returns a list of integer coordinates whose centres lay within a rectangle
        of given parameters.
        '''
        width = upper_right[0] - lower_left[0]
        height = upper_right[1] - lower_left[1]
        # pdb.set_trace()
        retvall = np.indices((width, height)).reshape(2, width * height).T
        retvall[:, 0] += lower_left[0]
        retvall[:, 1] += lower_left[1]
        retvall[:, 0] %= self._size[0]
        retvall[:, 1] %= self._size[1]
        return retvall

    def _gen_np_arena(self):
        '''
        Create and return a numpy array representing the arena. In its first
        dimention positive values signify occupancy, negative values indicate
        points of interest. The second dimention is intened to hlold the
        state of the Antuino-generated markers.
        '''
        narena = np.zeros((4,) + self._size)

        home_points = self._gen_coord_gird_circle(
            self.start_point, STRT_PNT_RD)
        for goal in self._goals:
            goal_points = self._gen_coord_gird_circle(goal, GL_PNT_RD)

        obstacle_points = None
        for obst in self._obstacles:
            if obst[0] == 'rect':
                cur_obst = self._gen_coord_gird_rectangle(obst[1], obst[2])
            elif obst[0] == 'circ':
                cur_obst = self._gen_coord_gird_circle(obst[1], obst[2])

            if obstacle_points is None:
                obstacle_points = cur_obst
            else:
                obstacle_points = np.concatenate((obstacle_points, cur_obst))

        narena[0, obstacle_points[:, 0], obstacle_points[:, 1]] = 1
        narena[2, home_points[:, 0], home_points[:, 1]] = 1
        narena[3, goal_points[:, 0], goal_points[:, 1]] = 1

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
        axes.pcolormesh((self.narena[0, ...]).T == 1,
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
        return self.narena[0, positions[:, 0], positions[:, 1]] != 1


# Module-run code (used for testing)
if __name__ == '__main__':
    print('Starting')
    arena = Arena(obstacles=[['rect', (20, 30), (20, 45)],
                             ['circ', (75, 75), 8]])
    plt.figure()
    axes = plt.gca()
    arena.figurize_arena(axes)
    plt.show()

    pass

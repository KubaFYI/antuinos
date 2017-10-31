#!usr/bin/env

'''
This module models a single agent in the Antuino simulation.
There are some pretty non-pythonic things here, especially in he way I choose
to represent certain variables but this is anticipating work on performance
improvement.
'''

# Imports
import numpy as np
import arena
from matplotlib import pyplot as plt

# Variables for indecies of gradient indication


class Ant():
    '''
    Class used to generate a single agent in the Antuino simulation. Contains
    each agent's internal state, absolute position, manages its sensory input
    delivery and decision making.

    Each Ant can try to exectute the following actions (planned future actions
    mared with an asterisk *):
        - Move North
        - Move South
        - Move East
        - Move West
        - Attach*
        - Detach*
        - Marker Laying On*
        - Marker Laying Off*

    Each Ant is capable of taking in the folowing sensory inputs:
        - 'Pheromone' marker strengh postitive gradient -> one of four values
                        (N/S/W/E) plus diagonal combinations (e.g. NE but not
                        NS) or an indication of a local maximum (only negative
                        gradients) as well as the magnitude.
        - Local vision -> An indicator if the currently occupied cell is a:
                        - Home
                        - Goal
                        - Obstacle
                        - empty
        - Out vision -> an binary indicator for each of the eight cells
                        surrounding agent's own cell communicating if it's
                        accesible or not
    '''
    def __init__(self, arena, start_position=(0, 0), decision_cb=None):
        self._narena = arena
        self._position = start_position

        self._senses = {}
        self._state = None
        self._decision_cb = decision_cb
        self._decision = None

        self._rel_coord = [(-1, 1), (0, 1), (1, 1), (-1, 0),
                           (1, 0), (-1, -1), (0, -1), (1, -1)]

    def where_am_i(self):
        '''
        Extract the type of the cell at current location
        '''
        if self._narena[(0,) + self._position] < 0:
            # Cell of interest
            return self._narena[(0,) + self._position]
        else:
            # Empty field
            return 0

    def what_do_i_see(self):
        '''
        Extract information about the occupancy of the surroundings of the
        agent.

        returns a 1D array where each element is a bool signifying presence of
        an obstacle, ordered as such
        _______
        |0|1|2|
        |3|X|4|
        |5|6|7|

        In cases of being close to the arena edge -> Treat as obstacle
        -------
        '''
        result = np.empty(8, dtype=bool)
        # Handle cases of being close to the arena edge -> Treat as obstacle
        for i in range(result.shape[0]):
            x = self._positon[0] + self._rel_coord[i][0]
            y = self._positon[1] + self._rel_coord[i][1]

            # Check if at the arena edge
            if x < 0 or y < 0 or \
               x >= self._narena.shape[0] or y >= self._narena.shape[1]:
                result[i] = True
            else:
                result[i] = self._narena[0, x, y] == arena.OBSTACLE

    def what_do_i_smell(self):
        '''
        Extract information about the gradient of the pheromone marker at the
        current cell.

        The function returns a tuple of two values: the first one is the
        magnitude of the largest positive gradient and the second is a
        8-element list indicating the direction of the gradient.
        '''
        marker_mag_here = self._narena[1, x, y]
        strongest_dir = None
        strongest_pos_grad = 0
        for i in range(result.shape[0]):
            x = self._positon[0] + self._rel_coord[i][0]
            y = self._positon[1] + self._rel_coord[i][1]
            marker_mag = self._narena[1, x, y] - marker_mag_here
            if i in [0, 2, 5, 7]:
                divider = np.sqrt(2)
            else:
                divider = 1
            if (marker_mag - marker_mag_here)/divider > strongest_pos_grad:
                strongest_pos_grad = (marker_mag - marker_mag_here)/divider
                strongest_dir = i

        result_dirs = [False] * 8
        result_dirs[strongest_dir] = True

        return strongest_pos_grad, result_dirs

    def sense(self):
        '''
        Update sensory input.
        '''
        self._senses['loc_vis'] = self.where_am_i()
        self._senses['out_vis'] = self.what_do_i_see()
        self._senses['marker'] = self.what_do_i_smell()

    def decide(selfs):
        '''
        Chooses an action to perform from available sensory information by
        invoking an appropriate callback.
        '''
        self.sense()
        self._decisions = self._decision_cb(self._senses, self._state)

    def update_position(self, new_postion):
        '''
        Function used by the simulator to let ant know of a new position.
        '''
        self._positon = new_postion

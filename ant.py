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
from enum import Enum


class Decision(Enum):
    GO_N = 0
    GO_S = 1
    GO_E = 2
    GO_W = 3
    # ATTACH = 4
    # DETACH = 5
    MARKER_SPOT = 6
    DO_NOTHING = 7


class ANTS_IDX(Enum):
    IDX = 0
    POS = 1
    SENS = 2
    DEC = 3


motions = [Decision.GO_N.value, Decision.GO_S.value,
           Decision.GO_E.value, Decision.GO_W.value]


# Variables for directions
class Direction(Enum):
    NW = 0
    N = 1
    NE = 2
    W = 3
    E = 4
    SW = 5
    S = 6
    SE = 7


class Senses(Enum):
    ON_OBSTACLE_EDGE = 0
    CARRYING_FOOD = ON_OBSTACLE_EDGE + 1
    PHEROMONE_GRAD = CARRYING_FOOD + 1
    END = PHEROMONE_GRAD + len(Direction) / 2
    # VISION = PHEROMONE_GRAD + len(Direction) / 2
    # END = VISION + len(Direction)


sense_idx = {'on_obst_edge': (0, 1),
             'carry_food': (1, 2),
             'phero_grad': (2, 2 + len(Direction) / 2)}

# Numpy datatype representing single agent data
default_dtype = np.float64
position_dim = 2
senses_dim = int(1 + 1 + len(Direction) / 2)
decisions_dim = len(Decision)
directional_senses = 1
half_directional_senses = 1

ant_dtype = np.dtype([('position', np.int, position_dim),
                      ('senses', default_dtype, senses_dim),
                      ('decisions', np.bool, decisions_dim)])

RELATIVE_COORDS = [(-1, 1), (0, 1), (1, 1), (-1, 0),
                   (1, 0), (-1, -1), (0, -1), (1, -1)]

# Catalogue of decision-making functions
DECISION_FNCTN = {'random': 'decision_random',
                  'nothing': 'decision_do_nothing',
                  'linear': 'decision_linear'}


class Ants():
    '''
    Class used to handle all the agents in the Antuino simulation. Contains
    each agent's internal and external state. And handles production of
    decision data out of agents' sensory inputs.

    Each Ant can try to execute the following actions (planned future actions
    marked with an asterisk *):
        - Move North
        - Move South
        - Move East
        - Move West
        - Attach*
        - Detach*
        - Marker Laying On*
        - Marker Laying Off*

    Each Ant is capable of taking in the flowing sensory inputs:
        - 'Pheromone' marker strength positive gradient -> one of four values
                        (N/S/W/E) plus diagonal combinations (e.g. NE but not
                        NS) or an indication of a local maximum (only negative
                        gradients) as well as the magnitude.
        - Occupancy -> An indicator if the currently occupied cell is a:
                        - Home
                        - Goal
                        - Obstacle
                        - empty
        - Out vision -> an binary indicator for each of the eight cells
                        surrounding agent's own cell communicating if it's
                        accesible or not

    self._ants is an array of shape (N, ...) containing a matrix of states for
    each one of N agents. The state matrices themselves are sub-arrays of shape
    (Pos, Se, De) where Pos is the position of an agent (2 elements x and y),
    Se is the input of the sensory apparatus at that given moment (8 elements
    for Pheromone Gradient Indicator and Out Vision (each) and 4 elements for
    the Occupancy indicator, 20 elements in total), and De is the current
    decision matrix (9 elements in total).
    '''

    def __init__(self, narena, start_positions=[[0, 0]], decision_mode=None):
        self._narena = narena
        self._position_dim = 2
        self._directional_senses = 2
        self._senses_dim = senses_dim
        self._decisions_dim = decisions_dim

        self.ant_no = start_positions.shape[0]
        self._positions = np.array(start_positions)
        self._senses = np.random.rand(
            len(start_positions), self._senses_dim).astype(default_dtype)
        self._decisions = np.empty((len(start_positions), self._decisions_dim),
                                   dtype=default_dtype)

        if decision_mode is not None:
            self._decision_cb = eval('self.' + DECISION_FNCTN[decision_mode])
        else:
            self._decision_cb = self.decision_do_nothing

        self.lin_dec_mat = None

    def where_do_i_stand(self):
        '''
        Extract the type of the cell at current location
        '''
        ants_on_home = self._narena[arena.HOME_LAYER,
                                    self._positions[:, 0],
                                    self._positions[:, 1]] == 1
        ants_on_goal = self._narena[arena.GOAL_LAYER,
                                    self._positions[:, 0],
                                    self._positions[:, 1]] == 1

        self._senses[ants_on_home, sense_idx['carry_food'][0]] = 0
        self._senses[ants_on_goal, sense_idx['carry_food'][0]] = 1

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
        pass
        # result = np.empty(8, dtype=bool)
        # # Handle cases of being close to the arena edge -> Treat as obstacle
        # for i in range(result.shape[0]):
        #     x = self.position[0] + RELATIVE_COORDS[i][0]
        #     y = self.position[1] + RELATIVE_COORDS[i][1]

        #     # Check if at the arena edge
        #     if x < 0 or y < 0 or \
        #        x >= self._narena.shape[0] or y >= self._narena.shape[1]:
        #         result[i] = True
        #     else:
        #         result[i] = self._narena[0, x, y] == arena.OBSTACLE

    def what_do_i_smell(self):
        '''
        Extract information about the gradient of the pheromone marker at the
        current cell.

        The function modifies the sensory input array
        '''

        grad_thresh = 0

        marker_mag_here = self._narena[1, self._positions[:, 0],
                                       self._positions[:, 1]]
        strng_dir = np.zeros(self.ant_no, dtype=default_dtype) - 1
        # strng_pos_grad = np.zeros(self.ant_no, dtype=default_dtype)
        for i in range(len(Direction) / 2):
            x = self._positions[:, 0] + RELATIVE_COORDS[i / 2 + 1][0]
            y = self._positions[:, 1] + RELATIVE_COORDS[i / 2 + 1][1]
            diff_marker_mag = self._narena[1, x, y] - marker_mag_here
            change_idx = diff_marker_mag > grad_thresh
            # strng_pos_grad[change_idx] = strng_pos_grad[diff_marker_mag]
            strng_dir[change_idx] = i

        self._senses[:, sense_idx['phero_grad'][0]:sense_idx['phero_grad'][1]] = np.zeros(
            self.ant_no, len(Direction) / 2)
        self._senses[:, sense_idx['phero_grad'][0] + strng_dir] = True

    def sense(self):
        '''
        Update sensory input.
        '''
        self.where_do_i_stand()
        # self.what_do_i_see()
        # self.what_do_i_smell()
        self._senses = np.random.randint(0, 100, self._senses.shape)

    def decide(self):
        '''
        Chooses an action to perform from available sensory information by
        invoking an appropriate callback.
        '''
        self.sense()
        self._decision_cb()

    def decision_random(self):
        '''
        Attribute
        '''
        self._decisions = np.random.random(self._decisions.shape).argsort(1)
        self._decisions = (self._decisions == self._decisions.shape[1] - 1)

    def decision_do_nothing(self):
        '''
        Do nothing
        '''
        self._decisions = np.zeros_like(self._decisions)

    def decision_linear(self):
        '''
        Do nothing
        '''
        decision_strenghts = self.lin_dec_mat.dot(self._senses.T).T
        normalizers = np.max(decision_strenghts, axis=1)
        self._decisions = np.expand_dims(normalizers, 1) == decision_strenghts

    def set_decision_matrix(self, decision_matrix):
        '''
        Sets the linear decision matrix to be used for decisionmaking.
        The matrix must be of size Senses x Decisions
        '''
        self.lin_dec_mat = decision_matrix

    def update_positions(self, new_postions, valid_positions):
        '''
        Function used by the simulator to let ant know of a new position.
        '''
        self._positions[np.argwhere(valid_positions)] = \
            new_postions[np.argwhere(valid_positions)]

    def get_agent_number(self):
        '''
        Return the number of agents present.
        '''
        return self._positions.shape[0]

#!usr/bin/env

'''
This module models a single agent in the Antuino simulation.
There are some pretty non-pythonic things here, especially in he way I choose
to represent certain variables but this is anticipating work on performance
improvement.
'''

# Imports
import numpy as np

# Catalogue of decision-making functions
DECISION_FNCTN = {'random': 'decision_random',
                  'nothing': 'decision_do_nothing',
                  'linear': 'decision_linear'}

default_dtype = np.float64


class Ants():
    '''
    Class used to handle all the agents in the Antuino simulation. Contains
    each agent's internal and external state. And handles production of
    decision data out of agents' sensory inputs.
    '''

    def __init__(self, arena, ant_no, decision_mode=None):
        self.arena = arena
        self.ant_no = ant_no

        # Sensor on each side of the agent plus food detector
        self._senses_dim = self.arena.directions.shape[0] + 1

        # Move any direction plus send signal plus do nothing
        self._actions_dim = self.arena.directions.shape[0] + 1 + 1
        self._action_signal_idx = self.arena.directions.shape[0] + 0
        self._action_do_nothing_idx = self.arena.directions.shape[0] + 1

        self._positions = np.empty((self.ant_no, self.arena.dim),
                                   dtype=default_dtype)
        self._senses = np.empty((self.ant_no, self._senses_dim),
                                dtype=default_dtype)
        self._actions = np.empty((self.ant_no, self._actions_dim),
                                   dtype=default_dtype)

        if decision_mode is not None:
            self._decision_cb = eval('self.' + DECISION_FNCTN[decision_mode])
        else:
            self._decision_cb = self.decision_do_nothing

        self.lin_dec_mat = None

    def sense(self):
        '''
        Update sensory input.
        '''
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
        self._actions = np.random.random(self._actions.shape).argsort(1)
        self._actions = (self._actions == self._actions.shape[1] - 1)

    def decision_do_nothing(self):
        '''
        Do nothing
        '''
        self._actions = np.zeros_like(self._actions)
        self._actions[:, self._action_do_nothing_idx] = 1

    def decision_linear(self):
        '''
        Do nothing
        '''
        if len(self.lin_dec_mat.shape) > 2:
            # Each ant has it's own decision making brain
            decision_strenghts = np.empty((self.ant_no, self._actions_dim))
            # import pdb; pdb.set_trace()
            for i in range(self.ant_no):
                decision_strenghts[i] = self.lin_dec_mat[i].dot(
                    self._senses[i])
        else:
            decision_strenghts = self.lin_dec_mat.dot(self._senses.T).T
        normalizers = np.max(decision_strenghts, axis=1)
        self._actions = np.expand_dims(normalizers, 1) == decision_strenghts

    def set_decision_matrix(self, decision_matrix):
        '''
        Sets the linear decision matrix to be used for decision-making.
        The matrix must be of size Senses x Decisions
        '''
        self.lin_dec_mat = decision_matrix

    def update_positions(self, new_postions, valid_positions=None):
        '''
        Function used by the simulator to update ant positions.
        '''
        if valid_positions is not None:
            self._positions[np.argwhere(valid_positions)] = \
                new_postions[np.argwhere(valid_positions)]
        else:
            self._positions = new_postions

    def get_agent_number(self):
        '''
        Return the number of agents present.
        '''
        return self._positions.shape[0]

#!usr/bin/env

'''
This module models a single agent in the Antuino simulation.
There are some pretty non-pythonic things here, especially in he way I choose
to represent certain variables but this is anticipating work on performance
improvement.
'''

# Imports
import numpy as np
import pdb

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
        self.senses_dim = self.arena.directions.shape[0] + 1

        # Move any direction plus send signal plus do nothing
        self.actions_dim = self.arena.directions.shape[0] + 1 + 1
        self.action_signal_idx = self.arena.directions.shape[0] + 0
        self.action_do_nothing_idx = self.arena.directions.shape[0] + 1

        self.positions = np.zeros((self.ant_no, self.arena.dim),
                                   dtype=default_dtype)
        self.senses = np.zeros((self.ant_no, self.senses_dim),
                                dtype=default_dtype)
        self.actions = np.zeros((self.ant_no, self.actions_dim),
                                   dtype=default_dtype)

        if decision_mode is not None:
            self.decision_cb = eval('self.' + DECISION_FNCTN[decision_mode])
        else:
            self.decision_cb = self.decision_do_nothing

        self.lin_dec_mat = None

        self.alive = None
        self.alive_no = 0

        self.signal_range = 100
        self.distances = np.zeros((self.ant_no, self.ant_no))
        self.signals = np.zeros((self.ant_no, self.arena.directions.shape[0]))
        self.signalled = np.zeros(self.ant_no)

    def recompute_distances(self):
        '''
        Computes distances between each pair of alive agents.
        '''
        self.distances = np.zeros_like(self.distances)
        upper_right = np.triu_indices_from(self.distances)
        lower_left = np.tril_indices_from(self.distances)
        xs = np.expand_dims(self.positions[:,0], 1)
        x_dif = xs - xs.T
        ys = np.expand_dims(self.positions[:,1], 1)
        y_dif = ys - ys.T
        if self.arena.dim == 2:
            self.distances[upper_right] = np.sqrt(x_dif[upper_right]**2 +
                                                  y_dif[upper_right]**2)
        elif self.arena.dim == 3:
            zs = np.expand_dims(self.positions[:,2], 1)
            z_dif = zs - zs.T
            self.distances[upper_right] = np.sqrt(x_dif[upper_right]**2 +
                                                  y_dif[upper_right]**2 +
                                                  z_dif[upper_right]**2)
        self.distances[lower_left] += (self.distances.T)[lower_left]

    def get_signal_input(self):
        '''
        Computes the signal reaching each agent.
        '''
        # self.signal_dir_mask = np.zeros_like(self.signal_dir_mask)
        self.signals = np.zeros_like(self.signals)
        signal_in_range_idx = np.argwhere(np.logical_and(self.signalled > 0, self.distances <= self.signal_range))
        signal_in_range_idx = signal_in_range_idx[signal_in_range_idx[:,0]!=signal_in_range_idx[:,1]]
        if len(signal_in_range_idx) > 0:
            for dir_idx, dire in enumerate(self.arena.directions):
                axis = np.argwhere(dire != 0)[0][0]
                sign = 1 if dire[axis] > 0 else -1
                emitters_at_direction_idx = signal_in_range_idx[(sign * self.positions[signal_in_range_idx[:, 0]][:, axis] > np.abs(self.positions[signal_in_range_idx[:, 1]][:, axis]))]
                for transfer in emitters_at_direction_idx:
                    self.signals[transfer[0], dir_idx] += self.signalled[transfer[1]] / self.distances[transfer[0], transfer[1]]

    def sense(self):
        '''
        Update sensory input.
        '''
        # pdb.set_trace()
        self.senses[self.alive] = np.random.randint(0, 1, (self.alive_no, self.senses_dim))
        self.recompute_distances()
        self.get_signal_input()
        self.senses[:, :self.arena.directions.shape[0]] = self.signals

    def decide(self):
        '''
        Chooses an action to perform from available sensory information by
        invoking an appropriate callback.
        '''
        self.decision_cb()
        self.signalled = np.zeros_like(self.signalled)
        self.signalled[self.alive] = self.actions[self.alive, self.action_signal_idx]

    def decision_random(self):
        '''
        Attribute
        '''
        self.actions = np.random.random(self.actions.shape).argsort(1)
        self.actions = (self.actions == self.actions.shape[1] - 1)

    def decision_do_nothing(self):
        '''
        Do nothing
        '''
        self.actions = np.zeros_like(self.actions)
        self.actions[:, self.action_do_nothing_idx] = 1

    def decision_linear(self):
        '''
        Do nothing
        '''
        if len(self.lin_dec_mat.shape) > 2:
            # Each ant has it's own decision making brain
            decision_strenghts = np.empty((self.ant_no, self.actions_dim))
            for i in (idx for idx in range(self.ant_no) if self.alive[idx]):
                decision_strenghts[i] = self.lin_dec_mat[i].dot(
                    self.senses[i])
        else:
            decision_strenghts[self.alive] = self.lin_dec_mat.dot(self.senses[self.alive].T).T
        normalizers = np.max(decision_strenghts[self.alive], axis=1)
        self.actions[self.alive] = np.expand_dims(normalizers, 1) == decision_strenghts[self.alive]

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
            self.positions[np.argwhere(valid_positions)] = \
                new_postions[np.argwhere(valid_positions)]
        else:
            self.positions = new_postions

    def get_agent_number(self):
        '''
        Return the number of agents present.
        '''
        return self.positions.shape[0]

    def reset_alive(self, new_alive):
        self.alive = new_alive
        self.alive_no = sum(self.alive)

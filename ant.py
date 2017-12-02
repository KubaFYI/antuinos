#!usr/bin/env

'''
This module models a single agent in the Antuino simulation.
There are some pretty non-pythonic things here, especially in he way I choose
to represent certain variables but this is anticipating work on performance
improvement.
'''

# Imports
import numpy as np
from numba import njit, prange
import pdb
from rnn import RNN

# Catalogue of decision-making functions
DECISION_FNCTN = {'random': 'decision_random',
                  'nothing': 'decision_do_nothing',
                  'linear': 'decision_linear',
                  'rnn': 'decision_rnn'}

default_dtype = np.float64


class Ants():
    '''
    Class used to handle all the agents in the Antuino simulation. Contains
    each agent's internal and external state. And handles production of
    decision data out of agents' sensory inputs.
    '''

    def __init__(self, arena, max_agents_no, decision_mode=None):
        self.arena = arena
        self.max_agents_no = max_agents_no

        # Sensor on each side of the agent plus food detector
        self.senses_dim = self.arena.directions.shape[0] + 1

        # Move any direction plus send signal plus do nothing
        self.actions_dim = 4
        self.action_go_straight_idx = 0
        self.action_turn_to_rand_side_idx = 1
        self.action_signal_idx = 2
        self.action_do_nothing_idx = 3

        self.positions = np.zeros((self.max_agents_no, self.arena.dim),
                                   dtype=default_dtype)
        self.orientations = np.zeros(self.max_agents_no,
                                   dtype=np.int)
        self.senses = np.zeros((self.max_agents_no, self.senses_dim),
                                dtype=default_dtype)
        self.actions = np.zeros((self.max_agents_no, self.actions_dim),
                                   dtype=default_dtype)

        if decision_mode is not None:
            self.decision_cb = eval('self.' + DECISION_FNCTN[decision_mode])
        else:
            self.decision_cb = self.decision_do_nothing

        self.lin_dec_mat = None

        self.alive = None
        self.alive_no = 0

        self.signal_range = 100
        self.distances = np.zeros((self.max_agents_no, self.max_agents_no))
        self.signals = np.zeros((self.max_agents_no, self.arena.directions.shape[0]))
        self.signalled = np.zeros(self.max_agents_no)
        self.energy_intake = np.zeros(self.max_agents_no, dtype=np.float)

        if decision_mode == 'rnn':
            self.hidden_layer_size = 10
            self.rnn = RNN(self.max_agents_no, self.senses_dim, self.actions_dim, self.hidden_layer_size)
            self.lin_dec_mat = self.rnn.weights

    @staticmethod
    @njit
    def recompute_distances(distances, positions, dim):
        '''
        Computes distances between each pair of alive agents.
        '''
        distances[...] = np.zeros_like(distances)
        xs = np.expand_dims(positions[:,0], 1)
        x_dif = xs - xs.T
        ys = np.expand_dims(positions[:,1], 1)
        y_dif = ys - ys.T
        if dim == 2:
            distances[:, :] = np.sqrt(x_dif[:, :]**2 +
                                                  y_dif[:, :]**2)
        elif dim == 3:
            zs = np.expand_dims(positions[:,2], 1)
            z_dif = zs - zs.T
            distances[:, :] = np.sqrt(x_dif[:, :]**2 +
                                                  y_dif[:, :]**2 +
                                                  z_dif[:, :]**2)
        return distances

    def get_signal_input(self):
        '''
        Computes the signal reaching each agent.
        '''
        self.signals = np.zeros_like(self.signals)
        signal_in_range_idx = np.argwhere(np.logical_and(self.signalled > 0, self.distances <= self.signal_range))
        signal_in_range_idx = signal_in_range_idx[signal_in_range_idx[:,0]!=signal_in_range_idx[:,1]]
        if len(signal_in_range_idx) > 0:
            self.signals = Ants.get_signal_input_compil(signal_in_range_idx, self.positions, self.signals, self.signalled, self.distances, self.arena.dim, self.arena.directions)

    @staticmethod
    @njit
    def get_signal_input_compil(signal_in_range_idx, positions, signals, signalled, distances, dimensions, directions):
            for dir_idx in prange(directions.shape[0]):
                dire = directions[dir_idx]
                if dire[0] != 0:
                    axis = 0
                elif dire[1] != 0:
                    axis = 1
                else:
                    axis = 2
                sign = 1 if dire[axis] > 0 else -1
                in_dir_cone = (sign * (positions[signal_in_range_idx[:, 0]][:, axis] - 
                                       positions[signal_in_range_idx[:, 1]][:, axis]) > 
                               np.abs(positions[signal_in_range_idx[:, 1]][:, (axis+1)%dimensions] -
                                      positions[signal_in_range_idx[:, 0]][:, (axis+1)%dimensions]))
                if dimensions == 3:
                    in_dir_cone = np.logical_and(in_dir_cone, (sign * (positions[signal_in_range_idx[:, 0]][:, axis] - 
                                       positions[signal_in_range_idx[:, 1]][:, axis]) > 
                               np.abs(positions[signal_in_range_idx[:, 1]][:, (axis+2)%dimensions] -
                                      positions[signal_in_range_idx[:, 0]][:, (axis+2)%dimensions])))

                emit_at_dir_idx = signal_in_range_idx[in_dir_cone]

                # for transfer in emit_at_dir_idx:
                for transfer_idx in range(emit_at_dir_idx.shape[0]):
                    transfer = emit_at_dir_idx[transfer_idx]
                    signals[transfer[0], dir_idx] += signalled[transfer[1]] / distances[transfer[0], transfer[1]]
            return signals

    def sense(self):
        '''
        Update sensory input.
        '''
        self.distances = Ants.recompute_distances(self.distances, self.positions, self.arena.dim)
        self.get_signal_input()
        signals_ahead = self.signals[:, self.orientations]
        signals_back = self.signals[:, self.arena.opposite_dirs[self.orientations]]
        signals_side = np.empty_like(signals_ahead)
        for side_idx in range((self.arena.dim-1)*2):
            signals_side += self.signals[:, self.arena.side_dirs[self.orientations, side_idx]]
        self.senses[:, :self.arena.directions.shape[0]] = self.signals
        self.senses[:, -1] = self.energy_intake

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
        Make a decision based on a linear model
        '''
        if len(self.lin_dec_mat.shape) > 2:
            # Each ant has it's own decision making brain
            decision_strenghts = np.empty((self.max_agents_no, self.actions_dim))
            for i in (idx for idx in range(self.max_agents_no) if self.alive[idx]):
                decision_strenghts[i, ...] = self.lin_dec_mat[i].dot(
                    self.senses[i, ...])
        else:
            decision_strenghts[self.alive, ...] = self.lin_dec_mat.dot(self.senses[self.alive, ...].T).T
        normalizers = np.max(decision_strenghts[self.alive, ...], axis=1)
        self.actions[self.alive, ...] = np.expand_dims(normalizers, 1) == decision_strenghts[self.alive, ...]

    def decision_rnn(self):
        '''
        Make a decision based on a RRN
        '''
        self.actions[...] = self.rnn.compute(self.senses[...], self.actions[...])
        normalizers = np.max(self.actions[self.alive, ...], axis=1)
        rand_action = normalizers == 0
        self.actions[np.argwhere(rand_action)[:, 0], np.random.randint(0, self.actions_dim, size=sum(rand_action))] = 1.
        deliberate_action = np.argwhere(rand_action == False)[:, 0]
        self.actions[deliberate_action, ...] = np.expand_dims(normalizers[deliberate_action], 1) == self.actions[deliberate_action, ...]

    def set_decision_matrix(self, decision_matrix=None):
        '''
        Sets the linear decision matrix to be used for decision-making.
        The matrix must be of size Senses x Decisions
        '''
        if decision_matrix is not None:
            self.lin_dec_mat = decision_matrix
        else:
            self.rnn.weights = self.lin_dec_mat

    def get_decision_matrix(self):
        '''
        Get the linear decision matrix to be used for decision-making.
        '''
        return decision_matrix

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

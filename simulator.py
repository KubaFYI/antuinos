#!usr/bin/env

'''
This module handles simulating life of the agents in the arena.
'''

# Imports
import sys
import numpy as np
import ant
import arena
import time
import signal
import pickle
from os import path
from matplotlib import pyplot as plt
from matplotlib import animation

# Constants used in scoring runs
scr_starting_score = 0
scr_step_cost = 0.1
scr_distance_mul = 5
scr_distance_adj = 3
scr_returning_mul = 3
scr_food_pickup_bonus = 1
scr_deposit_bonus = 0

# Constants for evolution
evo_death_thresh = -10
evo_mutation_rate = 0.01
evo_mut_val_var = 0.1
evo_kill_period = 500


data_save_period = 10000
data_file = 'brains.pickle'


def signal_handler(signal, frame):
        print('Ctrl+C pressed - terminating simulation!')
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class Simulator():
    '''
    Handles simulation of the entire life of agents in the arena.
    '''

    def __init__(self, target_arena, decision_mode, max_steps, agent_no, seed=None):
        self._arena = target_arena
        self._agents = []

        self._decision_mode = decision_mode
        self._agent_no = agent_no
        self._max_steps = max_steps

        self._step_number = 0
        self._fig = None
        self._axes = None
        self._anim = None

        self._scores = None

        np.random.seed(seed)


    def save_data(self):
        '''
        Saves brains evolved so far into a pickle file.
        '''
        with open(data_file, 'wb') as f:
            pickle.dump(self._agents.lin_dec_mat, f)

    def load_data(self):
        '''
        Load brains evolved so far from a pickle file.
        '''
        if path.exists(data_file):
            with open(data_file, 'rb') as f:
                self._agents.lin_dec_mat = pickle.load(f)
            print('Loaded brain data from {}'.format(data_file))

    def _gen_rand_pos(self, number, centre, std_dev):
        retval = np.tile(centre,
                         (number, 1))

        deviation = np.random.normal(0, 1,
                                     (number, 2))
        deviation2 = np.empty_like(deviation)
        deviation2[:, 0] = np.sin(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * std_dev
        deviation2[:, 1] = np.cos(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * std_dev
        retval += (deviation2 + 0.5).astype(int)
        return retval

    def _gen_starting_pos(self, number):
        starting_positions = self._gen_rand_pos(number,
                                                self._arena.start_point,
                                                arena.STRT_PNT_RD)

        invalid_pos = np.logical_not(
            self._arena.are_valid_positions(starting_positions))
        while invalid_pos.any():
            starting_positions[invalid_pos, :] = self._gen_rand_pos(sum(invalid_pos),
                                                                    self._arena.start_point,
                                                                    arena.STRT_PNT_RD)
            invalid_pos = np.logical_not(
                self._arena.are_valid_positions(starting_positions))
        return starting_positions

    def _populate(self):
        '''
        Populates arena with agents around the starting point
        '''
        starting_positions = self._gen_starting_pos(self._agent_no)

        self._agents = ant.Ants(self._arena.narena,
                                starting_positions,
                                self._decision_mode)

        self._agents.set_decision_matrix(
            np.random.rand(self._agents.ant_no, ant.decisions_dim, ant.senses_dim))

        self._scores = scr_starting_score * np.ones(self._agent_no)

    def tile_coords(self, points):
        '''
        Helper function used to generate sets of coordinates as if the arena
        was on a torus.
        '''
        tiled_points = np.array(points, ndmin=2)
        for offset in ant.RELATIVE_COORDS:
            offset_points = (np.array(self._arena._goals.copy()) +
                             np.array([offset[0] * self._arena._size[0],
                                       offset[1] * self._arena._size[1]]))
            np.concatenate((tiled_points, offset_points))
        return tiled_points

    def _score(self):
        '''
        Updates scores for all agents and returns the swarm's mean.
        '''

        # Just living costs something
        self._scores -= scr_step_cost

        # For non-carrying ants being nearer food source is good
        carrying = self._agents._senses[:, ant.sense_idx['carry_food']] == 1
        non_carrying = np.logical_not(carrying)

        # assume we are as far as only possible far
        food_distances = np.ones(self._agents.ant_no) * \
            np.max(self._arena._size)

        # Make sure we calculate the right distanes given we are on a torus
        tiled_goals = self.tile_coords(self._arena._goals)

        for goal in tiled_goals:
            distances = np.linalg.norm(self._agents._positions - goal,
                                       ord=2, axis=1)
            better_dists = food_distances > distances
            food_distances[better_dists] = distances[better_dists]
        self._scores[non_carrying] += (scr_distance_mul /
                                       (np.maximum(distances[non_carrying], arena.GL_PNT_RD - 1) +
                                        scr_distance_adj))

        # But for carrying we want to go to go back home instead
        home_distances = np.ones(self._agents.ant_no) * \
            np.max(self._arena._size)
        tiled_start_pos = self.tile_coords(self._arena.start_point)
        for home in tiled_start_pos:
            distances = np.linalg.norm(self._agents._positions - home,
                                       ord=2, axis=1)
            better_dists = home_distances > distances
            home_distances[better_dists] = distances[better_dists]

        self._scores[carrying] += (scr_distance_mul * scr_returning_mul /
                                   (np.maximum(home_distances[carrying], arena.GL_PNT_RD - 1) +
                                    scr_distance_adj))

        # We also need to reward picking up food
        self._scores[np.logical_and(
            non_carrying, self._agents.ants_on_goal)] += scr_food_pickup_bonus
        # And returning it to home
        self._scores[np.logical_and(
            carrying, self._agents.ants_on_home)] += scr_deposit_bonus

        return np.mean(self._scores)

    def _step(self):
        '''
        Execute one step of the simulaton.
        '''
        self._agents.sense()
        self._agents.decide()

        new_pos = np.copy(self._agents._positions)
        new_pos[np.argwhere(self._agents._decisions
                            [:, ant.Decision.GO_N.value])] += [0, 1]
        new_pos[np.argwhere(self._agents._decisions
                            [:, ant.Decision.GO_S.value])] += [0, -1]
        new_pos[np.argwhere(self._agents._decisions
                            [:, ant.Decision.GO_E.value])] += [1, 0]
        new_pos[np.argwhere(self._agents._decisions
                            [:, ant.Decision.GO_W.value])] += [-1, 0]

        new_pos %= self._arena.narena[0, ...].shape
        valid_pos = self._arena.are_valid_positions(new_pos)
        self._agents.update_positions(new_pos, valid_pos)

        self._step_number += 1
        mean_score = self._score()

        if self._step_number % evo_kill_period == 0:
            print('Step {}\tMean score: {}'.format(self._step_number, mean_score))
            self.kill_and_spawn()
        if self._step_number % data_save_period == 0:
            self.save_data()
        return mean_score

    def _draw_still_on_axes(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self._axes.clear()
        self._arena.figurize_arena(self._axes)
        self._axes.plot(self._agents._positions[:, 0],
                        self._agents._positions[:, 1],
                        'bo', ms=4, color='green', zorder=20)

    def _setup_anim(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self._arena.figurize_arena(self._axes)
        self._moving_bits, = self._axes.plot([], [], 'bo', ms=2, color='green',
                                             zorder=20)
        self.anim_counter = 0
        return self._moving_bits,

    def _animate(self, i):
        if self._step_number < i:
            self._step()
        self._moving_bits.set_data(self._agents._positions[:, 0],
                                   self._agents._positions[:, 1])
        # print('Animation step {}.'.format(self.anim_counter))
        self.anim_counter += 1
        return self._moving_bits,

    def run(self, score_history=False, figure=False, animate=False):
        if score_history:
            retval = []

        if animate:
            retval = None
            self._fig = plt.figure()
            self._axes = plt.gca()

            self._anim = animation.FuncAnimation(self._fig, self._animate,
                                                 init_func=self._setup_anim,
                                                 frames=self._max_steps,
                                                 interval=20,
                                                 repeat=False,
                                                 blit=True)
            plt.show()
        else:
            if figure:
                self._fig = plt.figure()
                self._axes = plt.gca()

            if self._max_steps is not None:
                while self._step_number < self._max_steps:
                    if score_history:
                        retval.append(self._step())
                    else:
                        retval = self._step()
            else:
                while True:
                    if score_history:
                        retval.append(self._step())
                    else:
                        retval = self._step()

            if figure:
                sim._draw_still_on_axes()
                plt.show()

        return retval

    def softmax(self, x):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def crossover(self, arrays):
        '''
        Return array_like arrays created by crossing over randomly paired
        elements of argument 'arrays'.
        '''
        retval = np.empty_like(arrays)
        crossover_points = np.random.randint(0, arrays.shape[1], size=3)
        shuffld_arrays = np.random.permutation(arrays)
        prev_idx_range = 0
        for idx, crossover_point in enumerate(crossover_points):
            idx_range = int(idx * arrays.shape[0] / 3)
            retval[prev_idx_range:idx_range] = np.concatenate((arrays[prev_idx_range:idx_range, :crossover_point],
                                                               shuffld_arrays[prev_idx_range:idx_range, crossover_point:]),
                                                              axis=1)
            prev_idx_range = idx_range
        return retval

    def mutate(self, arrays):
        '''
        In-place mutate values of the provided individuals (gaussian)
        '''
        if arrays.shape[0] > 0:
            arrays[...] *= (evo_mut_val_var * np.random.standard_normal(arrays.shape) + 1)

    def kill_and_spawn(self):
        '''
        Kill any useless agents and spawn new ones based on the well performing
        lot.
        '''
        sorted_idx = np.argsort(self._scores)
        probs = self.softmax(self._scores)

        # Randomly cross-over good performers
        crossover_choice = np.random.random(self._agent_no) < probs

        # And force cross-over for 4 best
        crossover_choice[sorted_idx[-4:]] = True

        # pdb.set_trace()
        if crossover_choice.shape[0] > 0:
            # Replace the worst lot with newly generated agents
            worst_lot = sorted_idx[:np.sum(crossover_choice)]
            # Set postitions around start point
            self._agents._positions[worst_lot] = self._gen_starting_pos(worst_lot.shape[0])
            # Set their scores
            self._scores[worst_lot] = scr_starting_score
            # Crate their brains from the ones chosen for crossover
            self._agents.lin_dec_mat[worst_lot] = self.crossover(self._agents.lin_dec_mat[crossover_choice])

        # Also, mutate a small number of individuals
        mutants = np.random.random(self._agent_no) < evo_mutation_rate
        self.mutate(self._agents.lin_dec_mat[mutants])

        print('Crossed-over {}\t, mutated {}\t'.format(np.sum(crossover_choice), np.sum(mutants)))


def enable_xkcd_mode():
    from matplotlib import patheffects
    from matplotlib import rcParams
    plt.xkcd()
    rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]


if __name__ == '__main__':
    # plt.close('all')
    print('Starting')
    max_steps = None
    agent_no = 50
    test_arena = arena.Arena(size=(200, 200),
                             start_point=(100, 100),
                             goals=[(175, 175), (40, 140)],
                             obstacles=(
        [['rect', (40, 40), (170, 70)],
         ['rect', (25, 120), (75, 127)],
         ['circ', (145, 170), 15],
         ['circ', (75, 75), 8]]))
    sim = Simulator(target_arena=test_arena,
                    decision_mode='linear',
                    # decision_mode='random',
                    max_steps=max_steps,
                    agent_no=agent_no,
                    seed=15)
    sim._populate()
    sim.load_data()

    start_time = time.time()
    scores = sim.run(score_history=True,
                     # figure=False)
                     animate=True)

    print('{}s total, {}ms p/a'.format((time.time() - start_time), (time.time() - start_time) / max_steps / agent_no * 1000))
    # plt.figure()
    # plt.plot(np.linspace(0, 1, max_steps), scores)
    # plt.show()
P
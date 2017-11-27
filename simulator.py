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
from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import pdb

class Simulator():
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

        # 'Physics' constants
        self._velocity = 0.2

        # Constants used in scoring runs
        self._scr_starting_score = 0
        self._scr_movement_cost = 0.1
        self._scr_signal_cost = 10
        self._scr_energy_rad = 5
        self._scr_energy_max_per_step = 5
        self._scr_energy_adj = 5

        # Evolution constants
        self.evo_mut_rate = 0.01
        self.evo_mut_val_var = 0.05


        self.report_period = 1.5e3
        self.data_save_period = 1e5
        self.data_file = 'brains'
        self.anim_rot_enabled = False
        self.anim_rot_speed = 0.1
        self.anim_rot_period = 100
        self.anim_base_azim = 30
        # self.anim_azim_dev = 30
        self.anim_azim_rot_dir = 1

        np.random.seed(seed)


    def save_data(self):
        '''
        Saves brains evolved so far into a pickle file.
        '''

        with open('{}_{}.pickle'.format(self.data_file, self._step_number), 'wb') as f:
            pickle.dump(self._agents.lin_dec_mat, f)

    def load_data(self):
        '''
        Load brains evolved so far from a pickle file.
        '''
        if path.exists(self.data_file + '.pickle'):
            with open(self.data_file, 'rb') as f:
                self._agents.lin_dec_mat = pickle.load(f)
            print('Loaded brain data from {}'.format(self.data_file))

    def _gen_rand_pos(self, arr=None):
        '''
        In place generate a number of random positions in the arena and put it
        in given array.
        '''
        if arr.shape[0] != 0:
            arr[...] = np.random.random([arr.shape[0], self._arena.dim])
            arr[...] *= np.array(self._arena.size)

    def _populate(self):
        '''
        Populates arena with agents.
        '''
        self._agents = ant.Ants(self._arena,
                                self._agent_no,
                                self._decision_mode)
        self._gen_rand_pos(self._agents._positions)

        self._agents.set_decision_matrix(
            np.random.rand(self._agents.ant_no, self._agents._actions_dim, self._agents._senses_dim))

        self._scores = self._scr_starting_score * np.ones(self._agent_no)

    def tile_coords(self, points):
        '''
        Helper function used to generate sets of coordinates as if the arena
        was on a torus.
        '''
        tiled_points = np.array(points, ndmin=2)
        if len(self._arena.size) == 2:
            for offset in self._arena.directions:
                offset_points = (np.array(self._arena._goals.copy()) +
                                 np.array([offset[0] * self._arena.size[0],
                                           offset[1] * self._arena.size[1]]))
                np.concatenate((tiled_points, offset_points))
        elif len(self._arena.size) == 3:
            for offset in self._arena.directions:
                offset_points = (np.array(self._arena._goals.copy()) +
                                 np.array([offset[0] * self._arena.size[0],
                                           offset[1] * self._arena.size[1],
                                           offset[2] * self._arena.size[2]]))
                np.concatenate((tiled_points, offset_points))
        return tiled_points

    def _score(self):
        '''
        Updates scores for all agents and returns the swarm's mean.
        '''

        # Action Costs
        self._scores[np.sum(self._agents._actions[:, :self._arena.directions.shape[0]]) == 1.] -= self._scr_movement_cost
        self._scores[self._agents._actions[:, self._agents._action_signal_idx] == 1.] -= self._scr_signal_cost

        # Make sure we calculate the right distanes given we are on a torus
        tiled_goals = self.tile_coords(self._arena._goals)

        smallest_dist = np.ones(self._agents.ant_no) * np.max(self._arena.size)
        for goal in tiled_goals:
            dist = np.linalg.norm(self._agents._positions - goal,
                                       ord=2, axis=1)
            better_dists = smallest_dist > dist
            smallest_dist[better_dists] = dist[better_dists]

        taking_energy = smallest_dist <= self._scr_energy_rad
        self._scores[taking_energy] += ((self._scr_energy_max_per_step * (self._scr_energy_adj + self._arena.GOAL_RADIUS)) /
                                       (np.maximum(smallest_dist[taking_energy], self._arena.GOAL_RADIUS) +
                                        self._scr_energy_adj))
        return np.mean(self._scores)

    def _step(self):
        '''
        Execute one step of the simulaton.
        '''
        self._agents.sense()
        self._agents.decide()

        new_pos = np.copy(self._agents._positions)
        for idx, direction in enumerate(self._arena.directions):
            self._agents._positions[np.argwhere(self._agents._actions[:, idx])] += self._velocity * direction

        # Wrap around edges
        self._agents._positions %= self._arena.size

        self._step_number += 1
        mean_score = self._score()

        self.kill_and_spawn()

        if self._step_number % self.report_period == 0:
            print('Step {}\tMean score: {}'.format(self._step_number, mean_score))
            self.kill_and_spawn()
        if self._step_number % self.data_save_period == 0:
            self.save_data()

        return mean_score

    def _draw_still_on_axes(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self._axes.clear()
        self._arena.figurize_arena(self._axes)
        if self._arena.dim == 2:
            self._axes.scatter(self._agents._positions[:, 0],
                               self._agents._positions[:, 1],
                               marker='o', color='blue', zorder=20)
        elif self._arena.dim == 3:
            self._axes.plot(self._agents._positions[:, 0],
                                 self._agents._positions[:, 1],
                                 self._agents._positions[:, 2],
                                 'bo', color='blue', zorder=20)

    def _setup_anim(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self._arena.figurize_arena(self._axes)

        if self._arena.dim == 2:
            self._moving_bits, = self._axes.scatter([], [], marker='o', 
                                                    color='blue',
                                                    zorder=20)
        elif self._arena.dim == 3:
            self._moving_bits, = self._axes.plot([], [], [],
                                 'bo', color='blue', zorder=20)
            # self._moving_bits = None
            # for agent in self._agents._positions:
            #     self._arena.draw_sphere(self._axes, centre=agent, radius=2, color='blue', transparency=1.)
        self.anim_counter = 0
        return self._moving_bits,

    def _animate(self, i, ax):
        if self._step_number < i:
            self._step()
        if self._arena.dim == 2:
            self._moving_bits.set_data(self._agents._positions[:, 0],
                                       self._agents._positions[:, 1])
        elif self._arena.dim == 3:
            if self.anim_rot_enabled:
                azim = self.anim_base_azim + np.abs(((self.anim_rot_speed * i) % self.anim_rot_period) - self.anim_rot_period / 2)
                ax.view_init(elev=30, azim=azim)
            self._moving_bits.set_data(self._agents._positions[:, 0],
                                       self._agents._positions[:, 1])
            self._moving_bits.set_3d_properties(self._agents._positions[:, 2])
            # ax.clear()
            # for agent in self._agents._positions:
            #     self._arena.draw_sphere(ax, centre=agent, radius=5, color='blue', transparency=1.)
            # pass
        self.anim_counter += 1
        return self._moving_bits,

    def run(self, score_history=False, figure=False, animate=False):
        plt.close('all')
        if score_history:
            retval = []

        if animate:
            retval = None
            self._fig = plt.figure()
            if self._arena.dim == 2:
                self._axes = plt.gca()
            elif self._arena.dim == 3:
                self._axes = p3.Axes3D(self._fig)
                # self._axes = self._fig.add_subplot(111, projection='3d')

            self._axes.view_init(dist=8, elev=30, azim=30)
            self._axes.set_proj_type('persp')

            self._anim = animation.FuncAnimation(self._fig, self._animate,
                                                 init_func=self._setup_anim,
                                                 frames=self._max_steps,
                                                 interval=20,
                                                 fargs=(self._axes,),
                                                 repeat=False,
                                                 blit=(not self.anim_rot_enabled))
            plt.show()
        else:
            if figure:
                self._fig = plt.figure()
                if self._arena.dim == 2:
                    self._axes = plt.gca()
                elif self._arena.dim == 3:
                    self._axes = self._fig.add_subplot(111, projection='3d')

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

    def crossover(self, arr):
        '''
        Return array_like arrays created by crossing over randomly paired
        elements of argument 'arrays'.
        '''
        retval = np.empty_like(arr)
        crossover_points = np.random.randint(0, arr.shape[1], size=3)
        shuffld_arr = np.random.permutation(arr)
        prev_idx_range = 0
        for idx, crossover_point in enumerate(crossover_points):
            idx_range = int(idx * arr.shape[0] / 3)
            retval[prev_idx_range:idx_range] = np.concatenate((arr[prev_idx_range:idx_range, :crossover_point],
                                                               shuffld_arr[prev_idx_range:idx_range, crossover_point:]),
                                                              axis=1)
            prev_idx_range = idx_range
        return retval

    def mutate(self, arr):
        '''
        In-place mutate values of the provided individuals (gaussian)
        '''
        if arr.shape[0] != 0:
            arr[...] *= (self.evo_mut_val_var * np.random.standard_normal(arr.shape) + 1)

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

        if crossover_choice.shape[0] > 0:
            # Replace the worst lot with newly generated agents
            worst_lot = sorted_idx[:np.sum(crossover_choice)]
            # Set postitions around start point
            self._gen_rand_pos(self._agents._positions[worst_lot])
            # Set their scores
            self._scores[worst_lot] = self._scr_starting_score
            # Crate their brains from the ones chosen for crossover
            self._agents.lin_dec_mat[worst_lot] = self.crossover(self._agents.lin_dec_mat[crossover_choice])

        # Also, mutate a small number of individuals
        mutants = np.random.random(self._agent_no) < self.evo_mut_rate
        self.mutate(self._agents.lin_dec_mat[mutants])

        # print('Crossed-over {}\t, mutated {}\t'.format(np.sum(crossover_choice), np.sum(mutants)))
        pass


def enable_xkcd_mode():
    from matplotlib import patheffects
    from matplotlib import rcParams
    plt.xkcd()
    rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]


if __name__ == '__main__':
    # plt.close('all')
    # enable_xkcd_mode()
    print('Starting')
    max_steps = 1000
    agent_no = 500
    test_arena = arena.Arena(size=(200, 200, 200))
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
                     # figure=True)
                     animate=True)

    print('{}s total, {}ms p/a'.format((time.time() - start_time), (time.time() - start_time) / max_steps / agent_no * 1000))
    # plt.figure()
    # plt.plot(np.linspace(0, 1, max_steps), scores)
    # plt.show()

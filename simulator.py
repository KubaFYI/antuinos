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

class Simulator():
    def __init__(self, target_arena, decision_mode, max_steps, min_agent_no=50, max_agent_no=250, seed=None):
        self.arena = target_arena

        self.decision_mode = decision_mode
        self.min_agent_no = min_agent_no
        self.max_agent_no = max_agent_no
        self.max_steps = max_steps

        self.step_number = 0
        self.fig = None
        self.axes = None
        self.anim = None

        self.start_time = None

        # 'Physics' constants
        self.velocity = 0.2

        # Constants used in scoring runs
        self.scr_starting_score = 10
        self.scr_movement_cost = 0.1
        self.scr_signal_cost = 0.001
        self.scr_energy_rad = 5
        self.scr_energy_max_per_step = 5
        self.scr_energy_adj = 5

        # Evolution constants
        self.evo_mut_rate = 0.01
        self.evo_mut_val_var = 0.05
        self.evo_mut_gene_prob = 0.05
        self.evo_repr_thresh = 10.
        self.evo_repr_cost = 8.

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

        # Initialize things
        self.agents = ant.Ants(self.arena,
                                self.max_agent_no,
                                self.decision_mode)
        self.gen_rand_pos(self.agents.positions)

        self.agents.set_decision_matrix(
            np.random.rand(self.agents.ant_no, self.agents.actions_dim, self.agents.senses_dim))

        self.scores = self.scr_starting_score * np.ones(self.max_agent_no)
        self.agents.reset_alive(np.ones_like(self.scores, dtype=np.bool))


    def save_data(self):
        '''
        Saves brains evolved so far into a pickle file.
        '''

        with open('{}_{}.pickle'.format(self.data_file, self.step_number), 'wb') as f:
            pickle.dump(self.agents.lin_dec_mat, f)

    def load_data(self):
        '''
        Load brains evolved so far from a pickle file.
        '''
        if path.exists(self.data_file + '.pickle'):
            with open(self.data_file, 'rb') as f:
                self.agents.lin_dec_mat = pickle.load(f)
            print('Loaded brain data from {}'.format(self.data_file))

    def gen_rand_pos(self, arr=None):
        '''
        In place generate a number of random positions in the arena and put it
        in given array.
        '''
        if arr.shape[0] != 0:
            arr[...] = np.random.random([arr.shape[0], self.arena.dim])
            arr[...] *= np.array(self.arena.size)

    def tile_coords(self, points):
        '''
        Helper function used to generate sets of coordinates as if the arena
        was on a torus.
        '''
        tiled_points = np.array(points, ndmin=2)
        if len(self.arena.size) == 2:
            for offset in self.arena.directions:
                offset_points = (np.array(self.arena.goals.copy()) +
                                 np.array([offset[0] * self.arena.size[0],
                                           offset[1] * self.arena.size[1]]))
                np.concatenate((tiled_points, offset_points))
        elif len(self.arena.size) == 3:
            for offset in self.arena.directions:
                offset_points = (np.array(self.arena.goals.copy()) +
                                 np.array([offset[0] * self.arena.size[0],
                                           offset[1] * self.arena.size[1],
                                           offset[2] * self.arena.size[2]]))
                np.concatenate((tiled_points, offset_points))
        return tiled_points

    def score(self):
        '''
        Updates scores for all agents and returns the swarm's mean.
        '''

        # Action Costs
        moved = np.logical_and(self.agents.alive, np.sum(self.agents.actions[:, :self.arena.directions.shape[0]]) == 1.)
        signalled = self.agents.signalled > 0
        self.scores[moved] -= self.scr_movement_cost
        self.scores[signalled] -= self.scr_signal_cost

        # Make sure we calculate the right distanes given we are on a torus
        tiled_goals = self.tile_coords(self.arena.goals)

        smallest_dist = np.ones(self.agents.ant_no) * np.max(self.arena.size)
        for goal in tiled_goals:
            dist = np.linalg.norm(self.agents.positions - goal,
                                       ord=2, axis=1)
            better_dists = smallest_dist > dist
            smallest_dist[better_dists] = dist[better_dists]

        taking_energy = np.logical_and(self.agents.alive, smallest_dist <= self.scr_energy_rad)
        self.scores[taking_energy] += ((self.scr_energy_max_per_step * (self.scr_energy_adj + self.arena.GOAL_RADIUS)) /
                                       (np.maximum(smallest_dist[taking_energy], self.arena.GOAL_RADIUS) +
                                        self.scr_energy_adj))
        return np.mean(self.scores[self.agents.alive])

    def step(self):
        '''
        Execute one step of the simulaton.
        '''
        self.agents.sense()
        self.agents.decide()

        new_pos = np.copy(self.agents.positions)
        for idx, direction in enumerate(self.arena.directions):
            self.agents.positions[np.argwhere(self.agents.actions[:, idx])] += self.velocity * direction

        # Wrap around edges
        self.agents.positions %= self.arena.size

        self.step_number += 1
        mean_score = self.score()

        self.kill_and_spawn()

        if self.step_number % self.report_period == 0:
            print('Step {}\tMean score: {}'.format(self.step_number, mean_score))
            self.kill_and_spawn()
        if self.step_number % self.data_save_period == 0:
            self.save_data()

        return mean_score

    def draw_still_on_axes(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self.axes.clear()
        self.arena.figurize_arena(self.axes)
        if self.arena.dim == 2:
            self.axes.scatter(self.agents.positions[:, 0],
                               self.agents.positions[:, 1],
                               marker='o', color='blue', zorder=20)
        elif self.arena.dim == 3:
            self.axes.plot(self.agents.positions[:, 0],
                                 self.agents.positions[:, 1],
                                 self.agents.positions[:, 2],
                                 'bo', color='blue', zorder=20)

    def setup_anim(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self.arena.figurize_arena(self.axes)

        if self.arena.dim == 2:
            self.moving_bits, = self.axes.scatter([], [], marker='o', 
                                                    color='blue',
                                                    zorder=20)
        elif self.arena.dim == 3:
            self.moving_bits, = self.axes.plot([], [], [],
                                 'bo', color='blue', zorder=20)
        self.anim_counter = 0
        return self.moving_bits,

    def animate(self, i, ax):
        if self.step_number < i:
            self.step()
        if self.arena.dim == 2:
            self.moving_bits.set_data(self.agents.positions[self.agents.alive, 0],
                                       self.agents.positions[self.agents.alive, 1])
        elif self.arena.dim == 3:
            if self.anim_rot_enabled:
                azim = self.anim_base_azim + np.abs(((self.anim_rot_speed * i) % self.anim_rot_period) - self.anim_rot_period / 2)
                ax.view_init(elev=30, azim=azim)
            self.moving_bits.set_data(self.agents.positions[self.agents.alive, 0],
                                       self.agents.positions[self.agents.alive, 1])
            self.moving_bits.set_3d_properties(self.agents.positions[self.agents.alive, 2])
        self.anim_counter += 1
        return self.moving_bits,

    def run(self, history=False, figure=False, animate=False):
        plt.close('all')
        if history:
            retval = []
        self.start_time = time.time()
        if animate:
            retval = None
            self.fig = plt.figure()
            if self.arena.dim == 2:
                self.axes = plt.gca()
            elif self.arena.dim == 3:
                self.axes = p3.Axes3D(self.fig)
                # self.axes = self.fig.add_subplot(111, projection='3d')

            self.axes.view_init(dist=8, elev=30, azim=30)
            self.axes.set_proj_type('persp')

            self.anim = animation.FuncAnimation(self.fig, self.animate,
                                                 init_func=self.setup_anim,
                                                 frames=self.max_steps,
                                                 interval=20,
                                                 fargs=(self.axes,),
                                                 repeat=False,
                                                 blit=(not self.anim_rot_enabled))
            plt.show()
        else:
            if figure:
                self.fig = plt.figure()
                if self.arena.dim == 2:
                    self.axes = plt.gca()
                elif self.arena.dim == 3:
                    self.axes = self.fig.add_subplot(111, projection='3d')

            if self.max_steps is not None:
                while self.step_number < self.max_steps:
                    if history:
                        retval.append(self.step())
                    else:
                        retval = self.step()
            else:
                while True:
                    if history:
                        retval.append(self.step())
                    else:
                        retval = self.step()
            print('Duration {}s ({}ms per step)'.format(time.time() - start_time, 
                                                        (time.time() - start_time) * 1000 / self.max_steps))
            if figure:
                sim.draw_still_on_axes()
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
            genes_mutated = np.random.random(arr.shape) < self.evo_mut_gene_prob
            if genes_mutated:
                arr[genes_mutated] *= (self.evo_mut_val_var * np.random.standard_normal(sum(genes_mutated)) + 1)

    def kill_and_spawn(self):
        '''
        Kill and spawn new agents as required
        '''
        # Kill off agents first.
        self.agents.reset_alive(self.scores > 0)

        # Get ranking of the scores (starting with the worst)
        ranking = np.argsort(self.scores[self.agents.alive])

        # Now reproduce the ones which are able to
        reproductors = self.scores > self.evo_repr_thresh
        if sum(reproductors) > 0:
            self.scores[reproductors] -= self.evo_repr_cost
    
            # If there is no space for the new agents so kill off some bad ones
            need_to_kill = sum(reproductors) - (self.max_agent_no - self.agents.alive_no)
            idx = 0
            while need_to_kill > 0:
                if self.agents.alive[ranking[i]]:
                    self.agents.alive[ranking[i]] = False
                    self.agents.alive_no -= 1
                    need_to_kill -= 1
    
            children = np.argwhere(np.logical_not(self.agents.alive))[:sum(reproductors)]
    
            # Resurect some of the dead as the children of the reproductors
            self.agents.alive[children] = True
            self.gen_rand_pos(self.agents.positions[children])
            self.agents.lin_dec_mat[children] = self.agents.lin_dec_mat[reproductors].copy()
            self.scores[children] = self.scr_starting_score
            print('Reproduced {}'.format(sum(reproductors)))

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
    min_agent_no = 50
    max_agent_no = 100
    test_arena = arena.Arena(size=(300, 300, 300))


    sim = Simulator(target_arena=test_arena,
                    decision_mode='linear',
                    # decision_mode='random',
                    max_steps=max_steps,
                    min_agent_no=min_agent_no,
                    max_agent_no=max_agent_no,
                    seed=15)
    sim.load_data()

    start_time = time.time()
    scores = sim.run(history=True,
                     figure=True)
                     # animate=True)

    # print('{}s total, {}ms p/a'.format((time.time() - start_time), (time.time() - start_time) / max_steps / max_agent_no * 1000))
    # plt.figure()
    # plt.plot(np.linspace(0, 1, max_steps), scores)
    # plt.show()

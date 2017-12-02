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
from numba import njit, prange
import pdb
sys.path.append('/usr/local/cuda/lib64')

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
        self.axes_vals = None
        self.anim = None

        self.start_time = None

        # 'Physics' constants
        self.velocity = 0.2

        # Constants used in scoring runs
        self.scr_starting_score = 2
        self.scr_movement_cost = 0.001
        self.scr_signal_cost = 0.0001
        self.scr_energy_rad = 200
        self.scr_energy_max_per_step = 2.5
        self.scr_energy_adj = 5
        self.scr_same_action_penalty = 1
        self.scr_same_action_thresh = 10

        # Evolution constants
        self.evo_mut_rate = 0.01
        self.evo_mut_val_var = 0.05
        self.evo_mut_gene_prob = 0.05
        self.evo_repr_thresh = 10.
        self.evo_repr_cost = 8.

        self.report_period = 1.5e2
        self.data_save_period = 1e4
        self.goal_move_period = 2.5e3
        self.data_file = 'brains_neuro'
        self.anim_rot_enabled = True
        self.anim_rot_speed = 0.1
        self.anim_rot_period = 100
        self.anim_base_azim = 30
        # self.anim_azim_dev = 30
        self.anim_azim_rot_dir = 1
        self.writer = None
        self.bookkeeping = None

        np.random.seed(seed)

        # Initialize things
        self.agents = ant.Ants(self.arena,
                                self.max_agent_no,
                                self.decision_mode)
        self.agents.positions, self.agents.orientations = self.gen_rand_pos(self.agents.positions, self.agents.orientations)

        # For use with a linear model
        if self.decision_mode == 'linear':
            self.agents.set_decision_matrix(
            np.random.rand(self.agents.max_agents_no, self.agents.actions_dim, self.agents.senses_dim))
        # # For use with a rnn
        # self.agents.set_decision_matrix(
        #     np.random.rand(self.agents.max_agents_no, self.agents.actions_dim, self.agents.senses_dim))

        self.scores = self.scr_starting_score * np.ones(self.max_agent_no)
        self.agents.reset_alive(np.ones_like(self.scores, dtype=np.bool))
        self.agents.alive[:np.int((self.max_agent_no-self.min_agent_no)/2)] = False
        self.agents.alive_no -= int((self.max_agent_no-self.min_agent_no)/2)

        self.past_actions = np.zeros((self.max_agent_no, 2))

        self.past_not_moved = np.zeros(self.max_agent_no)
        self.ages = np.zeros(self.max_agent_no)
        self.max_age = int(2 * self.goal_move_period)
        self.mean_score = 0
        self.loaded = False


    def save_data(self):
        '''
        Saves brains evolved so far into a pickle file.
        '''
        if not self.loaded:
            name = '{}_{}.pickle'.format(self.data_file, self.step_number)
        else:
            name = '{}.pickle'.format(self.data_file)

        with open('numbers.log'.format(self.data_file, self.step_number), 'wb') as f:
            pickle.dump(self.bookkeeping, f)

        with open(name, 'wb') as f:
            pickle.dump(self.agents.lin_dec_mat, f)
        print('SAVED BRAIN DATA! (in {})'.format(name))

    def load_data(self):
        '''
        Load brains evolved so far from a pickle file.
        '''
        filename = self.data_file + '.pickle'
        if path.exists(filename):
            with open(filename, 'rb') as f:
                self.agents.lin_dec_mat = pickle.load(f)
            self.loaded = True
            print('Loaded brain data from {}'.format(filename))

    def gen_rand_pos(self, arr_pos=None, arr_orient=None):
        '''
        Generate a number of random positions in the arena.
        '''
        if arr_pos.shape[0] != 0:
            arr_pos[...] = np.random.random(arr_pos.shape)
            arr_pos[...] *= np.array(self.arena.size)
            if arr_orient is not None:
                arr_orient[...] = np.random.randint(0,
                                                    self.arena.directions.shape[0],
                                                    arr_orient.shape)
                return arr_pos, arr_orient
            return arr_pos


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
        moved_forward = np.logical_and(self.agents.alive, self.agents.actions[:, self.agents.action_go_straight_idx] == 1.)
        moved = np.logical_and(self.agents.alive, np.logical_or(moved_forward, self.agents.actions[:, self.agents.action_turn_to_rand_side_idx] == 1.))
        signalled = self.agents.signalled > 0
        # self.scores-= self.scr_movement_cost
        self.scores[moved] -= self.scr_movement_cost
        self.scores[signalled] -= self.scr_signal_cost
        # action_idxs = np.argwhere(self.agents.actions == 1.)
        # if action_idxs.shape[0] > 0:
        #     self.past_actions[action_idxs[:, 1], 0][(self.past_actions[action_idxs[:, 0], 1] == action_idxs[:, 1])] += 1
        #     self.scores[self.past_actions[:, 0] > self.scr_same_action_thresh] -= self.scr_same_action_penalty
        #     self.past_actions[action_idxs[:, 0], 1] = action_idxs[:, 1]
        same_action_penalized = np.zeros_like(moved, dtype=np.bool)
        if not moved_forward.all():
            immobile = np.logical_not(moved_forward)
            self.past_not_moved[moved_forward] = 0
            self.past_not_moved[immobile] += 1
            same_action_penalized = self.past_not_moved > self.scr_same_action_thresh
            self.scores[same_action_penalized] -= self.scr_same_action_penalty
        # Make sure we calculate the right distanes given we are on a torus
        tiled_goals = self.tile_coords(self.arena.goals)

        smallest_dist = np.ones(self.agents.max_agents_no) * np.max(self.arena.size)
        for goal in tiled_goals:
            dist = np.linalg.norm(self.agents.positions - goal,
                                       ord=2, axis=1)
            better_dists = smallest_dist > dist
            smallest_dist[better_dists] = dist[better_dists]
        closest_idx = (np.argwhere(smallest_dist == np.min(smallest_dist[self.agents.alive])))[0,0]
        taking_energy = np.logical_and(self.agents.alive, smallest_dist <= self.scr_energy_rad)
        # Immobile agents cant take energy
        taking_energy = np.logical_and(taking_energy, np.logical_not(same_action_penalized))
        self.agents.energy_intake = np.zeros_like(self.agents.energy_intake)
        # self.agents.energy_intake[taking_energy] = ((self.scr_energy_max_per_step * (self.scr_energy_adj + self.arena.GOAL_RADIUS)) /
        #                                (np.maximum(smallest_dist[taking_energy], self.arena.GOAL_RADIUS) +
        #                                 self.scr_energy_adj))
        self.agents.energy_intake[taking_energy] = ((self.scr_energy_max_per_step) /
                                       smallest_dist[taking_energy])
        self.scores[taking_energy] += self.agents.energy_intake[taking_energy]
        # pdb.set_trace()
        # print('{} {} {} {} {}'.format(self.scores[closest_idx], self.agents.positions[closest_idx], self.agents.energy_intake[closest_idx], moved[closest_idx], self.past_not_moved[closest_idx]))
        return np.mean(self.scores[self.agents.alive])

    def step(self):
        '''
        Execute one step of the simulaton.
        '''
        # First score in order to initialize the energy intake array
        self.score()
        self.agents.sense()
        self.agents.decide()

        new_pos = np.copy(self.agents.positions)
        # Handle 'go forward action'
        moving_forward = np.argwhere(self.agents.actions[:, self.agents.action_go_straight_idx])
        self.agents.positions[moving_forward] += self.velocity * self.arena.directions[self.agents.orientations[moving_forward]]

        # Hadle 'turn' actions
        turning = np.argwhere(self.agents.actions[:, self.agents.action_turn_to_rand_side_idx])
        self.agents.orientations[turning] = np.random.randint(0, self.arena.directions.shape[0], (turning.shape[0], 1))

        # Wrap around edges
        self.agents.positions %= self.arena.size

        self.step_number += 1
        self.mean_score = self.score()
        self.ages += 1

        self.kill_and_spawn()

        if self.step_number % self.report_period == 0:
            print('Step {}\tAlive agents {}\tMean score: {}\tMax score {}'.format(self.step_number, self.agents.alive_no, self.mean_score, np.max(self.scores[self.agents.alive])))
            # self.kill_and_spawn()
        if self.step_number % self.data_save_period == 0:
            self.save_data()
        if np.random.rand(1) < ( 10 if (self.agents.alive_no == self.max_agent_no) else 1 ) / self.goal_move_period:
            self.regenerate_goal()

        return self.mean_score

    def regenerate_goal(self):
        self.arena.goals = self.gen_rand_pos(self.arena.goals)
        self.axes.clear()
        self.setup_anim()
        print('new goal pos: {}'.format(self.arena.goals))

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
                azim = self.anim_base_azim + np.abs(((self.anim_rot_speed * i/100) % self.anim_rot_period) - self.anim_rot_period / 2)
                ax.view_init(elev=30, azim=azim)
            self.moving_bits.set_data(self.agents.positions[self.agents.alive, 0],
                                       self.agents.positions[self.agents.alive, 1])
            self.moving_bits.set_3d_properties(self.agents.positions[self.agents.alive, 2])
        self.anim_counter += 1
        return self.moving_bits,

    def run(self, history=False, figure=False, animate=False, record=False):
        plt.close('all')
        if history and self.max_steps is not None:
            self.bookkeeping = np.empty((self.max_steps, 2))
        self.start_time = time.time()

        if record:
            plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
            Writer = animation.writers['ffmpeg']
            self.writer = Writer(fps=60, metadata=dict(artist='Me'), extra_args=[
                '-vcodec', 'h264_nvenc'], bitrate=10000)
        if animate:
            self.bookkeeping = None
            self.fig = plt.figure()
            if self.arena.dim == 2:
                self.axes = plt.gca()
            elif self.arena.dim == 3:
                self.axes = p3.Axes3D(self.fig)
                # self.axes = self.fig.add_subplot(111, projection='3d')

            self.axes.view_init(dist=8, elev=30, azim=30)
            self.axes.set_proj_type('persp')
            self.regenerate_goal()
            self.anim = animation.FuncAnimation(self.fig, self.animate,
                                                 init_func=self.setup_anim,
                                                 frames=self.max_steps,
                                                 interval=5,
                                                 fargs=(self.axes,),
                                                 repeat=False,
                                                 blit=(not self.anim_rot_enabled))
            if record:
                self.anim.save('some_name.mp4', writer=self.writer)
            else:
                plt.show()
        else:
            if figure:
                self.fig = plt.figure()
                if self.arena.dim == 2:
                    self.axes = plt.gca()
                elif self.arena.dim == 3:
                    if history and self.max_steps is not None:
                        self.axes = self.fig.add_subplot(211, projection='3d')
                        self.axes_vals = self.fig.add_subplot(212)
                    else:
                        self.axes = self.fig.add_subplot(111, projection='3d')
            self.regenerate_goal()
            if self.max_steps is not None:
                while self.step_number < self.max_steps:
                    if history and self.max_steps is not None:
                        self.bookkeeping[self.step_number-1, :] = np.array([self.agents.alive_no, self.step()])
                    else:
                        self.bookkeeping = self.step()
            else:
                while True:
                    if history and self.max_steps is not None:
                        self.bookkeeping[self.step_number-1, :] = np.array([self.agents.alive_no, self.step()])
                    else:
                        self.bookkeeping = self.step()
            print('Duration {}s ({}ms per step)'.format(time.time() - start_time, 
                                                        (time.time() - start_time) * 1000 / self.max_steps))
            if figure:
                self.draw_still_on_axes()
                # print(self.bookkeeping)
                self.axes_vals.plot([x for x in range(len(self.bookkeeping))], self.bookkeeping[:,1])
                plt.show()

        return self.bookkeeping

    # @staticmethod
    # @njit
    # def crossover(arr):
    #     '''
    #     Return array_like arrays created by crossing over randomly paired
    #     elements of argument 'arrays'.
    #     '''
    #     retval = np.empty_like(arr)
    #     crossover_points = np.random.randint(0, arr.shape[1], size=3)
    #     shuffld_arr = np.random.permutation(arr)
    #     prev_idx_range = 0
    #     for idx in prange(crossover_points):
    #         crossover_point = crossover_points[idx]
    #         idx_range = int(idx * arr.shape[0] / 3)
    #         retval[prev_idx_range:idx_range] = np.concatenate((arr[prev_idx_range:idx_range, :crossover_point],
    #                                                            shuffld_arr[prev_idx_range:idx_range, crossover_point:]),
    #                                                           axis=1)
    #         prev_idx_range = idx_range
    #     return retval

    @staticmethod
    # @njit
    def mutate(arr, evo_mut_gene_prob, evo_mut_val_var):
        '''
        In-place mutate values of the provided individuals (gaussian)
        '''
        # arcpy = arr.copy()
        if arr.shape[0] != 0:
            # genes_mutated = np.empty((arr.shape[0] * arr.shape[1] * arr.shape[2], 3), dtype=np.int64)
            # count = 0
            # for i in prange(arr.shape[0]):
            #     for j in range(arr.shape[2]):
            #         rand = np.random.rand(arr.shape[3])
            #         for k in range(arr.shape[3]):
            #             if rand[k] < evo_mut_gene_prob:
            #                 # genes_mutated[count] = (i, j, k)
            #                 print(arr.shape)
            #                 qwe = np.random.standard_normal(1)
            #                 print(i)
            #                 print(j)
            #                 print(k)
                            # arr[i, 0, j, k] += evo_mut_val_var * qwe * arr[i, 0, j, k]
            # pdb.set_trace()
            to_mutate = np.random.standard_normal(arr.shape) < evo_mut_gene_prob
            arr[to_mutate] += evo_mut_val_var * np.random.standard_normal(arr[to_mutate].shape) * arr[to_mutate]
        # print(np.sum(arcpy==arr))
        return arr
                            # count += 1
            # if genes_mutated.any():
# 
    def kill_and_spawn(self):
        '''
        Kill and spawn new agents as required
        '''
        # Kill off agents first.
        pre_kill = self.agents.alive_no
        self.agents.reset_alive(np.logical_and(self.agents.alive, np.logical_and(self.scores > 0, self.ages < self.max_age)))
        killed = pre_kill - self.agents.alive_no
        # Get ranking of the scores (starting with the worst)
        ranking = np.argsort(self.scores[self.agents.alive])

        # Now reproduce the ones which are able to
        reproductors = self.scores > self.evo_repr_thresh
        repr_idx = np.argwhere(reproductors)
        self.scores[reproductors] -= self.evo_repr_cost
        reproductors[repr_idx[np.random.random(repr_idx.shape) > self.mean_score / 10]] = False
        if sum(reproductors) > 0:
            # self.scores[reproductors] -= self.evo_repr_cost
    
            # If there is no space for the new agents so kill off some bad ones
            need_to_kill = sum(reproductors) - (self.max_agent_no - self.agents.alive_no)
            idx = 0
            while need_to_kill > 0:
                if self.agents.alive[ranking[idx]]:
                    self.agents.alive[ranking[idx]] = False
                    self.agents.alive_no -= 1
                    need_to_kill -= 1
                idx += 1
    
            children = np.argwhere(np.logical_not(self.agents.alive))[:sum(reproductors)][:,0]
    
            # Resurect some of the dead as the children of the reproductors
            self.agents.alive[children] = True
            self.ages[children] = 0
            self.agents.alive_no += children.shape[0]
            self.agents.positions[children, :], self.agents.orientations[children] = self.gen_rand_pos(self.agents.positions[children, :], self.agents.orientations[children])
            self.agents.lin_dec_mat[children, ...] = self.agents.lin_dec_mat[reproductors, ...].copy()
            self.agents.lin_dec_mat[children, ...] = Simulator.mutate(self.agents.lin_dec_mat[children, ...], self.evo_mut_gene_prob, self.evo_mut_val_var)
            self.scores[children] = self.scr_starting_score

        # Make sure we've got enought agents left
        replenishment = self.min_agent_no - self.agents.alive_no
        if replenishment > 0:
            # If not make more from random individuals
            add_agents = np.argwhere(np.logical_not(self.agents.alive))[:self.min_agent_no-self.agents.alive_no]
            self.agents.alive[add_agents] = True
            self.ages[add_agents] = 0
            self.agents.alive_no += add_agents.shape[0]
            self.agents.positions[add_agents, :], self.agents.orientations[add_agents] = self.gen_rand_pos(self.agents.positions[add_agents, :], self.agents.orientations[add_agents])
            parents = np.argwhere(self.agents.alive)
            if type(parents) is int:
                parents = np.array([parents])
            if type(add_agents) is int:
                add_agents = np.array([add_agents])
            parents = parents[np.random.randint(0, parents.shape[0], add_agents.shape[0])]
            self.agents.lin_dec_mat[add_agents, ...] = self.agents.lin_dec_mat[parents, ...].copy()
            self.agents.lin_dec_mat[add_agents, ...] = Simulator.mutate(self.agents.lin_dec_mat[add_agents, ...], self.evo_mut_gene_prob, self.evo_mut_val_var)
            self.scores[add_agents] = self.scr_starting_score

        # print('alv:{}\tKilled\t{}Repr\t{}(good) + \t{}(repl)'.format(self.agents.alive_no, killed, sum(reproductors), replenishment if replenishment > 0 else 0))
        self.agents.set_decision_matrix()

def enable_xkcd_mode():
    from matplotlib import patheffects
    from matplotlib import rcParams
    plt.xkcd()
    rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]

if __name__ == '__main__':
    # plt.close('all')
    # enable_xkcd_mode()
    print('Starting')
    max_steps = int(3e4)
    # max_steps = None
    min_agent_no = 100
    max_agent_no = 500
    test_arena = arena.Arena(size=(400, 400, 400))


    sim = Simulator(target_arena=test_arena,
                    # decision_mode='linear',
                    # decision_mode='random',
                    decision_mode='rnn',
                    max_steps=max_steps,
                    min_agent_no=min_agent_no,
                    max_agent_no=max_agent_no)
    sim.load_data()

    start_time = time.time()
    scores = sim.run(history=False,
                     # figure=True)
                     animate=True, record=True)
    # print('{}s total, {}ms p/a'.format((time.time() - start_time), (time.time() - start_time) / max_steps / max_agent_no * 1000))
    # plt.figure()
    # plt.plot(np.linspace(0, 1, max_steps), scores)
    # plt.show()

#!usr/bin/env

'''
This module handles simulating life of the agents in the arena.
'''

# Imports
import numpy as np
import ant
import arena
import time
from matplotlib import pyplot as plt
from matplotlib import animation

# Constants used in scoring runs
scr_act_cost = 0.1
scr_distance_mul = 0.25
scr_distance_adj = 3
scr_food_pickup_bonus = 20
scr_deposit_bonus = 40


class Simulator():
    '''
    Handles simulation of the entire life of agents in the arena.
    '''

    def __init__(self, target_arena, decision_mode, max_steps, agent_no):
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

    def _populate(self):
        '''
        Populates arena with agents around the starting point
        '''
        starting_positions = self._gen_rand_pos(self._agent_no,
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

        self._agents = ant.Ants(self._arena.narena,
                                starting_positions,
                                self._decision_mode)

        self._agents.set_decision_matrix(
            np.random.rand(ant.decisions_dim, ant.senses_dim))

        self._scores = np.zeros(self._agent_no)

    def _score(self):
        '''
        Updates scores for all agents and returns the swarm's mean.
        '''

        # Performing an action costs something
        active_ants = np.logical_not(
            self._agents._decisions[:, ant.Decision.DO_NOTHING.value] == 1)
        self._scores[active_ants] -= scr_act_cost

        # For non-carrying ants being nearer food source is good
        carrying = self._agents._senses[:, ant.sense_idx['carry_food']] == 1
        non_carrying = np.logical_not(carrying)

        # assume we are as far as only possible far
        food_distances = np.ones(self._agents.ant_no) * \
            np.max(self._arena._size)

        for goal in self._arena._goals:
            distances = np.linalg.norm(self._agents._positions - np.array(goal),
                                       ord=2, axis=1)
            better_dists = food_distances > distances
            food_distances[better_dists] = distances[better_dists]
        self._scores[non_carrying] += (scr_distance_mul /
                                       (np.maximum(distances[non_carrying], arena.GL_PNT_RD - 1) +
                                        scr_distance_adj))

        # But for carrying we want to go to go back home instead
        home_distances = np.linalg.norm(self._agents._positions - np.array(self._arena.start_point),
                                        ord=2, axis=1)
        self._scores[carrying] += (scr_distance_mul /
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
        print('Mean score: {}'.format(mean_score))

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

    def run(self, animate=True):
        if animate:
            self._anim = animation.FuncAnimation(self._fig, self._animate,
                                                 init_func=self._setup_anim,
                                                 frames=self._max_steps,
                                                 interval=20,
                                                 repeat=False,
                                                 blit=True)
        else:
            while self._step_number < self._max_steps:
                self._step()


def enable_xkcd_mode():
    from matplotlib import patheffects
    from matplotlib import rcParams
    plt.xkcd()
    rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]


if __name__ == '__main__':
    # plt.close('all')
    print('Starting')
    max_steps = 2000
    agent_no = 200
    test_arena = arena.Arena(start_point=(50, 50),
                             goals=[(95, 75), (40, 87)],
                             obstacles=(
        [['rect', (20, 30), (30, 45)],
         ['rect', (15, 15), (90, 20)],
         ['circ', (50, 25), 15],
         ['circ', (75, 75), 8]]))
    sim = Simulator(target_arena=test_arena,
                    decision_mode='linear',
                    # decision_mode='random',
                    max_steps=max_steps,
                    agent_no=agent_no)
    sim._populate()
    animate = True
    # animate = False
    # plt.ion()
    sim._fig = plt.figure()
    sim._axes = plt.gca()

    start_time = time.time()
    sim.run(animate)
    print('Average time per step per agent = {}ms'.format(
        (time.time() - start_time) / max_steps / agent_no * 1000))
    if not animate:
        sim._draw_still_on_axes()
    plt.show()

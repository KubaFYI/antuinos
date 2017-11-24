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

    def _populate(self):
        '''
        Populates arena with agents around the starting point
        '''
        starting_positions = np.tile(self._arena.start_point,
                                     (self._agent_no, 1))

        deviation = np.random.normal(0, 1,
                                     (self._agent_no, 2))
        deviation2 = np.empty_like(deviation)
        deviation2[:, 0] = np.sin(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * arena.STRT_PNT_RD
        deviation2[:, 1] = np.cos(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * arena.STRT_PNT_RD
        starting_positions += (deviation2 + 0.5).astype(int)

        self._agents = ant.Ants(self._arena.narena,
                                starting_positions,
                                self._decision_mode)

        self._agents.set_decision_matrix(
            np.random.rand(ant.decisions_dim, ant.senses_dim))

        self._scores = np.zeros(self._agent_no)

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
    max_steps = 20000
    agent_no = 100
    test_arena = arena.Arena(start_point=(50, 50), obstacles=(
        [['rect', (20, 30), (30, 45)],
         ['rect', (15, 15), (90, 20)],
         ['circ', (50, 25), 6],
         ['circ', (75, 75), 8]]))
    sim = Simulator(target_arena=test_arena,
                    decision_mode='linear',
                    # decision_mode='random',
                    max_steps=max_steps,
                    agent_no=agent_no)
    sim._populate()
    animate = True
    # animate = True
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

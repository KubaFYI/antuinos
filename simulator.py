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

        self._scores = np.zeros(self._agent_no)

    def _step(self):
        '''
        Execute one step of the simulaton.
        '''
        print('step')
        self._agents.sense()
        self._agents.decide()

        new_pos = np.copy(self._agents._positions)
        new_pos[np.argwhere(
            self._agents._decisions[ant.Decision.GO_N.value] == 1)] += [0, 1]
        new_pos[np.argwhere(
            self._agents._decisions[ant.Decision.GO_S.value] == 1)] += [0, -1]
        new_pos[np.argwhere(
            self._agents._decisions[ant.Decision.GO_E.value] == 1)] += [1, 0]
        new_pos[np.argwhere(
            self._agents._decisions[ant.Decision.GO_W.value] == 1)] += [-1, 0]

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
        self._moving_bits, = self._axes.plot([], [], 'bo', ms=4, color='green',
                                             zorder=20)
        return self._moving_bits,

    def _animate(self, i):
        if self._step_number < i:
            self._step()
        self._moving_bits.set_data(self._agents._positions[:, 0],
                                   self._agents._positions[:, 1])
        return self._moving_bits,

    def run(self, animate=True):
        if animate:
            self._anim = animation.FuncAnimation(self._fig, self._animate,
                                                 init_func=self._setup_anim,
                                                 frames=self._max_steps,
                                                 interval=20,
                                                 blit=True)
        else:
            while self._step_number < self._max_steps:
                self._step()


if __name__ == '__main__':
    print('Starting')
    max_steps = 100
    agent_no = 100
    test_arena = arena.Arena(start_point=(50, 50), obstacles=(
        ['x >= 15 and x < 20 and y >= 10 and y < 30']))
    sim = Simulator(target_arena=test_arena,
                    decision_mode='random',
                    max_steps=max_steps,
                    agent_no=agent_no)
    sim._populate()
    animate = True
    sim._fig = plt.figure()
    sim._axes = plt.gca()

    start_time = time.time()
    sim.run(animate)
    print('Average time per step per agent = {}ms'.format(
        (time.time() - start_time) / max_steps / agent_no * 1000))
    if not animate:
        sim._draw_still_on_axes()
    plt.show()

#!usr/bin/env

'''
This module handles simulating life of the agents in the arena.
'''

# Imports
import numpy as np
import ant
import arena
from matplotlib import pyplot as plt
from matplotlib import animation


class Simulator():
    '''
    Handles simulation of the entire life of agents in the arena.
    '''

    def __init__(self, target_arena, decision_mode, max_steps, max_agent_no):
        self._arena = target_arena
        self._agents = []

        self._decision_mode = decision_mode
        self._max_agent_no = max_agent_no
        self._max_steps = max_steps

        self._step_number = 0
        self._fig = None
        self._axes = None

    def _populate(self):
        '''
        Populates arena with agents around the starting point
        '''
        starting_positions = np.tile(self._arena.start_point,
                                     (self._max_agent_no, 1))

        deviation = np.random.normal(0, 1,
                                     (self._max_agent_no, 2))
        deviation2 = np.empty_like(deviation)
        deviation2[:, 0] = np.sin(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * arena.STRT_PNT_RD
        deviation2[:, 1] = np.cos(
            deviation[:, 0] * np.pi * 2) * deviation[:, 1] * arena.STRT_PNT_RD
        starting_positions += (deviation2 + 0.5).astype(int)
        for i in range(self._max_agent_no):
            self._agents.append(ant.Ant(self._arena.narena,
                                        starting_positions[i, :],
                                        self._decision_mode))

    def _step(self):
        '''
        Execute one step of the simulaton.
        '''
        for agent in self._agents:
            agent.sense()
            agent.decide()

            new_position = None
            if agent.decision == ant.Decision.GO_N:
                new_position = (agent.position[0], agent.position[1] + 1)
            elif agent.decision == ant.Decision.GO_S:
                new_position = (agent.position[0], agent.position[1] - 1)
            elif agent.decision == ant.Decision.GO_E:
                new_position = (agent.position[0] + 1, agent.position[1])
            elif agent.decision == ant.Decision.GO_W:
                new_position = (agent.position[0] - 1, agent.position[1])

            if self._arena.is_valid_position(new_position):
                agent.update_position(new_position)
            else:
                # Nothing to update in the external state of the agent
                pass
        self._step_number += 1

    def _draw_still_on_axes(self, axes):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        axes.clear()
        self._arena.figurize_arena(axes)
        ant_coords = np.empty((len(self._agents), 2))
        for idx, agent in enumerate(self._agents):
            ant_coords[idx, :] = agent.position
        axes.scatter(ant_coords[:, 0], ant_coords[:, 1],
                     color='green', zorder=2)

    def _setup_anim(self):
        '''
        Draws the graphical representation of the arena and the agents on the
        provided axis.
        '''
        self._arena.figurize_arena(self._axes)
        self._moving_bits, = self._axes.plot([], [], 'bo', ms=3, color='green',
                                             zorder=20)
        return self._moving_bits,

    def _animate(self, i):
        if self._step_number < i:
            self._step()
        ant_coords = np.empty((len(self._agents), 2))

        for idx, agent in enumerate(self._agents):
            ant_coords[idx, :] = agent.position
        self._moving_bits.set_data(ant_coords[:, 0], ant_coords[:, 1])
        return self._moving_bits,

    def run(self, animate='true'):
        if animate:
            anim = animation.FuncAnimation(self._fig, self._animate,
                                           init_func=self._setup_anim,
                                           frames=200, interval=20, blit=True)
            plt.show()
            anim.event_source.stop()
            del ani
            plt.close()


if __name__ == '__main__':
    print('Starting')
    test_arena = arena.Arena(start_point=(50, 50), obstacles=(
        ['x >= 15 and x < 20 and y >= 10 and y < 30']))
    sim = Simulator(target_arena=test_arena,
                    decision_mode='random',
                    max_steps=10,
                    max_agent_no=10)
    sim._populate()
    sim._fig = plt.figure()
    sim._axes = plt.gca()
    sim.run()

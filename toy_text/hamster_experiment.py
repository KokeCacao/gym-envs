# coding=utf-8
import numpy as np
import sys
import time
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


# It does not need a map in training session, It interact with unknown map to build pi-strategy
# run in limited sensing ability
# solve local minumum problem
# safe path -> efficient path (easy to adjust: by introduce randomness)
# have weights to each obstacles

# change reward
# change probability of slippery
# change landscape

# mention Monte Carlo method, Markov chains (random sampling)

class HamsterExperimentEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    #############Configuration############
    # self.shape
    # self.start_state_index
    # Obs Location
    #############Configuration############

    def __init__(self):
        ########## YOU KNOW WHAT, I WILL TAKE MY RISK ##########
        # self.shape = (4, 12)  # shape of the map (y, x)
        # self.start_coord = (3, 0)
        # self.start_state_index = np.ravel_multi_index(self.start_coord, self.shape)  # return the id of the state on (y=3, x=0)
        # self.end_coord = (3, 11)
        # # Obs Location
        # self._obs = np.zeros(self.shape, dtype=np.bool)
        # self._obs[3, 1:-1] = True
        # self.obs_reward = -100
        # self.slippery = 0.1  # add slippery!!!!
        # self.not_moving = 1.0
        # self.not_moving_on_obs = 1.0
        # self.end_award = 1.0
        # self.step_award = -1.0

        # self.shape = (7, 12)  # shape of the map (y, x)
        # self.start_coord = (5, 0)
        # self.start_state_index = np.ravel_multi_index(self.start_coord, self.shape)  # return the id of the state on (y=3, x=0)
        # self.end_coord = (5, 11)
        # # Obs Location
        # self._obs = np.zeros(self.shape, dtype=np.bool)
        # self._obs[5, 1:-1] = True
        # self.obs_reward = -100
        # self.slippery = 0.1
        # self.not_moving = 1.0
        # self.not_moving_on_obs = 1.0
        # self.end_award = 10.0
        # self.step_award = -0.1

        # 0 = normal step
        # 1 = obs
        # 2 = start
        # 3 = end
        # 4 = rock
        self.map = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [2,1,1,1,1,1,1,1,1,1,1,3],
            [0,0,0,0,0,0,0,0,0,0,0,0],
        ])

        self.shape = self.map.shape()  # shape of the map (y, x)
        # TRY CHANGING THE COORDINATE FROM 5 TO 6
        self.start_coord = tuple(zip(*np.where(self.map == 2)))[0]
        self.start_state_index = np.ravel_multi_index(self.start_coord, self.shape)  # return the id of the state on (y=3, x=0)
        self.end_coord = tuple(zip(*np.where(self.map == 3)))[0]
        # Obs Location
        self.obs = tuple(zip(*np.where(self.map == 1)))
        self.obs_reward = -100

        self.rocky[5, 1:-1]
        self.rocky_reward = -5


        self.slippery = 0.2
        self.not_moving = 1.0
        self.not_moving_on_obs = 1.0
        self.end_award = 10.0
        self.step_award = -1.0  # STEP PENATY!

        # self.shape = (7, 12)  # shape of the map (y, x)
        # self.start_coord = (6, 0)
        # self.start_state_index = np.ravel_multi_index(self.start_coord, self.shape)  # return the id of the state on (y=3, x=0)
        # self.end_coord = (5, 11)
        # # Obs Location
        # self._obs = np.zeros(self.shape, dtype=np.bool)
        # self._obs[5, 1:-1] = True
        # self.obs_reward = -100
        # self.slippery = 0.2
        # self.not_moving = 1.0
        # self.not_moving_on_obs = 1.0
        # self.end_award = 10.0
        # self.step_award = -0.1  # STEP PENATY!

        nS = np.prod(self.shape)  # number of states
        nA = 4  # number of action
        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)  # from id=s to position=coord
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(HamsterExperimentEnv, self).__init__(nS, nA, P, isd)

    def get_start(self):
        return self.start_coord
    def get_start_index(self):
        return self.start_state_index
    def get_end(self):
        return self.end_coord
    def get_shape(self):
        return self.shape

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param potential incorrect coord
        :return: the correct coord in the boarder where it should be
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0, but maybe slippery.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        old_delta = delta
        choice = [delta, [-1, 0], [0, 1], [1, 0], [0, -1]]
        individual_slippery = self.slippery/4
        i = np.random.choice([0,1,2,3,4], p=[1-self.slippery, individual_slippery, individual_slippery, individual_slippery, individual_slippery])
        delta = choice[i]
        if delta != old_delta: print("slippery!\n")
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)  # convert a list of float to a list of int
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)  # get its state id
        if tuple(new_position) in self.obs:
            return [(self.not_moving_on_obs, self.start_state_index, self.obs_reward, False)]  # return to start, give reward -100
        is_done = tuple(new_position) == self.end_coord
        if self.rocky[tuple(new_position)]:
            return [(self.not_moving_on_obs, new_state, self.rocky_reward, is_done)]
        if is_done: return [(self.not_moving, new_state, self.end_award, is_done)]
        return [(self.not_moving, new_state, self.step_award, is_done)]

    def render(self, mode='human'):
        outfile = sys.stdout
        outfile.write('\rv1.1\n')
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " ★ "
            # Print terminal state
            elif position == self.end_coord:
                output = " √ "
            elif position in self.obs:
                output = " × "
            else:
                output = " □ "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')


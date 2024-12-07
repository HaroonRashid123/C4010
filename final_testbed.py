'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
            Joshua Olojede 101182941
'''

import numpy as np
import time

import gym
import chess
import random
# import gym_chess

# Modify gym environment to support Chess960
class Chess960Env(gym.Env):
    def __init__(self):
        # super().__init__()
        self.board = chess.Board(chess960=True)

    def reset(self):
        sp = random.randint(0, 959)
        self.board = chess.Board.from_chess960_pos(sp)
        return str(self.board)

def _runAlgorithms():

    gamma = 0.9
    step_size = 0.9
    epsilon = 0.9
    max_episode = 1000
    max_model_step = 10

    # Create a Chess960 environment
    env = Chess960Env()
    # env.reset()
    print(env.reset())
    env.reset()
    print(env.reset())
    
if __name__ == "__main__":
    # Testing Functions
    print('testing')
    _runAlgorithms()
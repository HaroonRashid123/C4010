'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
'''

import numpy as np
import time

import final_codes as fc
import CustomChess as cc

def _runAlgorithms():

    gamma = 0.9
    step_size = 0.9
    epsilon = 0.9
    max_episode = 1000
    max_model_step = 10

    env = cc.CustomChess()

    # Algorithms
    Pi, q = fc.QLearning(env, gamma, step_size, epsilon, max_episode)
    print(Pi)
    print()
    print(q)


if __name__ == "__main__":
    # Testing Functions
    print('testing')
    _runAlgorithms()
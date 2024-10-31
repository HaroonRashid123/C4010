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
import mini_chess as mc

def _runExperiments():

    goal = 0
    gamma = 0.9
    step_size = 0.9
    epsilon = 0.9
    max_episode = 1000
    max_model_step = 10

    env = mc.CustomChess6x6()

    # Main loop to demonstrate integration
    custom_board = mc.CustomChess6x6()
    # custom_board.display()

    while True:  # Run until there's a winner
        mc.play_chess(custom_board)
        custom_board.display()
        
        winner = custom_board.check_winner()
        if winner:
            print(winner)
            break

if __name__ == "__main__":
    # Testing Functions
    print('testing')
    _runExperiments()
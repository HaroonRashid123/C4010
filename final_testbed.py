'''
    COMP 4010 F24 - Final Project
    Carleton University
'''

import numpy as np
import time

import final_codes as fc
import mini_chess as mc

game = pyspiel.load_game("chess")
state = game.new_initial_state()

while not state.is_terminal():
  state.apply_action(np.random.choice(state.legal_actions()))
  print(str(state) + '\n')

if __name__ == "__main__":
    # Testing Functions
    print('testing')
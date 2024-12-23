'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
            Joshua Olojede 101182941
            Owen Yang 101185223
            Haroon Rashid 101183600
'''

import numpy as np
import time

import gym
import chess
import random
import final_codes as fc
import matplotlib.pyplot as plt





class Chess960Env(gym.Env):
    def __init__(self):
        self.board = chess.Board(chess960=True)
        self.n_states = 64 * 12
        self.n_actions = 64 * 64
        self.action_space = gym.spaces.Discrete(self.n_actions) 

    

    def reset(self):
        sp = random.randint(0, 959)
        self.board = chess.Board.from_chess960_pos(sp)
        return str(self.board)

    def step(self, action):
        
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)

        legal_moves = list(self.board.legal_moves)
        if any(m.from_square == move.from_square and m.to_square == move.to_square for m in legal_moves):
            self.board.push(move)
            
            
            if self.board.is_game_over():
                reward = 1.0 if self.board.is_checkmate() else 0 
                return str(self.board), reward, True, False, {}
            
            
            opponent_moves = list(self.board.legal_moves)
            if opponent_moves:
                opponent_move = random.choice(opponent_moves)
                self.board.push(opponent_move)
            
            
            reward = 0 
            if self.board.is_checkmate():
                reward = -1.0  
            done = self.board.is_game_over()
            
            return str(self.board), reward, done, False, {}
        else:
            return str(self.board), -1.0, True, False, {}

    def get_possible_actions(self):
        legal_moves = list(self.board.legal_moves)
        actions = []
        for move in legal_moves:
            action_idx = move.from_square * 64 + move.to_square
            actions.append(action_idx)
        return actions

    def encode_state(self, state_str):
        return hash(state_str) % self.n_states

def _runAlgorithms():
    gamma = 0.9
    step_size = 0.9
    epsilon = 0.9
    max_episode = 1000
    max_model_step = 10

    # Create a Chess960 environment
    env = Chess960Env()
    print(env.reset())

    # Test Dyna-Q
    # print("\nTesting Dyna-Q...")
    # start_time = time.time()
    # Q_dyna = fc.DynaQ(env, 
    #                   gamma=gamma, 
    #                   step_size=step_size, 
    #                   epsilon=epsilon,
    #                   max_episode=200, 
    #                   max_model_step=max_model_step)
    # print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Test Q-Learning with Function Approximation
    print("\nTesting Q-Learning with Function Approximation...")
    start_time = time.time()
    
    # Create simple featurizer
    class SimpleFeaturizer:
        def __init__(self, n_features=100):
            self.n_features = n_features
            
        def featurize(self, state):
            
            state_hash = hash(state)
            features = np.zeros(self.n_features)
            for i in range(self.n_features):
                features[i] = ((state_hash + i) % 2) * 2 - 1
            return features

    
    def eval_func(env, featurizer, W, policy_func):
        total_reward = 0
        state = env.reset()
        state = featurizer.featurize(state)
        done = False
        max_steps = 400
        steps = 0
        
        while not done and steps < max_steps:
            steps += 1
            action = policy_func(state, W)
            if action is None:  
                break
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if not done:
                state = featurizer.featurize(next_state)
        
        return total_reward
 
    featurizer = SimpleFeaturizer(n_features=100)
    env.action_space = gym.spaces.Discrete(env.n_actions) 

    W, eval_returns = fc.QLearningFA(
        env=env,
        featurizer=featurizer,
        eval_func=eval_func,
        gamma=gamma,
        step_size=0.01, 
        epsilon=epsilon,
        max_episodes=500,
        evaluate_every=20
    )

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    

if __name__ == "__main__":
    print('testing')
    _runAlgorithms()
   

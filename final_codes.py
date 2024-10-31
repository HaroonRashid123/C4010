'''
    COMP 4010 [Fall 2024]
    Carleton University

    Project Details:
        ~ RL Chess Agent ~
        Date: October 15, 2024

        Group Members:
            Kyle Eng 101192595
'''

# ==================================================================================================================================================================================== #

import numpy as np
import random
import time

# ==================================================================================================================================================================================== #

# ============================================================ #
#                      MC (EXPLORING STARTS)                   #
# ============================================================ #
def MC_ExploringStarts(env, gamma, step_size, epsilon, max_episode):
    return -1

# ============================================================ #
#                         MC (E-SOFT)                          #
# ============================================================ #
def MC_ESoft(env, gamma, step_size, epsilon, max_episode):
    return -1

# ============================================================ #
#                         EXPECTED SARSA                       #
# ============================================================ #
def ExpectedSARSA(env, gamma, step_size, epsilon, max_episode):
    return -1

# ============================================================ #
#                          Q-LEARNING                          #
# ============================================================ #
def QLearning(env, gamma, step_size, epsilon, max_episode):
    # Initialize Q
    q = np.zeros((env.n_states, env.n_actions))
    
    for _ in range(max_episode):
        state, _ = env.reset()
        while True:
            # Choose e-greedy action
            actions = env.get_possible_actions()
            if random.random() < epsilon:
                # Choose random action
                action = random.choice(actions)
            else:
                # Randomly select one of the best actions
                top_actions = np.flatnonzero(q[state] == np.amax(q[state]))
                action = random.choice(top_actions)
        
            # get S' and reward
            state_next, reward, terminated, _, _ = env.step(action)
            
            #Update Q(s,a)
            q[state, action] += step_size*(reward + gamma*np.amax(q[state_next]) - q[state, action])

            # Go to next state
            state = state_next

            # Check if goal has been reached (Game is over)
            game_over, winner = env.check_winner()
            if (game_over):
                break
        
    # After convergence, Find Optimal Policy
    Pi = np.zeros((env.n_states, env.n_states*env.n_actions))
    optimal_actions = np.argmax(q, axis=1) + (np.arange(env.n_states) * env.n_actions)
    Pi[np.arange(env.n_states), optimal_actions] = 1.0
    q = q.reshape(-1, 1)

    return Pi, q

# ============================================================ #
#                           DYNA-Q                             #
# ============================================================ #
def DynaQ(env, gamma, step_size, epsilon, max_episode, max_model_step):
    # Initialize Q, model
    q = np.zeros((env.n_states, env.n_actions))
    model = {}
    for _ in range(max_episode):
        state, _ = env.reset()
        while True:
            # Choose e-greedy action
            if random.random() < epsilon:
                # Choose random action
                action = np.random.randint(env.n_actions)
            else:
                # Randomly select one of the best actions
                top_actions = np.flatnonzero(q[state] == np.amax(q[state]))
                action = random.choice(top_actions)
        
            # get S' and reward
            state_next, reward, terminated, _, _ = env.step(action)
            
            #Update Q(s,a)
            q[state, action] += step_size*(reward + gamma*np.amax(q[state_next]) - q[state, action])

            # Add to Model
            model[(state, action)] = (reward, state_next)

            # Planning Update
            for _ in range(max_model_step):
                S, A = random.choice(list(model.keys()))
                R, S_next = model[S, A] 
                q[S, A] += step_size*(R + gamma*np.amax(q[S_next]) - q[S, A])
            
            # Go to next state
            state = state_next

            if (terminated):
                break
    
    # After convergence, Find Optimal Policy
    Pi = np.zeros((env.n_states, env.n_states*env.n_actions))
    optimal_actions = np.argmax(q, axis=1) + (np.arange(env.n_states) * env.n_actions)
    Pi[np.arange(env.n_states), optimal_actions] = 1.0
    q = q.reshape(-1, 1)

    return Pi, q

# ==================================================================================================================================================================================== #

# ============================================================ #
#                          TESTING                             #
# ============================================================ #

def runQLExperiments(env):

    def repeatExperiments(gamma=0.9, step_size=0.1, epsilon=0.1, max_episode=500):
        n_runs = 5
        RMSE = np.zeros([n_runs])
        for r in range(n_runs):
            Pi, q = QLearning(env, gamma, step_size, epsilon, max_episode)
            # TODO: compute RMSE(q, q_star)
            RMSE[r] = np.sqrt(np.mean((q - q_star) ** 2))

        # TODO: compute and return the *average* RMSE over runs
        averageRMSE = np.mean(RMSE)
        return averageRMSE

    step_size_list = [0.1, 0.2, 0.5, 0.9]
    epsilon_list = [0.05, 0.1, 0.5, 0.9]
    max_episode_list = [50, 100, 500, 1000]
    
    step_size_results = np.zeros([len(step_size_list)])
    epsilon_results = np.zeros([len(epsilon_list)])
    max_episode_results = np.zeros([len(max_episode_list)])
    
    q_star = np.load('optimal_q.npy')

    # TODO: Call repeatExperiments() with different step_size in the step_size_list,
    # *while fixing others as default*. Save the results to step_size_results.
    for i in range(len(step_size_list)):
        step_size_results[i] = repeatExperiments(step_size=step_size_list[i])

    # TODO: Call repeatExperiments() with different epsilon in the epsilon_list,
    # *while fixing others as default*. Save the results to epsilon_results.
    for i in range(len(epsilon_list)):
        epsilon_results[i] = repeatExperiments(epsilon=epsilon_list[i])

    
    # TODO: Call repeatExperiments() with different max_episode in the max_episode_list,
    # *while fixing others as default*. Save the results to max_episode_results.
    for i in range(len(max_episode_list)):
        max_episode_results[i] = repeatExperiments(max_episode=max_episode_list[i])

    return step_size_results, epsilon_results, max_episode_results

def runDynaQExperiments(env):
    
    def repeatExperiments(gamma=0.9, step_size=0.1, epsilon=0.5, max_episode=100, max_model_step=10):
        n_runs = 5
        RMSE = np.zeros([n_runs])
        for r in range(n_runs):
            Pi, q = DynaQ(env, gamma, step_size, epsilon, max_episode, max_model_step)
            # TODO: compute RMSE(q, q_star)
            RMSE[r] = np.sqrt(np.mean((q - q_star) ** 2))
        
        # TODO: compute and return the average RMSE over runs
        averageRMSE = np.mean(RMSE)
        return averageRMSE

    max_episode_list = [10, 30, 50]
    max_model_step_list = [1, 5, 10, 50]
    
    results = np.zeros([len(max_episode_list), len(max_model_step_list)])
    
    q_star = np.load('optimal_q.npy')
    
    # TODO: Call repeatExperiments() with different max_episode in
    # the max_episode_list and max_model_step in the max_model_step_list
    # *while fixing others as default*. Save the results to the results array.
    for i in range(len(max_episode_list)):
        for j in range(len(max_model_step_list)):
            results[i][j] = repeatExperiments(max_episode=max_episode_list[i], max_model_step=max_model_step_list[j])
    return results
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
'''

# ========================================================================================================================

import jax.numpy as jnp
import numpy as np

from scipy.spatial.distance import cdist

import random
import time

# ========================================================================================================================

class RbfFeaturizer():
    '''
        This class converts the raw state/obvervation features into
        RBF features. It does a z-score normalization and computes the
        Gaussian kernel values from randomly selected centers.
    '''

    def __init__(self, env, n_features=1000):
        centers = np.array([env.observation_space.sample()
                            for _ in range(n_features)])
        self._mean = np.mean(centers, axis=0, keepdims=True)
        self._std = np.std(centers, axis=0, keepdims=True)
        self._centers = (centers - self._mean) / self._std
        self.n_features = n_features

    def featurize(self, state):
        z = state[None, :] - self._mean
        z = z / self._std
        dist = cdist(z, self._centers)
        return np.exp(- (dist) ** 2).flatten()

# ========================================================================================================================

# ============================================================
#       MC (EXPLORING STARTS)
# ============================================================
def MC_ExploringStarts(env, num_episodes=500, gamma=0.9):
    Q = {}
    returns = {}

    for _ in range(num_episodes):
        state = env.reset()
        # Instead of using cc.board_LosAlamos, use Chess960 position
        sp = random.randint(0, 959)  # Random Chess960 starting position
        env.board = chess.Board.from_chess960_pos(sp)  # Set to random position
        action = random.choice(env.get_possible_actions())

        # ... rest of the function remains the same ...

        episode = []
        done = False
        while not done:
            next_state, reward, terminated, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if not terminated:
                action = random.choice(env.get_possible_actions())
            done = terminated

        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                if (s, a) not in returns:
                    returns[(s, a)] = []
                returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(returns[(s, a)])
                visited.add((s, a))

    return Q


# ============================================================
#       MC (E-SOFT)                         
# ============================================================
def MC_ESoft(env, num_episodes=500, gamma=0.9, epsilon=0.1):
    Q = {}
    returns = {}

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode = []
        done = False

        while not done:
            possible_actions = env.get_possible_actions()
            if random.uniform(0, 1) < epsilon:
                action = random.choice(possible_actions)
            else:
                q_values = [Q.get((state, a), 0) for a in possible_actions]
                action = possible_actions[np.argmax(q_values)]

            next_state, reward, terminated, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated

        G = 0
        visited = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited:
                if (s, a) not in returns:
                    returns[(s, a)] = []
                returns[(s, a)].append(G)
                Q[(s, a)] = np.mean(returns[(s, a)])
                visited.add((s, a))

    return Q


# ============================================================ #
#                         EXPECTED SARSA                       #
# ============================================================ #
def ExpectedSARSA(env, num_episodes=500, gamma=0.9, alpha=0.1, epsilon=0.1):
    Q = {}

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            possible_actions = env.get_possible_actions()
            if random.uniform(0, 1) < epsilon:
                action = random.choice(possible_actions)
            else:
                q_values = [Q.get((state, a), 0) for a in possible_actions]
                action = possible_actions[np.argmax(q_values)]

            next_state, reward, terminated, _, _ = env.step(action)

            if not terminated:
                next_q_values = [Q.get((next_state, a), 0) for a in env.get_possible_actions()]
                expected_q = (1 - epsilon) * max(next_q_values) + epsilon * np.mean(next_q_values)
            else:
                expected_q = 0

            Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward + gamma * expected_q - Q.get((state, action), 0))
            state = next_state
            done = terminated

    return Q


# ============================================================ #
#                          Q-LEARNING                          #
# ============================================================ #
def QLearning(env, gamma, step_size, epsilon, max_episode):
    # Initialize Q
    # q = np.zeros((env.n_states, env.n_actions))
    q = {}
    
    for _ in range(max_episode):
        state_tuple, _ = env.reset()
        state = env.encode_state(state_tuple)

        while True:
            # Choose e-greedy action
            if random.random() < epsilon:
                # Choose random action
                all_actions = env.get_possible_actions()
                action = random.choice(range(len(all_actions)))
            else:
                # Randomly select one of the best actions
                top_actions = np.flatnonzero(q[state] == np.amax(q[state]))
                action = random.choice(top_actions)
        
            # get S' and reward
            state_next_tuple, reward, terminated, _, _ = env.step(action)
            state_next = env.encode_state(state_next_tuple)

            #Update Q(s,a)
            q[state, action] += step_size*(reward + gamma*np.amax(q[state_next]) - q[state, action])

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


# ============================================================ #
#                           DYNA-Q                             #
# ============================================================ #
def DynaQ(env, gamma, step_size, epsilon, max_episode, max_model_step):
    Q = {}
    model = {}
    
    for episode in range(max_episode):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite games
        
        while steps < max_steps:
            steps += 1
            # Get possible actions
            possible_actions = env.get_possible_actions()
            if not possible_actions:  # If no legal moves
                break
                
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(possible_actions)
            else:
                # Get Q-values for possible actions
                q_values = [Q.get((state, a), 0.0) for a in possible_actions]
                max_q = max(q_values)
                # Get all actions with the max value (there might be multiple)
                best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
                action = random.choice(best_actions)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Update Q-value
            state_action = (state, action)
            if state_action not in Q:
                Q[state_action] = 0.0
            
            # Get next state's best Q-value
            next_possible_actions = env.get_possible_actions()
            if next_possible_actions:
                next_q_values = [Q.get((next_state, a), 0.0) for a in next_possible_actions]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0.0
            
            # Q-learning update
            Q[state_action] += step_size * (reward + gamma * max_next_q - Q[state_action])
            
            # Store in model
            model[state_action] = (next_state, reward, done)
            
            # Planning step
            if len(model) > 10:  # Only start planning after some experience
                for _ in range(max_model_step):
                    # Random sample from model
                    rand_state_action = random.choice(list(model.keys()))
                    s, a = rand_state_action
                    s_next, r, d = model[rand_state_action]
                    
                    # Get max Q-value for next state
                    next_actions = env.get_possible_actions()
                    if next_actions and not d:
                        next_qs = [Q.get((s_next, a), 0.0) for a in next_actions]
                        max_next_q = max(next_qs)
                    else:
                        max_next_q = 0.0
                    
                    # Update Q-value
                    if (s, a) not in Q:
                        Q[(s, a)] = 0.0
                    Q[(s, a)] += step_size * (r + gamma * max_next_q - Q[(s, a)])
            
            state = next_state
            if done:
                break
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {steps}")
    
    return Q

# ============================================================
#   Q-Learning with F/A                             
# ============================================================
def greedyPolicy(x, W, env):  # Added env parameter
    # Get Q-values for all actions
    q_S = W.T @ x
    
    # Get only legal actions
    legal_actions = env.get_possible_actions()
    
    if not legal_actions:
        return None
    
    # Only consider Q-values of legal actions
    legal_q_values = [(action, q_S[action]) for action in legal_actions]
    
    # Get the best legal action
    best_action = max(legal_q_values, key=lambda x: x[1])[0]
    return best_action

def QLearningFA(env, featurizer, eval_func, 
                gamma=0.99, 
                step_size=0.005, 
                epsilon=0.1, 
                max_episodes=400, 
                evaluate_every=20):
    
    W = np.random.uniform(low=-0.1, high=0.1, size=(featurizer.n_features, env.action_space.n))
    print(W.shape)
    eval_returns = []
    
    for i in range(1, max_episodes + 1):
        s = env.reset()
        s = featurizer.featurize(s)
        terminated = truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Get legal actions
            legal_actions = env.get_possible_actions()
            if not legal_actions:
                break
                
          
            if np.random.random() < epsilon:
                action_index = random.choice(legal_actions)
            else:
                action_index = greedyPolicy(s, W, env)
            
            state_next, reward, terminated, truncated, _ = env.step(action_index)
            total_reward += reward
            s_next = featurizer.featurize(state_next)

            # Update W only for legal actions
            q_sa = W[:, action_index].T @ s
            next_action = greedyPolicy(s_next, W, env)
            if next_action is not None:
                q_sa_prime = W[:, next_action].T @ s_next
                W[:, action_index] += step_size * (reward + gamma * q_sa_prime - q_sa) * s
            
            s = s_next
        
        if i % evaluate_every == 0:
            eval_return = eval_func(env, featurizer, W, lambda s, w: greedyPolicy(s, w, env))
            eval_returns.append(eval_return)
            print(f"Episode {i}, Return: {total_reward}, Eval Return: {eval_return}")
    
    return W, eval_returns


# ============================================================
#   Actor Critic using Softmax                        
# ============================================================

def softmaxProb(x, Theta, training=False):
    # Use Numpy or JAX depending on question (Training vs Checking Gradient)
    lib = np if training else jnp

    # Compute Softmax Probabilities
    h = Theta.T @ x
    m = lib.amax(h)
    h_shifted = h - m
    exp_shifted = lib.exp(h_shifted)
    softmax_probabilities = exp_shifted / lib.sum(exp_shifted)

    return softmax_probabilities

def softmaxPolicy(x, Theta):
    # Get Softmax Probabilities, (training=True) always, not used for gradient check
    softmax_probabilities = softmaxProb(x, Theta, training=True)
    
    # Sample Action a
    cumulative_probabilities = np.cumsum(softmax_probabilities)
    action_index = np.searchsorted(cumulative_probabilities, np.random.random())

    return action_index

def logSoftmaxPolicyGradient(x, a, Theta, training=False):
    # Get Softmax Probabilities
    softmax_probabilities = softmaxProb(x, Theta, training=training)

    # Compute gradient for actions B
    gradient = -(x.reshape(-1, 1)) * softmax_probabilities

    # Compute gradient for action A
    if training:
        gradient[:, a] += x
    else:
        gradient = gradient.at[:, a].add(x)
    
    return gradient

def ActorCritic(env,
                featurizer,
                eval_func,
                gamma=0.99,
                actor_step_size=0.005,
                critic_step_size=0.005,
                max_episodes=400,
                evaluate_every=20):

    # TODO: initialize actor parameters
    Theta = np.random.uniform(low=-0.1, high=0.1, size=(featurizer.n_features, env.action_space.n))

    # TODO: initialize critic parameters
    w  = np.random.uniform(low=-0.1, high=0.1, size=(featurizer.n_features))
    eval_returns = []

    for i in range(1, max_episodes + 1):
        s, info = env.reset()
        s = featurizer.featurize(s)
        terminated = truncated = False
        actor_discount = 1

        while not (terminated or truncated):
            
            # TODO: Compute essential quantities, update Theta and w
            # and proceed to next state of env

            # Choose Action, A
            a = softmaxPolicy(s, Theta)

            # Take A, observe S' and reward
            state_next, reward, terminated, truncated, info = env.step(a)
            s_next = featurizer.featurize(state_next)

            # Compute TD Error, v(ST;w) ≐ 0
            if terminated:
                TD_Error = reward + (gamma * 0) - (w.T @ s)
            else:
                TD_Error = reward + (gamma * (w.T @ s_next)) - (w.T @ s)

            # Semi-grad update critic
            w += critic_step_size * TD_Error * s

            # Policy grad update actor
            actor_discount *= gamma
            Theta += actor_step_size * TD_Error * actor_discount * logSoftmaxPolicyGradient(s, a, Theta, training=True)            

            # Proceed to next state
            s = s_next

        if i % evaluate_every == 0:
            eval_return = eval_func(env, featurizer, Theta, softmaxPolicy)
            eval_returns.append(eval_return)

    return Theta, w, eval_returns

# ==================================================================================================================================================================================== #

# ============================================================ #
#                          TESTING                             #
# ============================================================ #

def evaluate_policy(env, Pi, num_episodes=1000):
    total_rewards = []

    for _ in range(num_episodes):
        state_tuple, _ = env.reset()
        state = env.encode_state(state_tuple)
        episode_reward = 0

        while True:
            # Select the best action based on the policy
            action = np.argmax(Pi[state])
            state_next_tuple, reward, terminated, _, _ = env.step(action)
            state = env.encode_state(state_next_tuple)
            episode_reward += reward

            if terminated:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)









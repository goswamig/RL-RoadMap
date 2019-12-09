import gym
import numpy as np
import pickle

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

def discounted_rewards(r):
    # 1d array to discounted rewards
    discounted_r = np.zeros_like(r)
    running_add = 0 # ?

    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    
    return discounted_r

class agent:
    def  __init__(self, *args, **kwargs):
        if resume:
            self.model = pickle.load(open('save.p', 'rb'))
        else:
            self.model = {}
            self.model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization ?
            self.model['W2'] = np.random.randn(H) / np.sqrt(H)
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.model.items() }   # rmsprop memory
        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.model.items() }     # update buffers that add up gradients over a batch

        self.prev_x = np.zeros(D) # ? 

        # init - states, hidden layers, deriv. for backprop, rewards
        self.xs = []
        self.hs = []
        self.dlogps = []
        self.drs = []

        self.running_reward = None # ?

        self.reward_sum = 0
        self.episode_num = 0


    def policy_forward(self, x):
        h = self.model['W1'].dot(x)
        h[h < 0] = 0                # relu activation
        logp = self.model['W2'].dot(h)   # ?
        prob_up = self.sigmoid(logp)     # probability of moving up
        return prob_up, h

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))   # sigmoid activation func.

    def policy_backward(self, eph, eplogp):
        dw2 = eph.T.dot(eplogp).ravel() # ?
        dh = np.outer(eplogp, self.model['W2']) # ?
        dh[eph <= 0] = 0 # backprop relu
        dw1 = dh.T.dot(epx) # ?
        return {'W1': dw1, 'W2': dw2}

    def act(self):
        if render and episode_num % 100 == 0:
            env.render()

        # flatten the image and extract the info needed from the state
        # x - motion sense is captured using two frames
        cur_x = observation
        x = cur_x - self.prev_x
        self.prev_x = cur_x


        # forward propagation and choosing the action
        prob_up, hid = self.policy_forward(x)
        action = 2 if np.random.uniform() < prob_up else 3 # random move 
        # 2 - up and 3 - down for the GYM

        # backprop & bookkeeping and prob calculation
        xs.append(x)                # keeping track of states - observation
        hs.append(hid)              # keeping track of hidden states
        y = 1 if action == 2 else 0 # fake labeling
        dlogps.append(y - prob_up)  # log prob gradient added. - gradient that encourges the taken action to be taken (backprop)


env = gym.make('CartPole-v0')
observation = env.reset() # observations

while True:
    
    


    observation, reward, done, info = env.step(action) # given action to the env.
    reward_sum += reward # reward added to the total reward

    drs.append(reward) # record the reward - ?


    # if done resetting everything and keeping the vals
    if done:
        episode_num += 1

        epx = np.vstack(xs)             # ?
        eph = np.vstack(hs)             # ?
        epdlogp = np.vstack(dlogps)    # ?
        epr = np.vstack(drs)            # ?

        xs, hs, dlogps, drs = [], [], [], [] # Resetting the mem

    
        # compute discounted reward
        discounted_epr = discounted_rewards(epr)
        discounted_epr += discounted_epr.mean() # normalize it
        discounted_epr /= discounted_epr.std()

        epdlogp *= discounted_epr # gradient with advantage - policy gradient!
        grad = policy_backward(eph, epdlogp) # ?
        for k in model: # ?
            grad_buffer[k] += grad[k] # ?
        
        if episode_num % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / np.sqrt(rmsprop_cache[k] + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('%d resetting env. episode reward total was %f. running mean: %f' % (episode_num, reward_sum, running_reward))
        if episode_num % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = np.zeros(D)

        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_num, reward)) + ('' if reward == -1 else ' !!!!!!!!')
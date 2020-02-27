import numpy as np

class RandomWalk:

	def __init__(self, irregular_start=False, start=0, lr=0.1, gamma=0.99, n = 20):
		# num states do not include the beginning and ending states
		self.actions = ['right', 'left']
		self.n = n

		self.reset(irregular_start, start)
		
		self.lr = lr
		self.gamma = gamma
		self.n = n

		self.pos_begin = 0
		self.pos_end = n + 1

		# initializing the q values
		self.Q_Values = {}
		for s in range(self.n+2):
			self.Q_Values[s] = {}
			for action in self.actions:
				if s == self.pos_begin:
					self.Q_Values[s][action] = -1
				elif s == self.pos_end:
					self.Q_Values[s][action] = 1
				else:
					self.Q_Values[s][action] = 0

	def chooseAction(self):
		return np.random.choice(self.actions)

	def takeAction(self, action):
		new_state = self.state
		reward = 0
		if not self.end:
			if action == 'left':
				self.state -= 1
			else:
				self.state += 1

			if self.state == self.pos_begin:
				self.end = True
				reward = -1
			elif self.state == self.pos_end:
				self.end = True
				reward = 1
		return self.state, reward

	def reset(self, irregular_start=False, start=0):
		if irregular_start:
			self.state = start
		else:
			self.state = (self.n+1) // 2

		self.end = False

	def n_Step_Sarsa(self, num_episodes):
		for _ in range(num_episodes):
			self.reset()
			t = 0
			T = np.inf
			action = self.chooseAction()

			actions = [action]
			states = [self.state]
			rewards = [0]

			while True:
				if t < T:
					state, reward = self.takeAction(action)
					states.append(state)
					rewards.append(reward)
					if self.end:
						# print("End at state {} | number of states {}".format(state, len(states)))
						T = t+1
					else:
						actions.append(self.chooseAction())
				
				tau = t-self.n +1
				G = 0
				if tau >= 0:
					for i in range(tau+1, min(tau+self.n+1, T+1)):
						G += np.power(self.gamma, i-tau-1) * rewards[i]
					if tau+self.n < T:
						G += np.power(self.gamma, self.n) * self.Q_Values[states[tau+self.n]][actions[tau+self.n]]
					self.Q_Values[states[tau]][actions[tau]] += self.lr * (G - self.Q_Values[states[tau]][actions[tau]])
				
				if tau >= T-1:
					break

				t += 1
						
if __name__ == '__main__':

	actual_state_values = np.arange(-20, 22, 2) / 20.0
	episodes = 1000

	sq_errors = {}
	# ns = [i for i in range(10, 150, 5)]
	ns = [200]
	for n in ns:
		print("running estimation for step={}".format(n))
		rw = RandomWalk(n=n)
		rw.n_Step_Sarsa(episodes)
		estimate_state_values = [np.mean(list(v.values())) for v in rw.Q_Values.values()]	

		print(np.mean([er ** 2 for er in actual_state_values - np.array(estimate_state_values).reshape(len(estimate_state_values), 1)]))
from agent_dir.agent import Agent
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.layers.advanced_activations import *

EPISODES = 200000

class Agent_DQN(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_DQN,self).__init__(env)

		# if args.test_dqn:
		# 	#you can load your model here
		# 	print('loading trained model')

		##################
		# YOUR CODE HERE #
		##################
		self.env = env
		self.render = False
		self.load_model = False
		# environment settings
		self.state_size = (84, 84, 4)
		self.action_size = env.action_space.n
		# parameters about epsilon
		self.epsilon = 1.
		self.epsilon_start, self.epsilon_end = 1.0, 0.05
		self.exploration_steps = 1000000.
		self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
								  / self.exploration_steps
		# parameters about training
		self.batch_size = 32
		self.train_start = 10000#50000
		self.update_target_rate = 1000#10000
		self.discount_factor = 0.99
		self.memory = deque(maxlen=100000) #400000
		self.no_op_steps = 5
		# build
		self.model = self.build_model()
		self.target_model = self.build_model()
		self.update_target_model()

		self.optimizer = self.optimizer()

		self.sess = tf.InteractiveSession()
		K.set_session(self.sess)

		self.avg_q_max, self.avg_loss = 0, 0
		self.summary_placeholders, self.update_ops, self.summary_op = \
			self.setup_summary()
		self.summary_writer = tf.summary.FileWriter(
			'summary/breakout_ddqn', self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

		# if self.load_model:
		#     self.model.load_weights("./save_model/breakout_ddqn.h5")

		if self.load_model or args.test_dqn:
			print("loading trained model")
			self.model.load_weights("./save_model/breakout_ddqn.h5") ##D:/GoogleDrive/HW3_model

	def optimizer(self):
		a = K.placeholder(shape=(None, ), dtype='int32')
		y = K.placeholder(shape=(None, ), dtype='float32')

		py_x = self.model.output

		a_one_hot = K.one_hot(a, self.action_size)
		q_value = K.sum(py_x * a_one_hot, axis=1)
		error = K.abs(y - q_value)

		quadratic_part = K.clip(error, 0.0, 1.0)
		linear_part = error - quadratic_part
		loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

		optimizer = RMSprop(lr=1e-4, epsilon=0.01)
		updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
		train = K.function([self.model.input, a, y], [loss], updates=updates)

		return train
	# approximate Q function using Convolution Neural Network
	# state is input and Q Value of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
						 input_shape=self.state_size))
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.action_size))
		model.summary()

		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, history):
		history = np.float32(history)
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(history)
			return np.argmax(q_value[0])

	def get_action2(self, history):
		if np.random.random() < 0.001:
			return random.randrange(4)
		history = np.float32(history)
		q_value = self.model.predict(history)
		return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def replay_memory(self, history, action, reward, next_history, dead):
		self.memory.append((history, action, reward, next_history, dead))

	# pick samples randomly from replay memory (with batch_size)
	def train_replay(self):
		if len(self.memory) < self.train_start:
			return
		if self.epsilon > self.epsilon_end:
			self.epsilon -= self.epsilon_decay_step

		mini_batch = random.sample(self.memory, self.batch_size)

		history = np.zeros((self.batch_size, self.state_size[0],
							self.state_size[1], self.state_size[2]))
		next_history = np.zeros((self.batch_size, self.state_size[0],
								 self.state_size[1], self.state_size[2]))
		target = np.zeros((self.batch_size, ))
		action, reward, dead = [], [], []

		for i in range(self.batch_size):
			history[i] = np.float32(mini_batch[i][0] )#/ 255.)
			next_history[i] = np.float32(mini_batch[i][3] )#/ 255.)
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			dead.append(mini_batch[i][4])

		value = self.model.predict(next_history)
		target_value = self.target_model.predict(next_history)

		# like Q Learning, get maximum Q value at s' from target model
		for i in range(self.batch_size):
			if dead[i]:
				target[i] = reward[i]
			else:

				# selection of action is from model
				# update is from target model
				target[i] = reward[i] + self.discount_factor * \
										target_value[i][np.argmax(value[i])]

		loss = self.optimizer([history, action, target])
		self.avg_loss += loss[0]

	def save_model(self, name):
		self.model.save_weights(name)

	# make summary operators for tensorboard
	def setup_summary(self):
		episode_total_reward = tf.Variable(0.)
		episode_avg_max_q = tf.Variable(0.)
		episode_duration = tf.Variable(0.)
		episode_avg_loss = tf.Variable(0.)

		tf.summary.scalar('Total Reward/Episode', episode_total_reward)
		tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
		tf.summary.scalar('Duration/Episode', episode_duration)
		tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

		summary_vars = [episode_total_reward, episode_avg_max_q,
						episode_duration, episode_avg_loss]
		summary_placeholders = [tf.placeholder(tf.float32) for _ in
								range(len(summary_vars))]
		update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
					  range(len(summary_vars))]
		summary_op = tf.summary.merge_all()
		return summary_placeholders, update_ops, summary_op

	def write_score(self, episode, score):
		with open("./save_model/modelhistory_ddqn1_score.csv", "a") as myfile:
			myfile.write("%d, %d \n" % (episode, score))


	def init_game_setting(self):
		"""

		Testing function will call this function at the begining of new game
		Put anything you want to initialize if necessary

		"""
		##################
		# YOUR CODE HERE #
		##################
		observe = self.env.reset()
		state = np.transpose(observe,(2,1,0))
		state0 = state[0]
		# state1 = state[1]
		# state2 = state[2]
		# state3 = state[3]
		# history = np.stack((state0, state1, state2, state3), axis=2)
		history = np.stack((state0, state0, state0, state0), axis=2)
		self.history = np.reshape([history], (1, 84, 84, 4))

		# for _ in range(random.randint(1, agent.no_op_steps)):
		#     observe, _, _, _ = env.step(1)

		# pass


	def train(self):
		"""
		Implement your training algorithm here
		"""
		##################
		# YOUR CODE HERE #
		##################
		env = self.env
		# self = DQNAgent(action_size=3)

		scores, episodes, global_step = [], [], 0
		sca=0
		scaa=0
		avg_q_maxA=0
		avg_lossA=0

		for e in range(EPISODES):
			done = False
			dead = False
			# 1 episode = 5 lives
			step, score, start_life = 0, 0, 5
			observe = env.reset()

			# this is one of DeepMind's idea.
			# just do nothing at the start of episode to avoid sub-optimal
			for _ in range(random.randint(1, self.no_op_steps)):
				observe, _, _, _ = env.step(2)

			# At start of episode, there is no preceding frame
			# So just copy initial states to make history
			# state = pre_processing(observe)
			state = np.transpose(observe,(2,1,0)) ###
			state0 = state[0] ###
			# state1 = state[1]
			# state2 = state[2]
			# state3 = state[3]
			# state = np.stack
			# history = np.stack((state0, state1, state2, state3), axis=2)
			history = np.stack((state0, state0, state0, state0), axis=2)
			history = np.reshape([history], (1, 84, 84, 4))

			while not done:
				# if self.render:
				# 	env.render()
				global_step += 1
				step += 1

				# get action for the current history and go one step in environment
				action = self.get_action(history)
				# change action to real_action
				# if action == 0:
				# 	real_action = 1
				# elif action == 1:
				# 	real_action = 2
				# else:
				# 	real_action = 3

				observe, reward, done, info = env.step(action)
				# pre-process the observation --> history
				# next_state = pre_processing(observe)
				next_state = np.transpose(observe,(2,1,0))###
				next_state0 = next_state[0] ###
				# next_state2 = next_state[2]
				# next_state = np.stack((next_state0,next_state2),axis=2)
				# next_state = np.reshape([next_state], (1, 84, 84, 2))
				next_state0 = np.reshape([next_state0],(1,84,84,1))
				# next_history = np.append(next_state, history[:, :, :, :2], axis=3)
				next_history = np.append(next_state0,history[:, :, :, :3], axis=3)

				self.avg_q_max += np.amax(
					self.model.predict(np.float32(history))[0]) #/ 255.

				# if the agent missed ball, agent is dead --> episode is not over
				if start_life > info['ale.lives']:
					dead = True
					start_life = info['ale.lives']

				reward = np.clip(reward, -1., 1.) ##

				# save the sample <s, a, r, s'> to the replay memory
				self.replay_memory(history, action, reward, next_history, dead)
				# every some time interval, train model
				self.train_replay()
				# update the target model with model
				if global_step % self.update_target_rate == 0:
					self.update_target_model()

				score += reward

				# if agent is dead, then reset the history
				if dead:
					dead = False
				else:
					history = next_history

				# if done, plot the score over episodes
				if done:
					if global_step > self.train_start:
						stats = [score, self.avg_q_max / float(step), step,
								 self.avg_loss / float(step)]
						for i in range(len(stats)):
							self.sess.run(self.update_ops[i], feed_dict={
								self.summary_placeholders[i]: float(stats[i])
							})
						summary_str = self.sess.run(self.summary_op)
						self.summary_writer.add_summary(summary_str, e + 1)

					# print("episode:", e, "  score:", score, "  memory length:",
					# 	  len(self.memory), "  epsilon:", self.epsilon,
					# 	  "  global_step:", global_step, "  average_q:",
					# 	  self.avg_q_max / float(step), "  average loss:",
					# 	  self.avg_loss / float(step))
					avg_q_maxA +=self.avg_q_max
					avg_lossA += self.avg_loss
					self.avg_q_max, self.avg_loss = 0, 0
					# sca = sca*0.7 + score*0.3
					sca+=score
					

			if e % 30 == 0 :
				scaa = scaa*0.7 + (sca/30)*0.3
				self.write_score(e,scaa)
				print("episode "+str(e)+"  ====="+str(scaa)+"=====")
				print("  memory length:",
					  len(self.memory), "  epsilon:", self.epsilon,
					  "  global_step:", global_step, "  average_q:",
					  avg_q_maxA / 30, "  average loss:",
					  avg_lossA / 30)
				sca=0
				avg_lossA=0
				avg_q_maxA=0

			if e % 100 == 0:
				self.model.save("/home/leoqaz12/GoogleDrive/HW3_model/save_model/breakout_ddqn1.h5") #D:\GoogleDrive\HW3_model\save_model
			# pass


	def make_action(self, observation, test=True):
		"""
		Return predicted action of your agent

		Input:
			observation: np.array
				stack 4 last preprocessed frames, shape: (84, 84, 4)

		Return:
			action: int
				the predicted action from trained model
		"""
		##################
		# YOUR CODE HERE #
		##################

		"""#take 2 frame prediction
		next_state = np.transpose(observation,(2,1,0))###
		next_state0 = next_state[0] ###
		next_state2 = next_state[2]
		next_state = np.stack((next_state0,next_state2),axis=2)
		next_state = np.reshape([next_state], (1, 84, 84, 2))
		self.history = np.append(next_state, self.history[:, :, :, :2], axis=3)

		action = self.get_action2(self.history)


		"""#take[0]frame prediction
		next_state = np.transpose(observation,(2,1,0))
		next_state = next_state[0]
		next_state = np.reshape([next_state], (1, 84, 84, 1))
		self.history = np.append(next_state,self.history[:,:,:,:3],axis=3)

		action = self.get_action2(self.history)

		

		return action
		# return self.env.get_random_action()


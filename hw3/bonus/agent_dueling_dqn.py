from agent_dir.agent import Agent
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import *
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Flatten, Lambda, merge
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import *
from keras import backend as K

class Agent_DQN(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_DQN,self).__init__(env)

		if args.test_dqn:
			#you can load your model here
			print('loading trained model')

		##################
		# YOUR CODE HERE #
		##################
		self.env = env
		# env.render()
		self.render = False
		self.load_model = True
		# environment settings
		self.state_size = (84,84,4)#(84, 84, 16)
		self.action_size = env.action_space.n #4
		# parameters about epsilon
		self.epsilon = 1.
		self.epsilon_start, self.epsilon_end = 1.0, 0.05
		self.exploration_steps = 1000000.
		self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
								  / self.exploration_steps
		# parameters about training
		self.batch_size = 32
		self.train_start = 10000
		self.update_target_rate = 0.001 #10000
		self.discount_factor = 0.99
		self.memory = deque(maxlen=300000) ##400000
		self.no_op_steps = 5 #20
		# build
		self.model = self.build_model()
		self.count = 0 
		self.target_model = self.build_model()
		self.update_target_model()

		self.optimizer = self.optimizer()

		self.sess = tf.InteractiveSession()
		K.set_session(self.sess)

		self.avg_q_max, self.avg_loss = 0, 0
		self.summary_placeholders, self.update_ops, self.summary_op = \
			self.setup_summary()
		self.summary_writer = tf.summary.FileWriter(
			'summary/breakout_dueling_ddqn', self.sess.graph)
		self.sess.run(tf.global_variables_initializer())

		self.hist_score=[]

		if self.load_model or args.test_dqn:
			self.model.load_weights("D:/GoogleDrive/HW3_model/save_model/breakout_dueling_ddqn1.h5")



	def init_game_setting(self):
		"""

		Testing function will call this function at the begining of new game
		Put anything you want to initialize if necessary

		"""
		##################
		# YOUR CODE HERE #
		##################
		observe = self.env.reset()
		for _ in range(random.randint(1, self.no_op_steps)):
			observe, _, _, _ = self.env.step(1)
		observe = np.transpose(observe,(2,1,0))
		state = observe[0]
		history = np.stack((state, state, state, state), axis=2)
		self.history = np.reshape([history],(1,84,84,4))
		#pass

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

		optimizer = RMSprop(lr=1e-4, epsilon=0.01) #0.00025 #0.01
		updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
		train = K.function([self.model.input, a, y], [loss], updates=updates)

		return train

	def build_model(self):
		input = Input(shape=self.state_size)
		shared = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
		shared = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(shared)
		shared = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(shared)
		flatten = Flatten()(shared)

		# network separate state value and advantages
		advantage_fc = Dense(512)(flatten)
		advantage_fc = LeakyReLU()(advantage_fc)
		advantage = Dense(self.action_size)(advantage_fc)
		advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
						   output_shape=(self.action_size,))(advantage)
		#dueling
		value_fc = Dense(512)(flatten)
		value_fc = LeakyReLU()(value_fc)
		value =  Dense(1)(value_fc)
		value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
					   output_shape=(self.action_size,))(value)

		# network merged and make Q Value
		q_value = merge([value, advantage], mode='sum')
		model = Model(inputs=input, outputs=q_value)
		model.summary()

		return model
		"""
		input = Input(shape=self.state_size)
		shared = Conv2D(64, (8, 8), strides=(4, 4))(input) #, activation='relu'
		act = PReLU(alpha_initializer='zero', weights=None)(shared)
		shared = Conv2D(128, (4, 4), strides=(2, 2))(act) #, activation='relu'
		act = PReLU(alpha_initializer='zero', weights=None)(shared)
		shared = Conv2D(128, (3, 3), strides=(1, 1))(act) #, activation='relu'
		act = PReLU(alpha_initializer='zero', weights=None)(shared)
		flatten = Flatten()(act)#(shared)

		# network separate state value and advantages
		advantage_fc = Dense(2048)(flatten) #, activation='relu'
		acted = LeakyReLU()(advantage_fc)
		adv = Dense(2048)(acted)
		acted = LeakyReLU()(adv)
		adv = Dense(2048)(acted)
		acted = LeakyReLU()(adv)
		adv = Dense(1024)(acted)
		acted = LeakyReLU()(adv)
		adv = Dense(1024)(acted)
		acted = LeakyReLU()(adv)
		adv = Dense(512)(acted)
		acted = LeakyReLU()(adv)
		advantage = Dense(self.action_size)(acted)#(advantage_fc)
		advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
						   output_shape=(self.action_size,))(advantage)

		value_fc = Dense(2048, activation='relu')(flatten)
		acts = LeakyReLU()(value_fc)
		advs = Dense(2048)(acts)
		acts = LeakyReLU()(advs)
		advs = Dense(2048)(acts)
		acts = LeakyReLU()(advs)
		advs = Dense(1024)(acts)
		acts = LeakyReLU()(advs)
		advs = Dense(1024)(acts)
		acts = LeakyReLU()(advs)
		advs = Dense(512)(acts)
		acts = LeakyReLU()(advs)

		value =  Dense(1)(acts)#(value_fc)
		value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
					   output_shape=(self.action_size,))(value)

		# network merged and make Q Value
		q_value = merge([value, advantage], mode='sum')
		model = Model(inputs=input, outputs=q_value)
		model.summary()
		
		return model
		"""

	def build_model2(self):
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

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, history):
		history = np.float32(history) ##/255.0
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(history)
			return np.argmax(q_value[0])
	def get_action2(self,history):
		if np.random.random() < 0.001:
			return random.randrange(3)
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
			history[i] = np.float32(mini_batch[i][0] ) 
			next_history[i] = np.float32(mini_batch[i][3] )
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			dead.append(mini_batch[i][4])

		value = self.model.predict(history)
		target_value = self.target_model.predict(next_history)

		# like Q Learning, get maximum Q value at s'
		# But from target model
		for i in range(self.batch_size):
			if dead[i]:
				target[i] = reward[i]
			else:
				target[i] = reward[i] + self.discount_factor * \
										target_value[i][np.argmax(value[i])]

		loss = self.optimizer([history, action, target])
		self.avg_loss += loss[0]

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
		with open("./save_model/modelhistory_dueling_ddqn_score2.csv", "a") as myfile:
			myfile.write("%d, %d \n" % (episode, score))


	def train(self):
		"""
		Implement your training algorithm here
		"""
		##################
		# YOUR CODE HERE #
		##################
		scores, episodes, global_step = [], [], 0
		sca=0
		scaa=0
		avg_q_maxA=0
		avg_lossA=0

		for e in range(5000000):
			done = False
			dead = False
			# 1 episode = 5 lives
			step, score, start_life = 0, 0, 5
			observe = self.env.reset()  #84,84,4
			
			for _ in range(random.randint(1, self.no_op_steps)):
				observe, _, _, _ = self.env.step(2)

			#history = observe

			#history = np.reshape([observe], (1, 84, 84, 4))
			state = np.transpose(observe,(2,1,0))
			state = state[0]
			history = np.stack((state, state, state, state), axis=2)
			history = np.reshape([history], (1, 84, 84, 4))
			
			#for i in range(12):
				#history = np.append(history, np.expand_dims(history[:, :, :, 3], axis=3), axis=3)

			while not done:


				global_step += 1
				step += 1


				action = self.get_action(history)
				# if action == 0: real_action = 1 ###
				# elif action == 1: real_action = 2 ###
				# else: real_action = 3 ###

				observe, reward, done, info = self.env.step(action) #real_action
				# pre-process the observation --> history
				#next_state = pre_processing(observe) ##
				# observe = np.swapaxes(observe,0,2)
				observe = np.transpose(observe,(2,1,0))
				next_state = observe[0]
				next_state = np.reshape([next_state], (1, 84, 84, 1))
				next_history = np.append(next_state, history[:, :, :, :3], axis=3)
				#next_states = observe
				#next_state = np.reshape([next_states], (1, 84, 84, 4))
				#next_history = np.append(next_state, history[:, :, :, :12], axis=3)#, axis=3)
				# next_state = np.reshape([next_states[1]], (1, 84, 84, 1))
				# next_history = np.append(next_state, history[:, :, :, :3], axis=3)
				# next_state = np.reshape([next_states[2]], (1, 84, 84, 1))
				# next_history = np.append(next_state, history[:, :, :, :3], axis=3)
				# next_state = np.reshape([next_states[3]], (1, 84, 84, 1))
				# next_history = np.append(next_state, history[:, :, :, :3], axis=3)			
				self.avg_q_max += np.amax(self.model.predict(np.float32(history))[0]) #/255.
				if start_life > info['ale.lives']:
					dead = True
					start_life = info['ale.lives']
				
				reward = np.clip(reward, -1., 1.)


				self.replay_memory(history, action, reward, next_history, dead)
				# every some time interval, train model
				self.train_replay()

				if global_step % self.update_target_rate == 0:
					self.update_target_model()
				score += reward

				if dead:
					dead = False
				else:
					history = next_history
					
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
					sca += score
					avg_q_maxA +=self.avg_q_max
					avg_lossA += self.avg_loss

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

					#self.hist_score.append(score)
					self.avg_q_max, self.avg_loss = 0, 0

			if e % 100 == 0:
				self.model.save("D:/GoogleDrive/HW3_model/save_model/breakout_dueling_ddqn2.h5") ##


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

		"""
		state = observation
		history = np.stack((state,state,state,state),axis=2)
		history = np.reshape([history], (1, 84, 84, 16))
		#history = np.stack
		#for i in range(12):
			#history = np.append(history, np.expand_dims(history[:, :, :, 3], axis=3), axis=3) ##
		q_value = self.model.predict(history)
		"""
		# self.env.render()
		# if self.count < 2:
		# 	self.history = np.reshape([observation],(1,84,84,4))
			
		# else:
		observation = np.transpose(observation,(2,1,0))
		next_state = observation[0]
		# print(next_state.shape)
		#next_state = np.transpose(next_state,(1,0))
		next_state = np.reshape([next_state], (1, 84, 84, 1))
		self.history = np.append(next_state,self.history[:,:,:,:3],axis=3)
			# self.count += 1
		# self.count += 1
		action = self.get_action2(self.history)



		return action
		# return np.argmax(q_value[0])


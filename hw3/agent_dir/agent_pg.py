from agent_dir.agent import Agent
import numpy as np
import pickle as pickle
import gym
from collections import deque
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.advanced_activations import *

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.env = env
        super(Agent_PG,self).__init__(self.env)

        load_path = "./save_model/pong_rl_keras_init.h5"
        # hyperparameters
        self.state_size = 80 * 80
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.H = 512 
        self.batch_size = 16 
        self.decay_rate = 0.99 
        self.resume = False
        self.running_rewards = -21
        self.action_size = 2#env.action_space.n
        print("action size", self.action_size)
        print("state_size", self.state_size)
        self.memory = deque(maxlen=20000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05 #0.05 #0.1
        self.epsilon_decay = 0.999 #995
        
        self.D = 80*80
        if args.test_pg:
            #you can load your model here
            self.model = pickle.load(open('./models/breakout_keras_pong_pg.h5','rb'))
            print('loading trained model')
        elif self.resume:
            self.model = self._build_model() ##
            self.load(load_path)
            print("res")
        else:
            self.model = self._build_model()
        self.prev_x = None 
        self.epR=[]
        self.scores_hst=0
        
    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        # model.add(BatchNormalization())
        model.add(Conv2D(32, (6, 6), activation="relu", strides=(3, 3), padding="same", kernel_initializer="he_uniform"))
        #model.add(Conv2D(32, (6, 6), activation="selu", strides=(3, 3), padding="same", kernel_initializer="lecun_uniform"))
        model.add(Flatten())
        # model.add(BatchNormalization())
        model.add(Dense(1024, activation="relu", kernel_initializer="he_uniform")) #kernel_initialize?
        # model.add(BatchNormalization())
        model.add(Dense(512, activation="relu", kernel_initializer="he_uniform"))
        # model.add(BatchNormalization())
        model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(self.action_size, activation='softmax')) 
        opt = rmsprop(lr=self.learning_rate) #rmsprop #adam
        # See note regarding crossentropy in cartpole_reinforce.py
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
        
    def train_one(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        # rewards = rewards / np.std(rewards - np.mean(rewards))
        # rewards -= 0.3 #TOBY
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        # self.model.fit(X, Y, batch_size=32)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def sigmoid(self, x): 
        return 1.0 / (1.0 + np.exp(-x)) 
    def prepro(self, I):
        I = I[35:195] 
        I = I[::2,::2,0] 
        I[I == 144] = 0 
        I[I == 109] = 0 
        I[I != 0] = 1 
        return I.astype(np.float).ravel()
        
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
        
    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state
    
        
        
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    
    def write_score(self, episode, score):
        with open("./save_model/modelhistory_pg_numpy_score1.csv", "a") as myfile:
            myfile.write("%d, %4f \n" % (episode, score))


    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            # print(target_f[0].shape)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            # self.model.train_on_batch(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # self.model.load_weights(name)
        self.model = load_model(name)

    def save(self, name):
        # self.model.save_weights(name)
        self.model.save(name)


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        def preprop(I):
            I = I[35:195]
            I = I[::2, ::2, 0]
            I[I == 144] = 0
            I[I == 109] = 0
            I[I != 0] = 1
            return I.astype(np.float).ravel()
        
        state = self.env.reset()
        prev_x = None
        score = 0
        episode = 0
        while True:
            #.env.render()

            cur_x = preprop(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.state_size)
            prev_x = cur_x

            action, prob = self.act(x)#return this action
            state, reward, done, info = self.env.step(action)
            score += reward
            self.remember(x, action, prob, reward)

            if done:
                episode += 1
                self.train_one()
                print('Episode: %d - Score: %f.' % (episode, score)) #新增平均分數
                score = 0
                state = self.env.reset()
                prev_x = None
                if episode > 1 and episode % 10 == 0:
                    self.save('./save_model/pong_reinforce_'+"Keras_relu"+'.h5')




    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################

        cur_x = self.prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x

        aprob, h = self.policy_forward(x)
        action = 2 if 0.5 < aprob else 3 
        return action
        #return self.self.env.get_random_action()


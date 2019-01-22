
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sys import stdout
from threading import Thread
from keras.layers import *
from keras.layers.recurrent import *
from keras.layers.merge import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.convolutional_recurrent import *
from keras.layers.normalization import *
import keras
from keras.models import *
from numpy import *
from keras.callbacks import *
from keras.layers.advanced_activations import *


# In[2]:


pri_train = pd.read_csv('data/fbank/train.ark', sep=" ", header=None)


# In[3]:


col_name = ['sente_id']
for i in range(69):
    col_name.append(str(i))
pri_train.columns = col_name


# In[4]:


# label map 1
map2 = open('data/48phone_char.map', 'r')

dict2 = {}
for l in map2:
    temp = l.split()
    dict2[temp[0]] = temp[2]

# label map 2
map1 = open('data/phones/48_39.map', 'r')

dict1 = {}
for l in map1:
    temp = l.split()
    dict1[temp[0]] = dict2[temp[1]]
    


# In[5]:


pri_train_y = pd.read_csv('data/label/train.lab', header=None)

pri_train_y.columns = ['sente_id', 'label']

pri_train_y.label = pri_train_y.label.map(dict1)



# In[6]:


enc = LabelBinarizer()
enc.fit(pri_train_y['label'].unique())


# In[7]:


pri_train = pd.merge(pri_train, pri_train_y, how='left', on='sente_id')


# In[8]:


def get_sente_name(sente_id):
    return sente_id[:sente_id.rfind('_')]


# In[9]:


def transform_sentence(df, train_x, train_y, i):
    current_sent_name = ""
    current_sent = []
    current_label = []
    for index, row in df.iterrows():
        current_sent.append(row[1:70].values)
        current_label.append(enc.transform([row[70]])[0])
    temp_x = np.zeros((777, 69))
    temp_x[:len(current_sent)] = np.array(current_sent) + 6
    temp_y = np.zeros((777, 39))
    temp_y[:len(current_label)] = current_label
    train_x[i] = temp_x
    train_y[i] = temp_y
#     print("Thread", i, "finish")


# In[10]:


pri_train['sente_name'] = pri_train.sente_id.apply(get_sente_name)


# In[ ]:


# train_x = []
# train_y = []
train_x = [None] * 3696
train_y = [None] * 3696
thread_list = []
name_list = pri_train.sente_name.unique()
for index, name in enumerate(name_list):
#     temp_x, temp_y = transform_sentence(pri_train[pri_train['sente_name'] == name], train_x, train_y, index)
#     train_x.append(temp_x)
#     train_y.append(temp_y)
    temp_thread = Thread(target=transform_sentence, args=(pri_train[pri_train['sente_name'] == name], train_x, train_y, index))
    thread_list.append(temp_thread)
    temp_thread.start()
    
    stdout.write(("\rprocessed %d/3695 sentences (%s)") % (index, name))
    stdout.flush()
print("All threads started.")
    
for t in thread_list:
    t.join()
print("Finished")


# In[ ]:


train_x_1 = np.array(train_x)
train_y_1 = np.array(train_y)


# # MODEL

# In[ ]:



model = Sequential()
# model.add(Merge(input_list))
# model.add(Conv1D(512, 5, input_shape=(777,69), activation='sigmoid'))
# model.add(Reshape((777, 69,1,1), input_shape=(777,69)))
model.add(LSTM(235,dropout=0.05,recurrent_dropout=0.05,return_sequences=True,input_shape=(777,69)))
model.add(LeakyReLU())
model.add(Bidirectional(LSTM(200,dropout=0.1,recurrent_dropout=0.1,return_sequences=True)))
model.add(Dense(160))
model.add(LeakyReLU(0.3))
model.add(LSTM(120,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
# model.add(LSTM(272,dropout=0.2))
model.add(Dense(80,activation='sigmoid'))
# model.add(ConvLSTM2D(filters=40,kernel_size=(3,3),input_shape=(777,69,1,1),
#                                    padding='same',return_sequences=True,activation='sigmoid'))

# model.add(Reshape((-1,), input_shape=(777,69)))
model.add(Dense(39, activation='softmax',input_shape=(3696,777,69)))
model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
# model.add(LSTM(272, input_dim=69, return_sequences=True, dropout=0.05, recurrent_dropout=0.05))
# model.add(LeakyReLU())
# model.add(LSTM(272, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))
# model.add(LSTM(272, return_sequences=True, activation='sigmoid', dropout=0.2, recurrent_dropout=0.2))

# model.compile(optimizer='Nadam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


#model = load_model('DATA/fbank/model_LLLLD_90.h5')
# from keras.callbacks import EarlyStopping
early = EarlyStopping(patience = 2)
# model.fit(np.expand_dims(train_x_1, -1), train_y_1, epochs=30, validation_data=(np.expand_dims(train_x_1, -1), train_y_1), callbacks=[early])
model.fit(train_x_1, train_y_1, epochs=100, validation_split=0.15, callbacks=[early])


# # Save Model

# In[ ]:


model.save('rnn_80.h5')


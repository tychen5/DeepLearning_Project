
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[3]:


from keras.layers import *
from keras.layers.recurrent import *
from keras.layers.merge import *
from keras.optimizers import *
from keras.layers.convolutional import *
from keras.layers.convolutional_recurrent import *
from keras.layers.normalization import *
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.models import *
import pandas as pd
import numpy as np
from numpy import *
from sys import stdout
from keras.callbacks import *
from keras.layers.advanced_activations import *
from threading import Thread
import multiprocessing


# # Preproccess

# In[4]:


pri_train = pd.read_csv('data/fbank/train.ark', sep=" ", header=None)


# ## Joining and Mapping dataframe

# In[5]:


pri_train[pri_train.columns[1:]].describe()


# In[6]:


col_name = ['sente_id']
for i in range(69):
    col_name.append(str(i))
pri_train.columns = col_name


# In[7]:


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
    


# In[8]:


pri_train_y = pd.read_csv('data/label/train.lab', header=None)

pri_train_y.columns = ['sente_id', 'label']

pri_train_y.label = pri_train_y.label.map(dict1)



# In[9]:


enc = LabelBinarizer()
enc.fit(pri_train_y['label'].unique())


# In[10]:


pri_train = pd.merge(pri_train, pri_train_y, how='left', on='sente_id')


# ## Normalization

# In[11]:


for c in pri_train.columns[1:-1]:
    pri_train[c] = (pri_train[c] - pri_train[c].mean()) / pri_train[c].std()


# In[12]:


pri_train.describe()


# ## Transform into trainable data

# In[13]:


def get_sente_name(sente_id):
    return sente_id[:sente_id.rfind('_')]


# In[14]:


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


# In[15]:


pri_train['sente_name'] = pri_train.sente_id.apply(get_sente_name)


# In[ ]:


#把FRAME串成一句話，做PADDING，也把他SHIFT6
train_x = []
train_y = []
current_sent_name = ""
current_sent = []
current_label = []
count = -1
for index, row in pri_train.iterrows():
    temp_sent_name = row[0][:row[0].rfind('_')]  #句子名字不含frame序號
    if not current_sent_name:
        current_sent_name = temp_sent_name
    if temp_sent_name != current_sent_name:
        temp = np.zeros((777, 69))
        temp[:len(current_sent)] = np.array(current_sent) + 6
#         current_sent2 = np.array(current_sent)
#         temp2 = current_sent2[:,:,newaxis,newaxis]
#         temp = temp2.tolist()
        train_x.append(temp)
        temp = np.zeros((777, 39))
        temp[:len(current_label)] = current_label
        train_y.append(temp)
            
        current_sent = []
        current_label = []
        current_sent_name = temp_sent_name
        count += 1
        stdout.write(("\rprocessed %d/3696 sentences (%s)") % (count, current_sent_name))
        stdout.flush()
    current_sent.append(row[1:70].values)
    current_label.append(enc.transform([row[70]])[0])
    
temp = np.zeros((777, 69))
temp[:len(current_sent)] = np.array(current_sent) + 6
# current_sent2 = np.array(current_sent)
# temp2 = current_sent2[:,:,newaxis,newaxis]
# temp = temp2.tolist()
train_x.append(temp)
temp = np.zeros((777, 39))
temp[:len(current_label)] = current_label
train_y.append(temp)

current_sent = []
current_label = []
current_sent_name = temp_sent_name
count += 1
stdout.write(("\rprocessed %d/3696 sentences (%s)") % (count, current_sent_name))
stdout.flush()


# In[ ]:


train_x_1 = np.array(train_x)
train_y_1 = np.array(train_y)


# In[6]:


train_x_1 = train_x_1.reshape(3696, 777, 69, 1)


# # Build Model

# In[ ]:


# act = 'relu'
#CNN + RNN : toby10_CMDTLDD_180
act = 'linear'

cnn = Sequential()
cnn.add(Conv1D(128, 4, input_shape=(69, 1), activation=act))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling1D(pool_size=8))
cnn.add(Conv1D(128, 4, activation=act))
cnn.add(LeakyReLU(alpha=0.1))
cnn.add(MaxPooling1D(pool_size=5))
cnn.add(Flatten())
cnn.add(Dense(128, activation=act))
cnn.add(LeakyReLU(alpha=0.1))
# cnn.add(Dropout(0.2))

model = Sequential()
model.add(TimeDistributed(cnn, input_shape=(777, 69, 1)))
model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256, activation=act))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(128, activation=act))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(39, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


#model = load_model('DATA/fbank/model_LLLLD_90.h5')
# from keras.callbacks import EarlyStopping
early = EarlyStopping(patience = 2)
# model.fit(np.expand_dims(train_x_1, -1), train_y_1, epochs=30, validation_data=(np.expand_dims(train_x_1, -1), train_y_1), callbacks=[early])
model.fit(train_x_1, train_y_1, epochs=50, validation_split=0.2, callbacks=[early])


# # Save model

# In[ ]:


model.save('best_model4.h5')


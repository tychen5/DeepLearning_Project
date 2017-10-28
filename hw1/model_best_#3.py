
# coding: utf-8

# In[ ]:


import tensorflow as tf


# In[ ]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[1]:


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

# In[ ]:


pri_train = pd.read_csv('DATA/fbank/train.ark', sep=" ", header=None)


# ## Joining and Mapping dataframe

# In[ ]:


pri_train[pri_train.columns[1:]].describe()


# In[ ]:


col_name = ['sente_id']
for i in range(69):
    col_name.append(str(i))
pri_train.columns = col_name


# In[2]:


# label map 1
map2 = open('DATA/48phone_char.map', 'r')

dict2 = {}
for l in map2:
    temp = l.split()
    dict2[temp[0]] = temp[2]

# label map 2
map1 = open('DATA/48_39.map', 'r')

dict1 = {}
for l in map1:
    temp = l.split()
    dict1[temp[0]] = dict2[temp[1]]
    


# In[3]:


pri_train_y = pd.read_csv('DATA/train.lab', header=None)

pri_train_y.columns = ['sente_id', 'label']

pri_train_y.label = pri_train_y.label.map(dict1)

pri_train_y


# In[4]:


enc = LabelBinarizer()
enc.fit(pri_train_y['label'].unique())


# In[ ]:


pri_train = pd.merge(pri_train, pri_train_y, how='left', on='sente_id')


# ## Transform into trainable data

# In[ ]:


def get_sente_name(sente_id):
    return sente_id[:sente_id.rfind('_')]


# In[ ]:


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


# In[ ]:


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


train_x_1 = train_x_1.reshape(3696, 777, 69, 1, 1)


# In[7]:


train_x_1.shape


# # Build Model

# In[ ]:


model = Sequential()
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 1),
                   input_shape=(777, 69, 1, 1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

# model.add(Bidirectional(ConvLSTM2D(filters=40, kernel_size=(3, 1),
#                    padding='same', return_sequences=True)))
# model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(filters=1, kernel_size=(3, 3, 1),
               activation='relu',
               padding='same', data_format='channels_last'))
model.add(Reshape((777, -1)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(39, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


# In[ ]:


#model = load_model('DATA/fbank/model_LLLLD_90.h5')
# from keras.callbacks import EarlyStopping
early = EarlyStopping(patience = 2)
# model.fit(np.expand_dims(train_x_1, -1), train_y_1, epochs=30, validation_data=(np.expand_dims(train_x_1, -1), train_y_1), callbacks=[early])
model.fit(train_x_1, train_y_1, epochs=50, validation_split=0.2, callbacks=[early])


# In[ ]:


type(train_x)


# In[ ]:


type(train_x[1])


# In[ ]:


train_x2.shape


# # Save model

# In[ ]:


model.save('DATA/fbank/best_model#3.h5')


# # Predict

# In[ ]:


pri_test = pd.read_csv('DATA/fbank/test.ark', sep=" ", header=None)


# In[ ]:


test_x = []
predict_frame_size = []
sent_names = []
current_sent_name = ""
current_sent = []
count = 0
for index, row in pri_test.iterrows():
    temp_sent_name = row[0][:row[0].rfind('_')]
    if not current_sent_name:
        current_sent_name = temp_sent_name
    if temp_sent_name != current_sent_name:
        temp = np.zeros((777, 69))
        predict_frame_size.append(len(current_sent))
        temp[:len(current_sent)] = np.array(current_sent) + 6        
#         current_sent2 = np.array(current_sent)
#         temp2 = current_sent2[:,:,newaxis]
#         temp = temp2.tolist()   
        test_x.append(temp)
        current_sent = []
        sent_names.append(current_sent_name)
        current_sent_name = temp_sent_name
        count += 1
        stdout.write(("\rprocessed %d/592 sentences (%s)") % (count, temp_sent_name))
        stdout.flush()
    current_sent.append(row[1:70].values)

temp = np.zeros((777, 69))
predict_frame_size.append(len(current_sent))
temp[:len(current_sent)] = np.array(current_sent) + 6
# current_sent2 = np.array(current_sent)
# temp2 = current_sent2[:,:,newaxis]
# temp = temp2.tolist()
test_x.append(temp)
current_sent = []
sent_names.append(current_sent_name)
current_sent_name = temp_sent_name
count += 1
stdout.write(("\rprocessed %d/592 sentences (%s)") % (count, temp_sent_name))
stdout.flush()


# In[ ]:


test_x_1 = np.array(test_x)


# In[ ]:


test_x_1 = test_x_1.reshape(592, 777, 69, 1, 1)


# In[ ]:


res = model.predict(test_x_1)


# In[ ]:


# transform res to labels
real_res = []
for index in range(len(res)):
    sent_res = []
    for frame in range(predict_frame_size[index]):
        temp = np.zeros_like(res[index][frame])
        temp[res[index][frame].argmax()] = 1
        sent_res.append(enc.inverse_transform(np.array([temp]))[0])
    real_res.append(sent_res)


# In[ ]:


#remove only one element
for r in real_res:
#     k=0
    for l in range(len(r)):
        if  (l>0) and (l< len(r)-1)  and (r[l]!=r[l-1])  and (r[l]!=r[l+1]):  #上一個下一個都跟自己不同，而且不是第一個獲最後一個元素
            del r[l]
            l-=1
for r in real_res:
#     k=0
    for l in range(len(r)):
        if  (l>0) and (l< len(r)-1)  and (r[l]!=r[l-1])  and (r[l]!=r[l+1]):  #上一個下一個都跟自己不同，而且不是第一個獲最後一個元素
            del r[l]
            l-=1


# In[ ]:


# remove duplicate labels
final_res = []
for r in real_res:
    temp_res = []
    current_label = ""
    for l in r:
        if l != current_label:
            temp_res.append(l)
            current_label = l
    final_res.append(temp_res)


# In[ ]:


phone_sequence = []
for r in final_res:
    s = ""
    for i in range(len(r)):
        if (i == 0 or i == len(r) - 1) and r[i] == 'L':
            continue
        s += r[i]
    phone_sequence.append(s)


# In[ ]:


res_df = pd.DataFrame(columns=['id', 'phone_sequence'])


# In[ ]:


res_df['id'] = sent_names
res_df['phone_sequence'] = phone_sequence


# In[ ]:


res_df.to_csv('./best_model#3.csv', index=False, encoding='utf-8')


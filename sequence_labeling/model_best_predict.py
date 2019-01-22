
# coding: utf-8

# In[1]:


import keras
print("=====preprocessing=====")
import time
start = time.time()
import pandas as pd
import numpy as np
import sys
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
from keras.models import *
from numpy import *
from keras.callbacks import *
from keras.layers.advanced_activations import *


# In[67]:


arg1 = sys.argv[1]
arg2 = sys.argv[2]

pri_test = pd.read_csv(arg1+'fbank/test.ark', sep=" ", header=None)


# In[69]:


# label map 1
map2 = open(arg1+'48phone_char.map', 'r')
# pri1= np.load('prepare1.map')
dict2 = {}
for l in map2:
    temp = l.split()
    dict2[temp[0]] = temp[2]

# label map 2
map1 = open(arg1+'phones/48_39.map', 'r')
# pri2 = np.load('prepare2.map')
dict1 = {}
for l in map1:
    temp = l.split()
    dict1[temp[0]] = dict2[temp[1]]



# In[70]:


pri_train_y = pd.read_csv(arg1+'label/train.lab', header=None)
pri_train_y.columns = ['sente_id', 'label']
pri_train_y.label = pri_train_y.label.map(dict1)



# In[71]:


enc = LabelBinarizer()
enc.fit(pri_train_y['label'].unique())


# # MODEL

# In[ ]:


model1 = load_model('rnn_80.h5',compile=False)
model2 = load_model('cnn_180.h5',compile=False)
model3 = load_model('ConvLSTM_30.h5',compile=False)


# # Predict

# In[114]:


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
        new_list = np.array(current_sent) + 6
        temp[:len(current_sent)] = new_list        
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
new_list = np.array(current_sent) + 6
temp[:len(current_sent)] = new_list
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
print("=====predicting model please wait=====")


# In[115]:


test_x_1 = np.array(test_x)
res1 = model1.predict(test_x_1)
print("predicting model#2 please wait...")
res3 = model3.predict(test_x_1)

test_x_2 = test_x_1.reshape(592, 777, 69, 1)
print("predicting model#3 please wait.....")
res2 = model2.predict(test_x_2)
# res3 = (res1 + res2*2) / 2
# res = (((res1*2 + res2)/3)+(res3*10))/11
res = (res1*19 + res2*14 + res3*14)/47
end1 = time.time()
elapsed = int
if (int(end1 - start)) > 420: 
    print("skip model#4 and model#5 due to the time")
    flag = False
else:
    flag = True


# In[ ]:



if flag == True:
    print("Predict model #4 please wait.......")
    model4 = load_model('best_model4.h5',compile=False)
    end2 =time.time()
    if (int(end2 - start)) > 400:
        print("Skip model #5 due to the time.")
        flag == False
        res = (res1*19 + res2*14 + res3*14 + res4*7)/54

if flag == True:
    model5 = load_model('best_model5.h5',compile=False)

    res5 = model5.predict(test_x_1)

    for c in pri_test.columns[1:]:
        pri_test[c] = (pri_test[c] - pri_test[c].mean()) / pri_test[c].std()

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
            new_list = np.array(current_sent) + 6
            temp[:len(current_sent)] = new_list 
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
    new_list = np.array(current_sent) + 6
    temp[:len(current_sent)] = new_list
    test_x.append(temp)
    current_sent = []
    sent_names.append(current_sent_name)
    current_sent_name = temp_sent_name
    count += 1
    stdout.write(("\rprocessed %d/592 sentences (%s)") % (count, temp_sent_name))
    stdout.flush()

    test_x_1 = test_x_1.reshape(592, 777, 69, 1)

    res4 = model4.predict(test_x_1)

    res = (res1*19+res2*14 + res3*14+res4*7 +res5*5)/59



# In[117]:


print("creating best model csv file ^^")
# transform res to labels
real_res = []
for index in range(len(res)):
    sent_res = []
    for frame in range(predict_frame_size[index]):
        temp = np.zeros_like(res[index][frame])
        temp[res[index][frame].argmax()] = 1
        sent_res.append(enc.inverse_transform(np.array([temp]))[0])
    real_res.append(sent_res)


# In[118]:


#remove only one element

for r in real_res:
    k=0
    for l in range(len(r)):
        try:
            if  (l>0) and (l< len(r))  and (r[l]!=r[l-1])  and (r[l]!=r[l+1]):  #上一個下一個都跟自己不同，而且不是第一個獲最後一個元素

                del r[l]
                l-=1
        except:
            continue
for r in real_res:
    k=0
    for l in range(len(r)):
        try:
            if  (l>0) and (l< len(r))  and (r[l]!=r[l-1])  and (r[l]!=r[l+1]):  #上一個下一個都跟自己不同，而且不是第一個獲最後一個元素

                del r[l]
                l-=1
        except:
            continue


# In[119]:


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


# In[120]:


phone_sequence = []
for r in final_res:
    s = ""
    for i in range(len(r)):
        if (i == 0 or i == len(r) - 1) and r[i] == 'L':
            continue
        s += r[i]
    phone_sequence.append(s)


# # Write to CSV

# In[121]:


res_df = pd.DataFrame(columns=['id', 'phone_sequence'])

res_df['id'] = sent_names
res_df['phone_sequence'] = phone_sequence

res_df.to_csv(arg2, index=False, encoding='utf-8')


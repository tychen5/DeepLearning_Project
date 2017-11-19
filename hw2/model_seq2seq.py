
# coding: utf-8

# In[1]:

from model import Video_Caption_Generator
from keras.preprocessing import sequence
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import json


# In[2]:

#===================================================#
                # Global Parameters #
#===================================================#
video_train_feat_path = 'data/training_data/feat/'
video_test_feat_path = 'data/testing_data/feat/'

video_train_data_path = 'test3.csv'
video_test_data_path = 'test3_test.csv'

model_path = 'models5-1/'

#===================================================#
                # Train Parameters #
#===================================================#
word_count_threshold = 3
keep_prob = 0.5
dim_image = 4096
dim_hidden = 880

n_video_lstm_step = 80 #80
n_caption_lstm_step = 20 #20

n_epochs = 3001
batch_size = 50
learning_rate = 0.0001


# In[3]:

def get_video_train_data(video_data_path, video_feat_path):
    video_data = pd.read_csv(video_data_path, sep=',', encoding='latin-1')
    # video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data['id'].apply(lambda id: id+'.npy')
    # video_data['video_path'] = video_data.apply(lambda row: row['id']+'.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(x))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    unique_filenames = sorted(video_data['video_path'].unique())
    train_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return train_data

def get_video_test_data(video_data_path, video_feat_path):
    video_data = pd.read_csv(video_data_path, sep=',',encoding='latin-1')
    video_data['video_path'] = video_data['id'].apply(lambda id: id+'.npy')
#     video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]
    return test_data


# In[4]:

def preProBuildWordVocab(sentence_iterator, word_count_threshold=3):
    # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words size from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector


# In[ ]:

def train():
    train_data = get_video_train_data(video_train_data_path, video_train_feat_path)
    train_captions = train_data['Description'].values
    test_data = get_video_test_data(video_test_data_path, video_test_feat_path)
    test_captions = test_data['Description'].values

    captions_list = list(train_captions) + list(test_captions)
    captions = np.asarray(captions_list, dtype=np.object)

    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    captions = map(lambda x: x.replace('"', ''), captions)
    captions = map(lambda x: x.replace('\n', ''), captions)
    captions = map(lambda x: x.replace('?', ''), captions)
    captions = map(lambda x: x.replace('!', ''), captions)
    captions = map(lambda x: x.replace('\\', ''), captions)
    captions = map(lambda x: x.replace('/', ''), captions)

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=word_count_threshold)
    
    np.save("./data/wordtoix", wordtoix)
    np.save('./data/ixtoword', ixtoword)
    np.save("./data/bias_init_vector", bias_init_vector)

    model = Video_Caption_Generator(
            batch_size=batch_size,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            dim_image=dim_image,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9) #per_process_gpu_memory_fraction=0.5
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    config = tf.ConfigProto(allow_soft_placement = True) #ADD
#     config = tf.ConfigProto() #add
    config.gpu_options.allocator_type ='BFC' #add
    config.gpu_options.allow_growth = True #add
    config.gpu_options.per_process_gpu_memory_fraction = 0.98 #add
#     config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.InteractiveSession(config = config) #ADD
#     sess = tf.InteractiveSession() #ORI no config config=config

    # saver = tf.train.Saver(max_to_keep=100, write_version=1)
    saver = tf.train.Saver()
    with tf.variable_scope(tf.get_variable_scope() , reuse=tf.AUTO_REUSE):
        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(tf_loss)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
#         train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.global_variables_initializer().run()  #ORI
#     tf.global_variables_initializer().run(session=sess)

    for epoch in range(0, n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.loc[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.iloc[np.random.choice(len(x))])
        current_train_data = current_train_data.reset_index(drop=True)

        for start, end in zip(
            range(0, len(current_train_data), batch_size),
            range(batch_size, len(current_train_data), batch_size)
        ):

            start_time = time.time()

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_video_lstm_step, dim_image))
            try:
                current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
            except:
                continue

            current_video_masks = np.zeros((batch_size, n_video_lstm_step))

            for ind,feat in enumerate(current_feats_vals):
                kk = len(current_feats_vals[ind])
                current_feats[ind][:kk] = feat
                current_video_masks[ind][:kk] = 1

            current_captions = current_batch['Description'].values
            current_captions = map(lambda x: '<bos> ' + x, current_captions)
            current_captions = map(lambda x: x.replace('.', ''), current_captions)
            current_captions = map(lambda x: x.replace(',', ''), current_captions)
            current_captions = map(lambda x: x.replace('"', ''), current_captions)
            current_captions = map(lambda x: x.replace('\n', ''), current_captions)
            current_captions = map(lambda x: x.replace('?', ''), current_captions)
            current_captions = map(lambda x: x.replace('!', ''), current_captions)
            current_captions = map(lambda x: x.replace('\\', ''), current_captions)
            current_captions = map(lambda x: x.replace('/', ''), current_captions)
            
            current_captions = list(current_captions)

            for idx, each_cap in enumerate(current_captions):
                word = each_cap.lower().split(' ')
                if len(word) < n_caption_lstm_step:
                    current_captions[idx] = current_captions[idx] + ' <eos>'
                else:
                    new_word = ''
                    for i in range(n_caption_lstm_step-1):
                        new_word = new_word + word[i] + ' '
                    current_captions[idx] = new_word + '<eos>'

            current_caption_ind = []
            for cap in current_captions:
                current_word_ind = []
                for word in cap.lower().split(' '):
                    if word in wordtoix:
                        current_word_ind.append(wordtoix[word])
                    else:
                        current_word_ind.append(wordtoix['<unk>'])
                current_caption_ind.append(current_word_ind)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = list( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video: current_feats,
                tf_caption: current_caption_matrix
#                 keep_prob= 0.5 #ADD
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        
                        })

            print('idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time)))

        if np.mod(epoch, 20) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model-' + str(epoch)))


# In[ ]:

if __name__ == "__main__":
    train()


# In[ ]:




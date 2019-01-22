
# coding: utf-8

# In[1]:

from model51 import Video_Caption_Generator #???
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json


# In[2]:

#=====================================================#
                # Global Parameters #
#=====================================================#
# testing_data = sys.argv[1]
data_path = sys.argv[1]
testing_data = data_path+'peer_review_id.txt'
# video_test_feat_path = sys.argv[2]

video_test_feat_path = data_path +'peer_review/feat/'
# output_name = 'output_testset.txt'
output_name = sys.argv[2]
default_model_path = 'model-660' #???

#=====================================================#
                # Train Parameters #
#=====================================================#
dim_image = 4096
dim_hidden = 880 #???

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

batch_size = 50


# In[3]:

def test(model_path=default_model_path):
    test_videos = open(testing_data, 'r').read().split('\n')[:-1]

    ixtoword = pd.Series(np.load('ixtoword.npy').tolist())
    bias_init_vector = np.load('bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
#             n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step, #???
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, _, _ = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    try:
        print ('\n=== Use model', model_path, '===\n')
        saver.restore(sess, model_path)
    except:
#         print ('\nUse default model\n')
        saver.restore(sess, default_model_path)

    with open(output_name, 'w') as out:
        generated_sentences = []
        for idx, video in enumerate(test_videos):
            print ('video =>', video)

            video_feat_path = os.path.join(video_test_feat_path, video) + '.npy'
            video_feat = np.load(video_feat_path)[None,...]
            if video_feat.shape[1] == n_frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            else:
                continue
    
            generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
            generated_words = ixtoword[generated_word_index]
    
            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation] 
            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<unk> ', '')
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')

            print ('generated_sentence =>', generated_sentence)
            
            outputText=str(video)+","+str(generated_sentence)+"\n"
            out.write(outputText)
#             generated_sentences.append({"caption": generated_sentence, "id": video})
#         json.dump(generated_sentences, out, indent=4)
#         out.write(str(generated_sentences))


# In[4]:

if __name__ == "__main__":
    test()

# if __name__ == "__main__":
#     test()


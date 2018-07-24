# Video Captioning

Implement one seq2seq model and attention mechanism.

**Input a video and output captions of the video**

#### Dataset:

MSVD dataset

(The provided features extracted by VGG that is pretrained on the ImageNet.)

#### Evaluation:

average bleu@1 score 


## Prediction Usage
`bash ./hw2_seq2seq.sh data/ test.txt peer.txt`


會從dropbox下載model

model架構寫於model51.py當中，不論是訓練或是預測皆需要

此外也需要ixtoword, bias_init_vector檔案


#### needed library:
* python-3.6.3
* pandas-0.21
* numpy-1.13.3
* os
* time
* json
* sys
* tensorflow-1.1
* tensorflow-gpu-1.4


*train code則會讀取test3.csv以及test3_test.csv*
*需要創建models5-1的資料夾*

#### Result:

old bleu score: 0.2878

new bleu score: 0.6522

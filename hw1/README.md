# ADLxMLDS2017
HW 1 - Sequence Labeling

三種model的predict方法：

./hw1_rnn.sh data/ rnn.csv

./hw1_cnn.sh data/ cnn.csv

./hw1_best.sh data/ best.csv


使用到的python package:

keras

time

pandas

numpy

sys

sklearn.preprocessing

threading


關於best model共會用到五個model，hw1_best.sh會執行model_best_predict.py:

因為時間因素，在model_best_predict中會依照剩餘時間決定predict要用到幾個model

1.rnn_80.h5 =>model_rnn.py:

2.cnn_180.h5 =>model_cnn.py

3.ConvLSTM_30.h5 =>model_best_#3.py

若時間允許才會追加執行:

4.best_model4.h5 =>model_best_#4.py

5.best_model5.h5 =>model_best_#5.py

(該model對應的training code為箭頭後方的python檔案)

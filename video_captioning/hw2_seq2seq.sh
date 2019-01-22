#!/bin/bash
wget https://www.dropbox.com/s/g7iaes2d1ahl14o/model-660.data-00000-of-00001?dl=0 -O ./model-660.data-00000-of-00001
python seq2seq_evalT51.py $1 $2
python seq2seq_evalP51.py $1 $3

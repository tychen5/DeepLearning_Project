#!/bin/bash

wget https://www.dropbox.com/s/z51aiw95xn6oz7i/-61.data-00000-of-00001?dl=0 -O ./models/-61.data-00000-of-00001
python3.5 train_generate.py $1

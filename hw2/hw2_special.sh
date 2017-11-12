#!/bin/bash
wget https://www.dropbox.com/s/3igq1anrwaaky6y/model-220.data-00000-of-00001?dl=0 -O ./model-220.data-00000-of-00001
python special_test.py $1 $2

#!/bin/bash

cd training
python dump_rnn.py weights.hdf5 ../src/rnn_data.c ../src/rnn_data orig
cd ..
make


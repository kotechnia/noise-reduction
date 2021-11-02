#!/bin/bash
#cd /home/nia2021/workspace/dns/rnnoise
cd src

# correct 100% volume
#ffmpeg -f concat -safe 0 -i train_clean.txt -af "pan=mono|c0=c0+c1" -ac 1 -ar 48000 -y train_clean.wav
#ffmpeg -f concat -safe 0 -i train_noisy.txt -af "pan=mono|c0=c0+c1" -ac 1 -ar 48000 -y train_noisy.wav
ffmpeg -f concat -safe 0 -i train_clean.txt -af "pan=mono|c0=c0+c1" -ac 1 -f s16le -acodec pcm_s16le -ar 48000 -y train_clean.raw
ffmpeg -f concat -safe 0 -i train_noisy.txt -af "pan=mono|c0=c0+c1" -ac 1 -f s16le -acodec pcm_s16le -ar 48000 -y train_noisy.raw

CleanFileSize=$(stat -c %s "train_clean.raw")
TrainMaxCount=$((CleanFileSize/(480*2)+1))
TrainMaxCount=$((TrainMaxCount))

echo "TrainSet TrainMaxCount ${TrainMaxCount}"

./compile_mic.sh
./denoise_training_mic train_clean.raw train_noisy.raw $TrainMaxCount > train_training.f32

cd ../training
python bin2hdf5.py ../src/train_training.f32 $TrainMaxCount 87 train_training.h5

#python rnn_train.py
#python dump_rnn.py weights.hdf5 ../src/rnn_data.c ../src/rnn_data orig

cd ..
#make


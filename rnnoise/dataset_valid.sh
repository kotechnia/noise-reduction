#!/bin/bash
#cd /home/nia2021/workspace/dns/rnnoise
cd src

# correct 100% volume
#ffmpeg -f concat -safe 0 -i valid_clean.txt -af "pan=mono|c0=c0+c1" -ac 1 -ar 48000 -y valid_clean.wav
#ffmpeg -f concat -safe 0 -i valid_noisy.txt -af "pan=mono|c0=c0+c1" -ac 1 -ar 48000 -y valid_noisy.wav
ffmpeg -f concat -safe 0 -i valid_clean.txt -af "pan=mono|c0=c0+c1" -ac 1 -f s16le -acodec pcm_s16le -ar 48000 -y valid_clean.raw
ffmpeg -f concat -safe 0 -i valid_noisy.txt -af "pan=mono|c0=c0+c1" -ac 1 -f s16le -acodec pcm_s16le -ar 48000 -y valid_noisy.raw

CleanFileSize=$(stat -c %s "valid_clean.raw")
TrainMaxCount=$((CleanFileSize/(480*2)+1))
TrainMaxCount=$((TrainMaxCount))

echo "ValidSet TrainMaxCount ${TrainMaxCount}"

./compile_mic.sh
./denoise_training_mic valid_clean.raw valid_noisy.raw $TrainMaxCount > valid_training.f32

cd ../training
python bin2hdf5.py ../src/valid_training.f32 $TrainMaxCount 87 valid_training.h5

#python rnn_train.py
#python dump_rnn.py weights.hdf5 ../src/rnn_data.c ../src/rnn_data orig

cd ..
#make


# noise-reduction

❏ noise-reduction의 구조는 다음과 같습니다.    
```
📂noise-reduction
├─ 📂dataloader (data loader)
├─ 📂models (NIA2021 result models)
├─ 📂share (data share for docker dicrectory)
├─ 📂tscn (tscn main module)
├─ 📂utils (estoi, etc)
├─ 📄dataset_maker.py (tscn train input csv maker sample)
├─ 📄denoise.py (tscn noise reduction sample)
├─ 📄requirements.txt
├─ 🔊sd1.wav ( sample for run denoise.py )
├─ 🔊sn1.wav ( sample for run denoise.py )
├─ 📉dataset_nia2021_noise_reduction.csv ( AI HUB dataset list )
└─ 📄train.py (tscn train sample)
```

❏ 테스트 시스템 사양은 다음과 같습니다.    
```
Ubuntu 20.04   
Python 3.8.10 
Torch 1.9.0+cu111 
CUDA 11.1
cuDnn 8.2.0    
```
❏ 사용 라이브러리 및 프로그램입니다.

```
$ pip install –r requirement.txt
```

# 실행 방법 (예시)

❏ dataset.csv(input csv) 생성 방법입니다.
```
python dataset_maker.py \
--wav_files_dir=share/data \
--csv_save_path=share/dataset.csv
```

❏ dataset.csv는 다음과 같은 구조를 갖습니다.
|sd_file_path|sn_file_path|train_val_test|
|:--:|:--:|:--:|
|share/data/sd_file_name_1.wav|share/data/sn_file_name_1.wav|TR|
|share/data/sd_file_name_2.wav|share/data/sn_file_name_2.wav|VA|
| ... | ... |... |
|share/data/sd_file_name_n.wav|share/data/sn_file_name_n.wav|TE|

❏ 훈련 방법입니다.
```
python train.py \
--model=models/tscn \
--csv_file=share/dataset.csv \
--cme_epochs=400 \
--finetune_epochs=40 \
--csr_epochs=400 \
--batch_size=8 \
--multi_gpu=True \
--preproc_path=share/preproc 
```

❏ 소음 억제 방법입니다.
```
python denoise.py \
--model=models/tscn \
--noisy=sn1.wav \
--denoise=de1.wav \
--clean=sd1.wav

python denoise.py \
--model=models/tscn \
--csv_file=share/dataset.csv \
--output_dir=share/denoise
```

|comment|wav player|
|:--:|:--:|
|SN ( 입력 )| https://user-images.githubusercontent.com/65753560/143393711-c9ec37a0-95ef-407f-8e72-444553c43bc0.mp4 |
|TSCN ( 출력 ) | https://user-images.githubusercontent.com/65753560/143393778-9dc9331c-915a-4555-b4f8-4197a575420f.mp4 |
|SD ( 정답 )| https://user-images.githubusercontent.com/65753560/143393794-f40d689c-9892-49bc-81d4-c28a3a5aeb18.mp4 |


# NIA 2021 noise-reduction  
❏ NIA 2021 AI 학습용 데이터로 8:1:1 훈련, 검증, 실험 분할 학습 진행  
```
NIA 2021 noise-reduction 데이터 총 7500h -> train 6000h valid 750h test 750h  
```
※ 전체 데이터는 [AI - HUB](https://aihub.or.kr/)에서 받을 수 있습니다.  
  
❏ 훈련된 모델의 loss history 입니다.  
|**TSCN**|
|:--:|
|![TSCN loss history](https://user-images.githubusercontent.com/65753560/146899445-347c5b6f-d34f-47e4-b7b2-97494dbb089c.png)|




❏ 훈련된 모델의 ESTOI 수치 결과입니다.  
||**TSCN**|
|:--:|:--:|
|**ESTOI**|0.76|

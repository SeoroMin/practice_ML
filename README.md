# practice_ML

# Data collection
1) 기존 데이터

2) 추가 데이터 수집

# Data preprocessing
1) labeling


2) Data Augmentation

spo데이터는 376개, spo가 아닌 데이터는 4624개로 data imbalance현상이 일어나 augmentation을 진행.
down sampling 수행 (spo가 아닌 데이터를 spo데이터의 개수로 맞추는 방법)

# modeling
1) RNN

2) LSTM

3) Bert


# Hyper parameter tuning

# conclusion

BERT)
MAX_len=64, Batch_size=32, lr=2e-5, epoch=20, augmentation O -> 98%
MAX_len=64, Batch_size=32, lr=2e-5, epoch=20, augmentation O

# practice_ML

# Data collection
1) 기존 데이터<br>
- 기존 nsmc 네이버 리뷰 데이터 중 5000개 추출.

2) 추가 데이터 수집<br>
- 최신 리뷰 데이터 및 추가 데이터 수집을 위해 리뷰 크롤링

# Data preprocessing
1) labeling
- 스포 -> 1, 스포x -> 0으로 labeling 직접 수행(2명)

스포 기준<br>
A. 결말 및 숨은 의도 해석<br>
B. 인물의 죽음, 부활 등 생사여부 포함<br>
C. 등장 인물간의 관계 및 내용 전개 설명<br>
D. 반전 언급<br>
E. 특정 장면에 대한 표현 및 해석<br>

해당 기준에 하나라도 적용 시 스포로 판단

2) Data Augmentation
- 스포 데이터는 376개, spo가 아닌 데이터는 4624개로 data imbalance현상이 일어나 augmentation을 진행.



3) down sampling 수행 (spo가 아닌 데이터를 spo데이터의 개수로 맞추는 방법)

# modeling
1) RNN

2) LSTM

3) Bert


# Hyper parameter tuning

# conclusion

BERT)
- MAX_len=64, Batch_size=32, lr=2e-5, epoch=20, augmentation O(5개) downsampling X -> 0.98
- MAX_len=64, Batch_size=32, lr=2e-5, epoch=20, augmentation X, downsampling X -> 0.92 => 성능은 높으나 data imbalance
- MAX_len=64, Batch_size=32, lr=2e-5, epoch=20, augmentation X, downsampling O -> 0.75

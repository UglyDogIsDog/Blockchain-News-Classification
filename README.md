# SRTP---Blockchain-News-Classification

**How to Run?**
```
#prepare
bash prepare.sh
screen -S run
bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=1 -max_seq_len 32 > bert.log 2>&1 &

#run
python3 run.py --batch_size=32 learning_rate=1e-3 epoch=20
```

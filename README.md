# SRTP---Blockchain-News-Classification

**How to Run?**
```
bash prepare.sh
screen -S run
bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -num_worker=1 -max_seq_len 32 &
python3 run.py 256 1e-3
```

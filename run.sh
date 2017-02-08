rm nohup.out
export CUDA_VISIBLE_DEVICES=0; nohup python3 -u train_vgg.py &
tail -f nohup.out

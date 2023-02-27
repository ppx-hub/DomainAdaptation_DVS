python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 47 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID1=$!; 
wait ${PID1}

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 1024 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID1=$!; 
wait ${PID1}
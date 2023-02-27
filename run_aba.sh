python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --seed 42 --traindata-ratio 0.1 --smoothing 0.0 --domain-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID1=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --seed 42 --traindata-ratio 0.1 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID2=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --DVS-DA --traindata-ratio 0.1 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID3=$!; 


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --domain-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID4=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --domain-loss --semantic-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID5=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID6=$!; 
wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}





python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --seed 42 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID1=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --seed 42 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID2=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --DVS-DA --traindata-ratio 0.4 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID3=$!; 


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID4=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID5=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID6=$!; 
wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}






python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --seed 42 --traindata-ratio 0.7 --smoothing 0.0 --domain-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID1=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --seed 42 --traindata-ratio 0.7 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID2=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --DVS-DA --traindata-ratio 0.7 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID3=$!; 

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 0.7 --smoothing 0.0 --domain-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID4=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 0.7 --smoothing 0.0 --domain-loss --semantic-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID5=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 42 --num-classes 101 --traindata-ratio 0.7 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID6=$!; 
wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}





python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID1=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second&
PID2=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --DVS-DA --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID3=$!; 

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID4=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --semantic-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
PID5=$!;

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second&
PID6=$!; 
wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}
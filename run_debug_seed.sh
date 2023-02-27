python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 47 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second --semantic-loss-coefficient 0.5&
PID1=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --seed 1024 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second --semantic-loss-coefficient 0.5&
PID2=$!;

python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --seed 114514 --traindata-ratio 0.4 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second --semantic-loss-coefficient 0.5&
PID3=$!;


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 114514 --traindata-ratio 1.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second --semantic-loss-coefficient 0.5&
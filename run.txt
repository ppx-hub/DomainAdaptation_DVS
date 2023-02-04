python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --DVS-DA  --traindata-ratio 0.1 --smoothing 0.0&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.1 --smoothing 0.0&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.1 --smoothing 0.0&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.1 --smoothing 0.0&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}


python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --DVS-DA  --traindata-ratio 0.4 --smoothing 0.0&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.4 --smoothing 0.0&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.4 --smoothing 0.0&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.4 --smoothing 0.0&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}


python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --DVS-DA  --traindata-ratio 0.8 --smoothing 0.0&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.8 --smoothing 0.0&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.8 --smoothing 0.0&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 0.8 --smoothing 0.0&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}


python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --DVS-DA  --traindata-ratio 1.0 --smoothing 0.0&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 1.0 --smoothing 0.0&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 1.0 --smoothing 0.0&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 1.0 --smoothing 0.0&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}





python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --epochs 800&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --epochs 800&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --epochs 800&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --num-classes 101 --traindata-ratio 0.1 --smoothing 0.0 --epochs 800&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --epochs 800&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --epochs 800&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --epochs 800&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --num-classes 101 --traindata-ratio 0.4 --smoothing 0.0 --epochs 800&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --num-classes 101 --traindata-ratio 0.8 --smoothing 0.0 --epochs 800&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --num-classes 101 --traindata-ratio 0.8 --smoothing 0.0 --epochs 800&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --num-classes 101 --traindata-ratio 0.8 --smoothing 0.0 --epochs 800&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --num-classes 101 --traindata-ratio 0.8 --smoothing 0.0 --epochs 800&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}

python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 0 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --epochs 800&
PID1=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --epochs 800&
PID2=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 2 --domain-loss --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --epochs 800&
PID3=$!;
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 3 --domain-loss --semantic-loss --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --epochs 800&
PID4=$!;
wait ${PID1} && wait ${PID2}  && wait ${PID3} && wait ${PID4}


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --domain-loss --semantic-loss --seed 42 --batch-size 120 --DVS-DA --traindata-ratio 1.0&


python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 4 --domain-loss --semantic-loss --seed 42 --num-classes 101 --traindata-ratio 1.0 --epochs 800&
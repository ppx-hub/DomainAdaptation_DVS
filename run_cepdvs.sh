python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 20 --domain-loss --semantic-loss --domain-loss-coefficient 0.0 --semantic-loss-coefficient 0.0 --TET-loss-first --TET-loss-second --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200

python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 20 --domain-loss --semantic-loss --domain-loss-coefficient 0.5 --semantic-loss-coefficient 0.0 --TET-loss-first --TET-loss-second --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200

python main_transfer.py --model Transfer_ResNet18 --node-type LIFNode --source-dataset RGBCEPDVS --target-dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --num-classes 20 --domain-loss --semantic-loss --domain-loss-coefficient 0.5 --semantic-loss-coefficient 0.5 --TET-loss-first --TET-loss-second --smoothing 0.0 --event-size 48 --train-portion 0.5 --DVS-DA --epochs 200


python main.py --model resnet18 --node-type LIFNode --dataset RGBCEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --event-size 48 --num-classes 20 --train-portion 0.5 --TET-loss-first --TET-loss-second --epochs 200&
PID1=$!;

python main.py --model resnet18 --node-type LIFNode --dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --event-size 48 --num-classes 20 --train-portion 0.5 --TET-loss-first --TET-loss-second --DVS-DA --epochs 200&
PID2=$!;
wait ${PID1} && wait ${PID2} 

python main.py --model resnet18 --node-type LIFNode --dataset CEPDVS --step 6 --batch-size 120 --act-fun QGateGrad --device 6 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --event-size 48 --num-classes 20 --train-portion 0.5 --TET-loss-first --TET-loss-second --DVS-DA --epochs 200 --resume /home/hexiang/DomainAdaptation_DVS/Results/Baseline/resnet18-RGBCEPDVS-6-seed_42-bs_120-DA_False-ls_0.0-lr_0.005-traindataratio_1.0-TET_loss_True-refined_False/model_best.pth.tar
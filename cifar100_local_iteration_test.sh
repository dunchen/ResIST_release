# bash script to sweep over local iterations
local_iter=50
for lr in 3e-2 
do
    for trial in 4
    do
        # freeze when calling the second script so you only run one test at a time
        echo "Starting test: LR="$lr "local_iter=" $local_iter "trial=" $trial
        python3 cifar100_local_iteration_test.py --model_name $lr"_"$trial"_cifar100_ws4_ist_final_160" --lr $lr --repartition_iter $local_iter --pytorch-seed $trial --rank=0 --cuda-id=0 &
        python3 cifar100_local_iteration_test.py --model_name $lr"_"$trial"_cifar100_ws4_ist_final_160" --lr $lr --repartition_iter $local_iter --pytorch-seed $trial --rank=1 --cuda-id=1 &
        python3 cifar100_local_iteration_test.py --model_name $lr"_"$trial"_cifar100_ws4_ist_final_160" --lr $lr --repartition_iter $local_iter --pytorch-seed $trial --rank=2 --cuda-id=2 &
        python3 cifar100_local_iteration_test.py --model_name $lr"_"$trial"_cifar100_ws4_ist_final_160" --lr $lr --repartition_iter $local_iter --pytorch-seed $trial --rank=3 --cuda-id=3 &
        wait # wait for everything to complete before starting next test
        echo "Test complete!"
    done
done

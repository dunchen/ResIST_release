# bash script to sweep over local iterations
local_iter=50
for lr in 3e-2
do
    for trial in 7
    do
        # freeze when calling the second script so you only run one test at a time
        echo "Starting test: LR="$lr "local_iter=" $local_iter "trial=" $trial
        python3 cifar100_local_sgd_test.py --model_name $trial"_cifar100_lsgd_ws4_final_160" --lr $lr --repartition_iter $local_iter --rank=0 --pytorch-seed $trial --cuda-id=0 &
        python3 cifar100_local_sgd_test.py --model_name $trial"_cifar100_lsgd_ws4_final_160" --lr $lr --repartition_iter $local_iter --rank=1 --pytorch-seed $trial --cuda-id=1 &
        python3 cifar100_local_sgd_test.py --model_name $trial"_cifar100_lsgd_ws4_final_160" --lr $lr --repartition_iter $local_iter --rank=2 --pytorch-seed $trial --cuda-id=2 &
        python3 cifar100_local_sgd_test.py --model_name $trial"_cifar100_lsgd_ws4_final_160" --lr $lr --repartition_iter $local_iter --rank=3 --pytorch-seed $trial --cuda-id=3 &
        wait # wait for everything to complete before starting next test
        echo "Test complete!"
    done
done

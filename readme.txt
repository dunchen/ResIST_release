This is the ResIST code for 4 GPU distributed cifar100 experiments. "cifar100_local_iteration_test.py" is the ResIST code while "cifar100_local_sgd_test.py" is the local sgd baseline. We tested them on AWS p3.8xlarge instance for this experiment. Please use AWS AMI 18.04 image and pytorch_p36 conda enviornment.

Run "cifar100_local_iteration_test.sh" for ResIST experiment.
Run "cifar100_local_sgd_test.sh" for local sgd baseline.
Result will be stored in the "log" folder.

# ResIST&#58; Layer-Wise Decomposition of ResNets for Distributed Training



## Abstract
We propose **ResIST**, a novel distributed training protocol for Residual Networks (ResNets). **ResIST** randomly decomposes a global ResNet into several shallow sub-ResNets that are trained independently in a distributed manner for several local iterations, before having their updates synchronized and aggregated into the global model. In the next round, new sub-ResNets are randomly generated and the process repeats. By construction, per iteration, **ResIST** communicates only a small portion of network parameters to each machine and never uses the full model during training. Thus, **ResIST** reduces the communication, memory, and time requirements of ResNet training to only a fraction of the requirements of previous methods. In comparison to common protocols like data-parallel training and data-parallel training with local SGD, **ResIST** yields a decrease in wall-clock training time, while being competitive with respect to model performance.

| ![fig_1.jpg](/images/resist/resnet_ist.png) |
|:--:|
| <b>Figure 1: The ResIST model: we depict the process of partitioning the layers of a ResNet to different sub-ResNets, then aggregating the updated parameters back into the global network. Row (a) represents the original global ResNet. Row (b) shows the creation of two sub-ResNets. Observe that subnetwork 1 contains the residual blocks 1, 2 and 4, while subnetwork 2 contains the residual blocks 3, 4 and 5. Row (c) shows the reassembly of the global ResNet, after locally training subnetworks 1 and 2 for some number of local SGD iterations; residual blocks that are common across subnetworks (e.g., residual block 4) are aggregated appropriately during the reassembly.</b>|


## Introduction
In recent years, the field of Computer Vision (CV) has seen a revolution, beginning with the introduction of AlexNet during the ILSVRC2012 competition. 
Following this initial application of deep convolutional neural networks (CNNs), more modern architectures were produced, thus rapidly pushing the state of the art in image recognition. In particular, the introduction of the residual connection (ResNets) allowed these networks to be scaled to massive depths without being crippled by issues of unstable gradients during training. Such ability to train large networks was only furthered by the development of architectural advancements, like batch normalization. The capabilities of ResNets have been further expanded in recent years, but the basic ResNet architecture has remained widely-used. While ResNets have become a standard building block for the advancement of CV research, the computational requirements for training them are significant. For example, training a ResNet50 on ImageNet with a single NVIDIA M40 GPU takes 14 days. 

Therefore, distributed training with multiple GPUs is commonly adopted to speed up the training process for ResNets. Yet, such acceleration is achieved at the cost of a remarkably large number of GPUs (e.g 256 NVIDIA Tesla P100 GPU). Additionally, frequent synchronization and high communication costs create bottlenecks that hinder such methods from achieving linear speedup with respect to the number of available GPUs. Asynchronous approaches avoid the cost of synchronization, but stale updates complicate their optimization process. Other methods, such as data-parallel training with local SGD, reduce the frequency of synchronization. Similarly, model-parallel training has gained in popularity by decreasing the cost of local training between synchronization rounds.

### This Project
We focus on efficient distributed training of convolutional neural networks with residual skip connections. Our proposed methodology accelerates synchronous, distributed training by leveraging ResNet robustness to layer removal. In particular, a group of high-performing subnetworks (sub-ResNets) is created by partitioning the layers of a shared ResNet model to create multiple, shallower sub-ResNets. These sub-ResNets are then trained independently (in parallel) for several iterations before aggregating their updates into the global model and beginning the next iteration. Through the local, independent training of shallow sub-ResNets, this methodology both limits synchronization and communicates fewer parameters per synchronization cycle, thus drastically reducing communication overhead. We name this scheme **ResNet Independent Subnetwork Training** (**ResIST**).

The outcome of this work can be summarized as follows:
1. We propose a distributed training scheme for ResNets, dubbed **ResIST**, that partitions the layers of a global model to multiple, shallow sub-ResNets, which are then trained independently between synchronization rounds.
2. We perform extensive ablation experiments to motivate the design choices for **ResIST**, indicating that optimal performance is achieved by i) using pre-activation ResNets, ii) scaling intermediate activations of the global network at inference time, iii) sharing layers between sub-ResNets that are sensitive to pruning, and iv) imposing a minimum depth on sub-ResNets during training.
3. **ResIST** is shown to achieve high accuracy and time efficiency in all cases. We conduct experiments on several image classification and object detection datasets, including CIFAR10/100, ImageNet, and PascalVOC.
4. We utilize **ResIST** to train numerous different ResNet architectures (e.g., ResNet101, ResNet152, and ResNet200) and provide implementations for each in PyTorch

## Methods

**ResIST** operates by partitioning the layers of a global ResNet to different, shallower sub-ResNets, training those independently, and intermittently aggregating their updates into the global model. The high-level process followed by **ResIST** is depicted in Fig 1 and outlined in more detail by Algorithm 1 below.

### Model Architecture
To achieve optimal performance with **ResIST**, the global model must be sufficiently deep.
Otherwise, sub-ResNets may become too shallow after partitioning, leading to poor performance.
For most experiments, a ResNet101 architecture is selected, which balances sufficient depth with reasonable computational complexity.

**ResIST** performs best with pre-activation ResNets. Intuitively, applying batch normalization prior to the convolution ensures that the input distribution of remaining residual blocks will remain fixed, even when certain layers are removed from the architecture.

| ![fig_2.jpg](/images/resist/resnet_model.png) |
|:--:|
| <b>Figure 2: The pre-activation ResNet101 model used in the majority of experiments. The figure identifies the convolutional blocks that are partitioned to subnetworks. The network is comprised of four major sections, each containing a certain number of convolutional blocks of equal channel dimension.</b>|


### Sub-ResNet Construction

Pruning literature has shown that strided layers, initial layers, and final layers within CNNs are sensitive to pruning. Additionally, repeated blocks of identical convolutions (i.e., equal channel size and spatial resolution) are less sensitive to pruning. Drawing upon these results, as shwon in Figure 2, **ResIST** only partitions blocks within the third section of the ResNet, while all other blocks are shared between sub-ResNets.

These blocks are chosen for partitioning because i) they account for the majority of network layers; ii) they are not strided; iii) they are located within the middle of the network (i.e., initial and final layers are excluded); and iv) they reside within a long chain of identical convolutions.

By partitioning only these blocks, **ResIST** allows sub-ResNets to be shallower than the global network, while maintaining high performance.

The process of constructing sub-ResNets follows a simple procedure, depicted in Fig 1.
As shown in the transition from row (a) to (b) within Fig. 1, indices of partitioned layers within the global model are randomly permuted and distributed to sub-ResNets in a round-robin fashion. Each sub-ResNet receives an equal number of convolutional blocks (e.g., see row (b) within Fig. 1). In certain cases, residual blocks may be simultaneously partitioned to multiple sub-ResNets to ensure sufficient depth (e.g., see block 4 in Fig. 1.

**ResIST** produces subnetworks with 1/S of the global model depth, where S represents the number of independently-trained sub-ResNets.

The shallow sub-ResNets created by **ResIST** accelerate training and reduce communication in comparison to methods that communicate and train the full model.
Table 1 shows the comparison of local SGD to **ResIST** with respect to the amount of data communicated during each synchronization round for different numbers of machines, highlighting the superior communication-efficiency of **ResIST**.

{% include image.html url="/images/resist/table_1.png" description="Table 1: Reports the amount of data communicated during each communication round (in GB) of both local SGD and ResIST across different numbers of machines with ResNet101." %}

### Distributed Training

The **ResIST** training procedure is outlined in Algorithm 1.

After constructing the sub-ResNets (i.e., **subResNets** in Algorithm 1), they are trained independently in a distributed manner (i.e., each on separate GPUs with different batches of data) for l iterations.

Following independent training, the updates from each sub-ResNet are aggregated into the global model. Aggregation (i.e., **aggregate** in Algorithm 1) sets each global network parameter to its average value across the sub-ResNets to which it was partitioned.
If a parameter is only partitioned to a single sub-ResNet, aggregation simplifies to copying the parameter into the global model.
After aggregation, layers from the global model are re-partitioned randomly to create a new group of sub-ResNets, and this entire process is repeated.

| ![fig_3.jpg](/images/resist/Decentralized.png) |
|:--:|
| <b>Figure 3: A depiction of the decentralized repartition procedure. This example partitions a ResNet with eight blocks into four different sub-ResNets. The blue-green-red squares dictate the data that lies per worker; the orange column dictates the last classification layer. As seen in the figure, each worker (from initialization partition to local training and decentralized repartition) is responsible for only a fraction of parameters of the whole network. The whole ResNet is never fully stored, communicated or updated on a single worker during training.</b>|

### Implementation Details

We provide an implementation of **ResIST** in PyTorch, using the NCCL communication package. 
We use basic **broadcast** and **reduce** operations for communicating blocks in the third section and **all reduce** for blocks in other sections.

We adopt the same communication procedure for the local SGD baseline (i.e., **broadcast** and **reduce** for the third section and **all reduce** for others) to ensure fair comparison.  

**The implementation of ResIST is decentralized, meaning that it does not assume a single, central parameter server.**

As shown in Fig. 3, during the synchronization and repartition step following local training, each sub-ResNet will directly send each of its locally-updated blocks to the designated new sub-ResNet (i.e., the parameters are not sent to an intermediate parameter server).
At any time step, each worker will only need sufficient memory to store a single sub-ResNet, thus limiting the memory requirements.
Such a decentralized implementation allows parallel communication between sub-ResNets, which leads to further speedups by preventing any single machine from causing slow-downs due to communication bottlenecks in the distributed procedure.
The implementation is easily scalable to eight or more machines, either on nodes with multiple GPUs or across distributed nodes with dedicated GPUs. 

**This work is focused on the algorithmic level of distributed ResNet training.**
**ResIST** significantly reduces the number of bits communicated at each synchronization round and accelerates local training with the use of shallow sub-ResNets.
The authors are well-aware of many highly-optimized versions of data-parallel and synchronous training methodologies. 
**ResIST** is fully compatible with these frameworks and can be further accelerated by leveraging highly-optimized distributed communication protocols at the systems level, which we leave as future work.

### Supplemental Techniques

#### Scaling Activations.

Activations must be scaled appropriately to account for the full depth of the resulting network at test time.
To handle this, the output of residual blocks in the third section of the network are scaled by 1/S, where S is the total number of sub-ResNets.
Such scaling allows the global model to perform well, despite using all layers at test time. 

#### Subnetwork Depth.
Within **ResIST**, sub-ResNets may become too shallow as the number of sub-ResNets increases.
To solve this issue, **ResIST** enforces a minimum depth requirement, which is satisfied by sharing certain blocks between multiple sub-ResNets.

#### Tuning Local Iterations.

We use a default value of l=50, as l<50 did not noticeably improve performance.
In some cases, the performance of **ResIST** can be improved by tuning $l$.
The optimal setting of $l$ within **ResIST** is further explored in our paper.

#### Local SGD Warm-up Phase.
Directly applying **ResIST** may harm performance on some large-scale datasets (e.g., ImageNet).
To solve this issue, we perform a few epochs with data parallel local SGD before training the model with **ResIST**.
By simply pre-training a model for a few epochs with local SGD, the remainder of training can be completed using **ResIST** without causing a significant performance decrease.

## Results

### Small-Scale Image Classification

| ![table_2.jpg](/images/resist/table_2.png) |
|:--:|
| <b>Table 2: Test accuracy of baseline LocalSGD versus ResIST on small-scale image classification datasets.</b>|


#### Accuracy.
The test accuracy of models trained with both **ResIST** and local SGD on small-scale image classification datasets is listed in Table 2.
**ResIST achieves comparable test accuracy in all cases where the same number of machines are used.**
Additionally, **ResIST** outperforms localSGD on CIFAR100 experiments with eight machines.
The performance of **ResIST** and local SGD are strikingly similar in terms of test accuracy.
In fact, the performance gap between the two method does not exceed 1% in any experimental setting.
Furthermore, **ResIST** performance remains stable as the number of sub-ResNets increases, allowing greater acceleration to be achieved without degraded performance (e.g., see CIFAR100 results in Table 2).
Generally, using four sub-ResNets yields the best performance with **ResIST**.

| ![table_3.jpg](/images/resist/table_2.png) |
|:--:|
| <b>Table 3: Total training time in seconds of baseline models and models trained with ResIST on small-scale image classification datasets.</b>|

#### Efficiency.
In addition to achieving comparable test accuracy to local SGD, **ResIST** significantly accelerates training.
This acceleration is due to i) fewer parameters being communicated between machines and ii) locally-trained sub-ResNets being shallower than the global model.
Wall-clock training times for four and eight machine experiments are presented in Tables 3. 
**ResIST** provides 3.58x to 3.81x speedup in comparison to local SGD.
For eight machine experiments, a significant speedup over four machine experiments is not observed due to the minimum depth requirement and a reduction in the number of local iterations to improve training stability.
We conjecture that for cases with higher communication cost at each synchronization and a similar number of synchronizations, eight worker **ResIST** could lead to more significant speedups in comparison to the four worker case. 
A visualization of the speedup provided by **ResIST** on the CIFAR10 and CIFAR100 datasets is illustrated in Fig. 4.
From these experiments, it is clear that the communication-efficiency of **ResIST** allows the benefit of more devices to be better realized in the distributed setting. 

| ![fig_4.jpg](/images/resist/cifar10_timing-1.png) |
|:--:|
| <b>Figure 4: Both methodologies complete 160 epochs of training. Accuracy values are smoothed using a 1-D gaussian filter, and shaded regions represent deviations in accuracy.</b>|

### Large-Scale Image Classification
#### Accuracy. 
The test accuracy of models trained with both **ResIST** and local SGD for different numbers of machines on the ImageNet dataset is listed in Table 4.
As can be seen, **ResIST achieves comparable test accuracy (<2% difference) to local SGD in all cases where the same number of machines are used.**
As many current image classification models overfit to the ImageNet test set and cannot generalize well to new data, models trained with both local SGD and **ResIST** are also evaluated on three different Imagenet V2 testing sets.
As shown in Table 4, **ResIST** consistently achieves comparable test accuracy in comparison to local SGD on these supplemental test sets. 

| ![table_4.jpg](/images/resist/table_4.png) |
|:--:|
| <b>Table 4: Performance of baseline models and models trained with ResIST on 1K Imagenet. MF stands for test set MatchedFrequency and was sampled to match the MTurk selection frequency distribution of the original ImageNet validation set for each class; T-0.7 stands for test set Threshold0.7 and was built by sampling ten images for each class among the candidates with selection frequency at least 0.7; TI stands for test set TopImages and contains the ten images with highest selection frequency in our candidate pool for each class.</b>|

#### Efficiency. 
As shown in Tables 4, **ResIST** significantly accelerates the ImageNet training process.
However, due to the use of fewer local iterations and the local SGD warm-up phase, the speedup provided by **ResIST** is smaller relative to experiments on small-scale datasets.
In Table 5, it is shown that **ResIST** can reduce the total communication volume during training, which is an important feature in the implementation of distributed systems with high computational costs.

| ![table_5.jpg](/images/resist/table_4.png) |
|:--:|
| <b>Table 5: Total training time on Imagenet (in hours) of models trained with both local SGD and ResIST using two and four machines to reach a fixed test accuracy.</b>|


## PReNet with GAM 

A deraining network based on the PReNet algorithm integrated with the GAM attention mechanism.

### Introduction
This thesis focuses on single-frame image de-raining research, utilizing deep learning methods and proposing a new method that enhances de-raining performance by integrating the Global Attention Mechanism (GAM) into the PReNet algorithm. PReNet is an efficient recursive network designed to remove raindrops from images, but it has some limitations in handling complex scenes. To address this issue, this paper introduces the GAM attention mechanism after each residual module, enhancing the network’s ability to perceive rainy areas and thereby more accurately removing rain streaks. Extensive experiments were con ducted on mainstream de-raining datasets such as Rain100L and Rain1400, and the results show that the new model integrated with the GAM attention mechanism significantly im proves both the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM)comparedtotheoriginal PReNetmodel. Theseresults demonstrate the effectiveness and potential application value of the GAM attention mechanism in the task of de-raining.

---

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
  <p><a href="https://arxiv.org/abs/1901.09221">[1901.09221] Progressive Image Deraining Networks: A Better and Simpler Baseline (arxiv.org)</a></p>
</div>

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
  <p><a href="https://arxiv.org/abs/2112.05561">[2112.05561] Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions(arxiv.org)</a></p>
</div>


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- MATLAB for computing [evaluation metrics](statistic/)

------

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 6             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 6                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results




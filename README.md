##                            Multi-scale face recognition model based on CrossScanViT+Resnet: A Competition Method Report

## Abstract

    This paper proposes a hybrid model for multi-scale face recognition based on the combination of ResNet50 and an improved Vision Transformer (ViT). By introducing ViT on the basis of ResNet50 to capture the global dependencies of images and improving ViT. To extract deep-level features, dilated convolution and the ASPP module are introduced to capture multi-scale information of images. Finally, the SE channel attention mechanism is used to fuse features. We conducted multiple sets of ablation experiments to verify the impact of each module on model performance. The experimental results show that this model has achieved excellent performance in complex face recognition tasks.

**keywords**：ResNet50, Vision Transformer, ASPP, Dilated convolution, SE, Face recognition

## 1 Introduction

​	As an important research direction in the field of computer vision, face recognition is widely used in fields such as security monitoring, identity verification, and human-computer interaction. However, due to factors such as illumination changes, pose changes, and occlusion, the face recognition task still faces many challenges. The rise of deep learning provides new ideas for solving these problems. Convolutional neural networks (CNNs) perform excellently in extracting local features, while the self-attention mechanism has advantages in capturing global dependencies. How to effectively combine these two methods to build a high-performance face recognition model is a current research hotspot.

​	Based on ResNet50, this approach combines an improved ViT model, introduces dilated convolution and ASPP modules, and uses the SE channel attention mechanism for feature fusion to propose a new face recognition model. Through multiple sets of ablation experiments, the contributions of each module to the model performance are verified. This model makes full use of local and global feature information and improves the accuracy of face recognition.

## 2 Related work

### 2.1 ResNet50

​	ResNet [1] successfully trained neural networks with a depth of 152 layers by introducing a residual structure. ResNet50 is one of the classic network structures with strong feature extraction capabilities. Its residual module enables the network to train more effectively and extract image features at a deeper level.

### 2.2 Dilated convolution and ASPP

​	Dilated convolutions [2] expand the receptive field without increasing the number of parameters by introducing holes (dilations) into the convolutional kernel. The ASPP (Atrous Spatial Pyramid Pooling) module [3] combines dilated convolutions with different dilation rates, enabling the capture of multi-scale contextual information and enhancing the model's ability to recognize objects of varying scales.

### 2.3 Vision Transformer (ViT)

​	ViT [4] introduces the Transformer architecture into the field of computer vision by dividing an image into fixed-size patches, which are then flattened into a sequence and fed into the Transformer. ViT has achieved performance comparable to or even better than CNNs in image classification tasks, demonstrating the effectiveness of the self-attention mechanism in visual tasks.

### 2.4 SE Channel attention mechanism

​	The Squeeze-and-Excitation (SE) network [5] introduces a channel attention mechanism that adaptively assigns weights to each channel, enhancing the representation of important features while suppressing irrelevant ones, thereby improving the overall performance of the model.

## 3 Method

### 3.1 Overview of model architecture

​	First, the input image passes through the convolutional and pooling layers of ResNet50 to extract preliminary features. Then, a Bottleneck module with dilated convolutions is introduced to expand the receptive field and extract deeper features. Next, the ASPP module is used to capture multi-scale contextual information. Subsequently, an improved ViT module is employed to capture global dependencies, and a multi-directional serialized input along with a novel feature fusion mechanism is introduced to process multi-directional inputs. Finally, the SE channel attention mechanism is applied to fuse the features, and the classification result is output through a fully connected layer.

### 3.2 ResNet50 and dilated convolution

​	ResNet50 is composed of multiple Bottleneck modules, each containing three convolutional layers. To expand the receptive field, we introduced dilated convolutions in the third and fourth stages of ResNet50, setting the stride of some convolutional layers to 1 and the dilation rate to a custom value. This approach allows for an enlarged receptive field and the acquisition of more contextual information without increasing the number of parameters.

### 3.3 ASPP moudle

The ASPP module is composed of multiple atrous convolutions with different dilation rates and a global average pooling layer, including:

- ** 1 × 1 convolution with expansion rate of 1 **
- ** 3 × 3 convolution with expansion rate of 3 **
- ** 3 × 3 convolution with expansion rate of 5 **
- ** 3 × 3 convolution with expansion rate of 7 **
- ** Global Average Pooling Layer **

The outputs of each branch are concatenated in the channel dimension, and then fused through 1×1 convolution and Batch Normalization. Finally, it passes through the ReLU activation function. The ASPP module can effectively capture features of different scales and improve the model's ability to recognize multi-scale targets.

```python
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.atrous_block5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.atrous_block7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=7, dilation=7, bias=False)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.atrous_block1(x)
        out3 = self.atrous_block3(x)
        out5 = self.atrous_block5(x)
        out7 = self.atrous_block7(x)
        out_global = self.global_avg_pool(x)
        out_global = F.interpolate(out_global, size=x.shape[2:], mode='bilinear', align_corners=False)

        out = torch.cat([out1, out3, out5, out7, out_global], dim=1)
        out = self.conv1(out)
        out = self.bn(out)
        return self.relu(out)
```



### 3.4 Improved ViT module

#### 3.4.1 Cross Scan[7]

The traditional Vision Transformer (ViT) mainly has the following disadvantages when processing images.

1. ** Insufficient direction invariance **: When traditional ViT divides the image into fixed-size patches and serializes them in row priority order, it lacks sufficient robustness to the direction changes such as rotation and flip of the image. This is in practical applications, especially in facial recognition tasks, face images with various poses and angles may be encountered, and traditional ViT is difficult to effectively cope with these changes.


In order to solve the above problems, we introduce the ** Cross Scan ** operation to transform the feature map in multiple directions, including:

1. ** Original direction **: that is, normal line priority scanning.

2. ** Horizontal and Vertical Flip **: Flip the feature map horizontally and vertically.

3. Transpose of the feature map: Swap the height and width of the feature map.
   
4. ** Flip after transposition **: Flip the transposed feature map horizontally and vertically.

   ```python
   def cross_scan(x):
       directions = []
       directions.append(x)
       flipped = torch.flip(x, dims=[2, 3]) 
       directions.append(flipped)
       transposed = x.transpose(2, 3)
       directions.append(transposed)
       transposed_flipped = flipped.transpose(2, 3)
       directions.append(transposed_flipped)
       return torch.stack(directions, dim=1) 
   ```

Through this multi-directional serialization approach, the model is capable of capturing feature information of images from various perspectives, thereby enhancing robustness to directional changes. Moreover, multi-directional serialized inputs can effectively increase the diversity and richness of features, addressing the shortcomings of traditional Vision Transformers (ViT) in terms of directional invariance and feature fusion.

```reStructuredText
We embed ViT in the middle layer. After the model extracts local features, it captures global relationships and enhances the features through deep convolution. Efficiency: Compared with embedding ViT in shallow layers, this method reduces the amount of computation while retaining enough feature representation capabilities.
```

#### 3.4.2 Feature serialization and position encoding

​	For the feature map in each direction, the convolutional operation is used for Patch Embedding to convert it into a sequence form. Specifically, a convolutional layer with a convolutional kernel size of patch_size is used to convert the feature map into an embedded representation of the specified dimension. Since the size of the input image may be different, the length of the sequence will also change. Therefore, on each forward propagation, the position code is dynamically generated according to the length of the current sequence to ensure that the position code matches the length of the sequence.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = None

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, embed_dim, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        if self.num_patches is None:
            self.num_patches = H * W
        return x
```



#### 3.4.3 Feature fusion mechanism

After obtaining the feature sequences of the four directions, we need to fuse them. To this end, we design a feature fusion mechanism based on ** FeatureFusion **. The specific steps are as follows:

1. **Global Feature Extraction**: The global average pooling of the feature maps in each direction is performed to obtain the global feature representation in each direction, with a shape of $[B, D, C] $, where $B $is the batch size, $D = 4 $is the number of directions, and $C $is the number of channels.

2. **Feature expansion and weight generation**：Flatten the global features in the four directions into a vector of $[B, 4C] $, enter it into FeatureFusion, and generate the corresponding weight $[B, 4C] $. The specific operation of FeatureFusion is:

   - **Fully connected layer 1**：Reduces the dimension of the input $[B, 4C] $to $[B,\ frac {4C} {r}] $, where $r $is the scaling factor.
   - **ReLU activation**：Introduce nonlinearity.
   - **Fully connected layer 2**：Updimension the feature back to $[B, 4C] $.
   - **Sigmoid activation**：Limit the weight to be between 0 and 1.

3. **Weight reshaping**：Reshape the weights to $[B, 4, C] $, corresponding to the channel weights in each direction.

4. **Direction weight calculation**：The weights in each direction are averaged over the channel dimension to obtain the direction weights of $[B, 4] $.

5. **Maximum weight direction selection**：For each sample, choose the directional index $i_ {\ text {max}} $with the largest average weight.

6. **特征融合**：

  For each direction $i$, calculate the weighted feature.

  - If $i = i_ {\ text {max}} $, the feature of that direction is taken directly, i.e. $fused\ _x += x_c [:, i,:] $.
  - If $i\ neq i_ {\ text {max}} $, then multiply the feature in that direction by the corresponding weight and add it up, i.e. $fused\ _x += w_c [:, i , :] * x_c [:, i,:] $.
Ultimately, the fused feature is expressed as $[B, C] $.

​	The core idea of this fusion strategy is to highlight the most important directional features while retaining useful information in other directions. Through the adaptive weight allocation of FeatureFusion, the model can automatically learn the importance of each direction, enhancing the robustness and discriminative power of feature representation.

**code implementation**：

```python
class FeatureFusion(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FeatureFusion, self).__init__()
        self.se = SEBlock1D(channels * 4, reduction)  # 四个方向

    def forward(self, x_c):
        B, D, C, H, W = x_c.shape  # x_c: [B, 4, C, H, W]
        x_c = x_c.view(B, D, C, H * W).mean(dim=-1)  # [B, 4, C]
        x_c_flat = x_c.view(B, D * C)  # [B, 4 * C]
        w_c_flat = self.se(x_c_flat)  # [B, 4 * C]
        w_c = w_c_flat.view(B, D, C)  # [B, 4, C]
        w_c_mean = w_c.mean(dim=2)  # [B, 4]
        _, i_max = torch.max(w_c_mean, dim=1)  # [B]
        fused_x = torch.zeros(B, C, device=x_c.device)
        for i in range(D):
            weight = w_c[:, i, :]  # [B, C]
            mask = (i == i_max).float().unsqueeze(1)  # [B, 1]
            fused_x += mask * x_c[:, i, :] + (1 - mask) * (weight * x_c[:, i, :])
        return fused_x  # [B, C]
```

​	Through the above fusion strategy, the model can automatically select the most discriminative direction among features in different directions for focused attention, and at the same time fuse beneficial information from other directions, enhancing the model's adaptability to complex face recognition tasks.

### 3.5 SE channel attention mechanism

​	The SE channel attention mechanism is embedded in the Bottleneck module and the feature fusion stage. In the Bottleneck module, the SE module performs channel weighting on the features extracted by convolution, enhancing the expression of important features. In the feature fusion stage, SEBlock1D is used to perform adaptive weighted fusion on multi-directional features, achieving effective integration of multi-directional features.

## 4 experiment

### 4.1 dataset

The dataset used in this study totaled ** 15,000 ** labeled face images, of which the ratio of AI-generated images to real images was basically maintained at 1:1. To make full use of the data and verify the generalization ability of the model, we adopted the following data partitioning strategies:

- **Division of training set and test set**：First, divide the total data set into a training set and a test set in an 8:2 ratio.
- ** Training dataset and validation set division **: The divided training dataset is further divided into the final training dataset and validation set in a 9:1 ratio.

### 4.2 data preprocessing

Data preprocessing is crucial for training deep learning models. We performed the following preprocessing steps on the image data:

- ** Maintain aspect ratio scaling **: Use a custom ResizeWithAspectRatio class to scale the image to the specified maximum size (e.g. 1024 pixels) while keeping the original aspect ratio unchanged. This class is defined in the dataset code to calculate the scaling ratio to ensure that the maximum edge length of the image does not exceed the set value while avoiding image distortion.

  ```python
  class ResizeWithAspectRatio:
      def __init__(self, max_size=1024):
          self.max_size = max_size

      def __call__(self, img):
          w, h = img.size
          if max(w, h) > self.max_size:
              scaling_factor = self.max_size / max(w, h)
              new_w, new_h = int(w * scaling_factor), int(h * scaling_factor)
              img = img.resize((new_w, new_h), Image.ANTIALIAS)
          return img
  ```

- ** Data enhancement **: During the training process, we performed data enhancement operations such as random horizontal flipping and random cropping on the image to improve the generalization ability of the model.

  ```python
  transform = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      ResizeWithAspectRatio(max_size=1024),
      transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
  ```

- ** Normalization **: Normalize the image using the mean and standard deviation of ImageNet to ensure that the distribution of the input data is consistent with the training data of the pre-trained model.

### 4.3 Experimental setup

#### 4.3.1 Ablation study

To verify the impact of each module on model performance, we conducted multiple sets of ablation experiments. There are a total of eight models, including:

1. ** ResNet50 ** (Baseline): Use the original ResNet50 model as a baseline.
2. ** ResNet50 + empty convolution **: Introduce empty convolution in ResNet50 to expand the receptive field.
3. ** ResNet50 + ViT **: Embed the ViT module on the basis of ResNet50 to capture global dependencies.
4. ** ResNet50 + ASPP **: Upgrade empty convolution to ASPP module to improve the diversity of feature extraction.
5. ** ResNet50 + empty convolution + ViT **: Combine empty convolution and ViT to further improve model performance.
6. ** ResNet50 + ASPP + ViT (ASPP before ViT) **: Use the ASPP module first, then embed the ViT.
7. ** ResNet50 + ViT + ASPP (ASPP after ViT) **: Embed ViT first, then use the ASPP module.
8. ** ResNet50 + ASPCrossScanViT **: Based on model 6, multi-directional serialized inputs and a new feature fusion mechanism are introduced.

In these experiments, we focus on the influence of multi-directional serialization input and feature fusion mechanism. This mechanism is reflected in the ResNet50_Dilated_ASPPCrossScanViT class of the model code, and the effective fusion of multi-directional features is achieved by introducing the Cross Scan operation and the FeatureFusion module.

#### 4.3.2 loss function

During training, a ** weighted cross entropy loss function ** ('nn. CrossEntropyLoss') was used. Since there may be class imbalances in our dataset, a weighted approach is better able to handle this situation. The class weights are set to equal, i.e. each class has a weight of 1.

```python
class_weights = torch.tensor([1.0] * num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### 4.3.3 Optimizer and Learning Rate Scheduling

- ** Optimizer **: AdamW optimizer [6] is adopted with an initial learning rate of 2e-4 and a weight decay coefficient of 1e-4. AdamW adds weight decay on top of Adam, which helps prevent overfitting.

  ```python
  optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
  ```

- ** Learning Rate Scheduler **: Use the ** Cosine Annealing Learning Rate Scheduler ** ('CosineAnnealing LR') with the T_max set to 60 and the minimum learning rate set to 1e-6. The scheduler is able to gradually reduce the learning rate during training to avoid falling into local optima.

  ```python
  scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
  ```

#### 4.3.4 training strategy

- ** Distributed training **: Supports multi-GPU distributed training, uses DistributedDataParallel to parallelize the model and speed up the training process.
- ** Early stop mechanism **: Set the number of early stop rounds to 20. If the validation set accuracy does not improve within 20 consecutive epochs, stop training in advance to prevent overfitting.
- ** batch size **: The batch size for training and verification is set to 32.
- ** Logging **: Use the'logging 'module and'TensorBoard' to record the loss, accuracy and other indicators during the training process for later analysis.

#### 4.3.5 Hardware and software environment

- ** Hardware Environment **: Training with 1 or more NVIDIA RTX4090D
- ** Software Environment **: Python 3.8, PyTorch 2.0

### 4.4 experimental results

We trained and evaluated the above eight models, and the results are shown in Table 1.

**Table 1：Experimental results of different models**

| models                                 | ACC（%） | LOSS     |
| ------------------------------------ | ----------- | ---------- |
| ResNet50（Baseline）                 | 97.6587     | 0.0855     |
| ResNet50 + dilated convlution                  | 96.5900     | 0.1189     |
| ResNet50 + ViT                       | 97.8968     | 0.0667     |
| ResNet50 + ASPP                      | 97.8175     | 0.1008     |
| ResNet50 + dilated convlution + ViT            | 98.0952     | 0.0716     |
| ResNet50 + ASPP + ViT(ASPP before ViT) | 98.2540     | 0.0890     |
| ResNet50 + ViT + ASPP(ASPP after ViT) | 98.2937     | 0.0889     |
| ResNet50 + ASPPCrossScanViT           | **98.3730** | **0.0716** |

The experimental results demonstrate that the introduction of dilated convolutions and the ASPP module effectively enhances the model's performance. After embedding the ViT module, the model's ability to capture global dependencies is strengthened, leading to further improvement in accuracy. In particular, the model incorporating multi-directional serialized inputs and the novel feature fusion mechanism (ResNet50 + ASPCrossScanViT) achieves the best performance. This validates the effectiveness of our proposed improvements.

## 5 conclusion

    This paper proposes a multi-scale face recognition model based on improved ResNet50 and ViT. By introducing dilated convolution and the ASPP module, the model can effectively capture multi-scale feature information. The improved ViT module enhances the model's ability to capture global and multi-directional features through the Cross Scan operation and feature fusion mechanism. Multiple sets of ablation experiments verify the contribution of each module to the model's performance. Experimental results show that this model has achieved excellent performance in complex face recognition tasks and has high practical application value.

## 6 References

[1] He X, Cao K, Yan K, et al. Pan-mamba: Effective pan-sharpening with state space model[J]. arXiv preprint arXiv:2402.12192, 2024.

[2] Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *International Conference on Learning Representations*.

[3] Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). Deeplab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(4), 834–848.

[4] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *arXiv preprint arXiv:2010.11929*.

[5] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 7132–7141.

[6] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*.

[7] Zhu L, Liao B, Zhang Q, et al. Vision mamba: Efficient visual representation learning with bidirectional state space model[J]. arXiv preprint arXiv:2401.09417, 2024.

## 7 Supplementary Note
    This work advanced to the national finals of the 2024 Sixth Global Campus Artificial Intelligence Algorithm Elite Competition. However, due to a submission error—only the official test set results were submitted, while the code and weights were not successfully uploaded—the final outcome was less than satisfactory. The code and weights are now open-sourced here as a cautionary reference.

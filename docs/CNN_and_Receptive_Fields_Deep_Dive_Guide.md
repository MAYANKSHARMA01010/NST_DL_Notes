# Deep Learning: Convolutional Neural Networks (CNNs) & Receptive Fields

> **Course**: Deep Learning (Semester 5)  
> **Topic**: Module 11 — Convolutional Neural Networks, Spatial Inductive Biases & Receptive Field Dynamics  
> **Instructor References**:
> - 🌐 **Interactive Lecture Presentation**: [Ashwin Tewary CNN Presentation](https://ashwin-tewary.github.io/dl-worksheets/presentations/cnn/#field) *(or local offline mirror in `03_HTML_Visualizations/Ashwin_Tewary_Presentations/presentations/cnn/`)*
> - 💻 **Class Demo Notebook**: [`Lab_13_CNN_Fashion_MNIST.ipynb`](../04_Notebooks/Lab_13_CNN_Fashion_MNIST/Lab_13_CNN_Fashion_MNIST.ipynb)
> - 📊 **Self-Implementation Dataset**: [Kaggle ImageNet-10k by Priye Rana](https://www.kaggle.com/datasets/priyerana/imagenet-10k)
> - 🐙 **Source Repository**: [ashwin-tewary/dl-worksheets](https://github.com/ashwin-tewary/dl-worksheets)

---

## 📌 Executive Summary: The Visual Revolution

Prior to Convolutional Neural Networks, computer vision relied on Multilayer Perceptrons (MLPs) feeding on flattened 1D pixel vectors. For an image of resolution $224 \times 224 \times 3$, a single hidden layer with 1,000 neurons required over **150 million parameters**, destroying spatial adjacency and quickly causing catastrophic overfitting.

CNNs solve this with two core inductive biases:
1. **Local Receptivity (Spatial Locality)**: Nearby pixels are strongly correlated; distant pixels are less immediately related. Neurons connect only to a local spatial patch rather than the whole image.
2. **Translation Equivariance (Weight Sharing)**: A feature detector (e.g., an edge, corner, or texture detector) that is useful in the top-left corner is equally useful in the bottom-right corner. The exact same small matrix of weights (a **kernel**) slides across the entire image.

---

## 1. Mathematical Anatomy of a Convolution

In deep learning, the operation implemented by `nn.Conv2d` is technically **2D discrete cross-correlation** (convolution without kernel flipping).

Given an input feature map $X \in \mathbb{R}^{H \times W}$ and a kernel $K \in \mathbb{R}^{k_h \times k_w}$ with bias $b \in \mathbb{R}$:

$$Y(i, j) = (X * K)(i, j) = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X(i \cdot S + m, j \cdot S + n) \cdot K(m, n) + b$$

where:
- $S$ is the **Stride** (step size between kernel applications).
- $P$ is the **Padding** (zero-valued border cells added around the input).

---

### The Universal Output Dimension Formula

For an input spatial dimension $W$, kernel size $K$, padding $P$, and stride $S$:

$$\boxed{O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1}$$

#### Two Standard Padding Modes:
1. **Valid Padding ($P = 0$)**:
   - No border is added. The output shrinks by $(K - 1)$ pixels when $S = 1$:
     $$O = W - K + 1$$
2. **Same Padding ($P = \frac{K - 1}{2}$ for odd $K$, with $S = 1$)**:
   - Borders are padded with zeros so that the output spatial dimension strictly matches the input spatial dimension:
     $$O = W$$
   - *Example*: For $W = 28$ and $K = 3$, choose $P = \frac{3 - 1}{2} = 1 \implies O = \frac{28 - 3 + 2(1)}{1} + 1 = 28$.

---

### Multi-Channel Convolutions & Parameter Counting

Real-world images have channels (e.g., RGB has $C_{in} = 3$; grayscale has $C_{in} = 1$). Subsequent feature maps have dozens or hundreds of channels ($C_{in} = 32, 64, \dots$).

1. Each filter is actually a **3D tensor** of shape $(C_{in}, K, K)$.
2. To produce $C_{out}$ distinct output feature maps, we need $C_{out}$ such 3D filters.
3. The complete weight tensor has shape:
   $$\mathbf{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$$
4. **Total Learnable Parameters**:
   $$\boxed{\text{Parameters} = C_{out} \times \left( C_{in} \times K^2 + 1 \right)}$$
   *(where $+1$ accounts for the learnable bias per output channel)*.

> **Crucial Comparison**: Notice that the parameter count is completely **independent of input image resolution ($H, W$)**! A $3 \times 3$ conv layer with 32 filters on a $28 \times 28$ image has the exact same number of weights as on a $4000 \times 4000$ image.

---

## 2. Receptive Field Dynamics (The `#field` Theory)

The instructor specifically emphasizes the **Receptive Field** ([ashwin-tewary.github.io/dl-worksheets/presentations/cnn/#field](https://ashwin-tewary.github.io/dl-worksheets/presentations/cnn/#field)).

### What is the Receptive Field (RF)?
The **Theoretical Receptive Field** of a unit in layer $\ell$ is the area of the original input image that mathematically determines the value of that unit.

```text
Layer 2 (Conv)    [ * ]         <-- RF = 5 pixels of original image
                 /  |  \
Layer 1 (Conv)  [*] [*] [*]     <-- RF = 3 pixels of original image
                /|\ /|\ /|\
Input Image    [. . . . . . .]  <-- Original raw pixels
```

---

### The Exact Recurrence Relation

To compute the receptive field of any unit through deep stacks of convolutions, striding, and pooling, track two variables from the raw pixels ($\ell = 0$):
1. **$\text{RF}_0 = 1$** (a raw pixel sees only itself).
2. **$J_0 = 1$** (the "jump" or spacing between adjacent units on the input pixel grid is 1).

At any subsequent layer $\ell$ with kernel size $K_\ell$ and stride $S_\ell$:

$$\boxed{J_\ell = J_{\ell-1} \cdot S_\ell}$$

$$\boxed{\text{RF}_\ell = \text{RF}_{\ell-1} + (K_\ell - 1) \cdot J_{\ell-1}}$$

---

### Worked Examples from Class

#### Case 1: Stack of Three $3 \times 3$ Convolutions (Stride $S = 1$)
- **Layer 1** ($K_1 = 3, S_1 = 1$):
  - $J_1 = 1 \cdot 1 = 1$
  - $\text{RF}_1 = 1 + (3 - 1) \cdot 1 = \mathbf{3}$
- **Layer 2** ($K_2 = 3, S_2 = 1$):
  - $J_2 = 1 \cdot 1 = 1$
  - $\text{RF}_2 = 3 + (3 - 1) \cdot 1 = \mathbf{5}$
- **Layer 3** ($K_3 = 3, S_3 = 1$):
  - $J_3 = 1 \cdot 1 = 1$
  - $\text{RF}_3 = 5 + (3 - 1) \cdot 1 = \mathbf{7}$

#### Case 2: Two $3 \times 3$ Convolutions with Stride $S = 2$
- **Layer 1** ($K_1 = 3, S_1 = 2$):
  - $J_1 = 1 \cdot 2 = 2$
  - $\text{RF}_1 = 1 + (3 - 1) \cdot 1 = \mathbf{3}$
- **Layer 2** ($K_2 = 3, S_2 = 2$):
  - $J_2 = 2 \cdot 2 = 4$
  - $\text{RF}_2 = 3 + (3 - 1) \cdot 2 = \mathbf{7}$

> **Key Insight**: Striding rapidly accelerates receptive field expansion across the original image!

---

### The VGG Principle: Why Stacking Small $3 \times 3$ Kernels Beats One $7 \times 7$

| Property | Single $7 \times 7$ Layer | Stack of Three $3 \times 3$ Layers | Advantage of $3 \times 3$ Stack |
| :--- | :---: | :---: | :--- |
| **Receptive Field** | $\text{RF} = 7$ | $\text{RF} = 7$ | Identical spatial coverage |
| **Parameters ($C$ channels)** | $49 C^2$ | $3 \times (9 C^2) = \mathbf{27 C^2}$ | **45% fewer parameters!** |
| **Non-Linearities** | 1 (single activation) | **3 (ReLU after each layer)** | Greater expressive capacity |

---

### Theoretical vs. Effective Receptive Field (ERF)
Although the *theoretical* receptive field grows linearly with kernel size, empirical research (Luo et al., NeurIPS 2016) proves that the **Effective Receptive Field (ERF)** has a **2D Gaussian distribution**:
- Pixels at the direct center of the receptive field have thousands of paths propagating forward to the output unit.
- Pixels at the outer periphery have only a few paths through the edges of the kernels.
- Thus, the central region has vastly stronger gradient influence than the edges.

---

## 3. Shape Tracing in PyTorch: Avoiding the #1 Bug

The most common error in deep learning labs is a dimension mismatch during the transition from the convolutional feature extractor to the fully-connected linear classifier:

$$\text{RuntimeError: mat1 and mat2 shapes cannot be multiplied }(B \times 3136 \text{ and } 1024 \times 128)$$

Let's trace our Fashion-MNIST model ([`Lab_13_CNN_Fashion_MNIST.ipynb`](../04_Notebooks/Lab_13_CNN_Fashion_MNIST/Lab_13_CNN_Fashion_MNIST.ipynb)) layer-by-layer:

```text
Input Image: [Batch, 1, 28, 28]
   │
   ▼
nn.Conv2d(1, 32, kernel_size=3, padding='same')
   Formula: O = 28 (due to 'same' padding)
   Output: [Batch, 32, 28, 28]
   │
   ▼
nn.MaxPool2d(kernel_size=2, stride=2)
   Formula: O = 28 / 2 = 14
   Output: [Batch, 32, 14, 14]
   │
   ▼
nn.Conv2d(32, 64, kernel_size=3, padding='same')
   Formula: O = 14 (due to 'same' padding)
   Output: [Batch, 64, 14, 14]
   │
   ▼
nn.MaxPool2d(kernel_size=2, stride=2)
   Formula: O = 14 / 2 = 7
   Output: [Batch, 64, 7, 7]
   │
   ▼
nn.Flatten()
   Formula: 64 * 7 * 7 = 3136
   Output: [Batch, 3136]
   │
   ▼
nn.Linear(3136, 128)
   Output: [Batch, 128]
   │
   ▼
nn.Linear(128, 64)
   Output: [Batch, 64]
   │
   ▼
nn.Linear(64, 10)
   Output: [Batch, 10]
```

---

## 4. PyTorch Reference Implementation

Here is the clean, modular CNN architecture from today's lab:

```python
import torch
import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        # 1. Feature Extractor
        self.features = nn.Sequential(
            # Block 1: (1, 28, 28) -> (32, 28, 28) -> (32, 14, 14)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: (32, 14, 14) -> (64, 14, 14) -> (64, 7, 7)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),  # 3136 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
```

---

## 5. Self-Implementation Assignment: Kaggle ImageNet-10k

As highlighted by instructor Ashwin Tewary in the class announcement, you are encouraged to test your CNN implementation on the **ImageNet-10k** dataset:
- 🔗 **Dataset**: [https://www.kaggle.com/datasets/priyerana/imagenet-10k](https://www.kaggle.com/datasets/priyerana/imagenet-10k)
- **Description**: 10,000 real-world color images across 10 diverse natural categories.

### Key Adaptations Required for ImageNet-10k:
1. **Input Channels**: $C_{in} = 3$ (RGB) instead of 1 (Grayscale).
2. **Resolution**: Images are typically $224 \times 224$ or $64 \times 64$.
3. **Data Augmentation**: Use `torchvision.transforms.Compose`:
   ```python
   from torchvision import transforms
   
   train_transforms = transforms.Compose([
       transforms.Resize((128, 128)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomRotation(degrees=15),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   ```

---

## 6. Milestone CNN Architectures Overview

| Architecture | Year | Key Innovations | Top-5 Error |
| :--- | :---: | :--- | :---: |
| **LeNet-5** | 1998 | First practical CNN for handwritten digit recognition (MNIST). Conv + Subsampling + Tanh. | — |
| **AlexNet** | 2012 | Sparked the Deep Learning revolution. Used ReLU activations, Dropout, GPU training (GTX 580), and Overlapping Pooling. | 15.3% |
| **VGG-16** | 2014 | Homogeneous design using exclusively stacks of small $3 \times 3$ convolutions and $2 \times 2$ max pooling. | 7.3% |
| **GoogLeNet / Inception** | 2014 | Multi-scale processing with parallel branches ($1 \times 1, 3 \times 3, 5 \times 5$, pooling) and $1 \times 1$ convs for bottleneck dimension reduction. | 6.7% |
| **ResNet** | 2015 | Introduced **Residual Skip Connections** ($F(x) + x$) enabling gradient flow through 50, 101, and 152 layers without degradation. | **3.6%** |

# Deep Learning: L1 Loss vs. L2 Loss Deep Dive

> **Course**: Deep Learning (Semester 5)  
> **Topic**: Module 3 & Lab 04 — Loss Functions, Regression Dynamics, and Optimization Landscapes  
> **Associated Notebooks**:
> - 📓 **Classroom Lab**: [`Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb`](../04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb)
> - 📓 **Practice & Landscapes**: [`Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb`](../04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb)
> - 📄 **Related Modules**: Module 03 (Forward Prop, Loss Functions & Autograd), Module 09 (Overfitting & Regularization)

---

## 📌 Executive Summary: The Penalty Philosophy

When training regression models (such as predicting house prices, stock returns, or bounding box coordinates), our core objective is to minimize the discrepancy between the ground truth target $y$ and the network's prediction $\hat{y}$.

The residual error is defined as:
$$e_i = \hat{y}_i - y_i$$

How a loss function transforms this residual error $e_i$ into a scalar penalty dictates:
1. How aggressively the model responds to **outliers**.
2. How gradients behave during **backpropagation**.
3. What statistical measure of central tendency (**mean** vs. **median**) the model ultimately learns.

```
       Residual Error (e = y_pred - y_true)
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
   L1 Loss (MAE)                 L2 Loss (MSE)
 Penalty = |e|                 Penalty = e²
 Linear scaling                Quadratic scaling
 Robust to outliers            Extremely sensitive to outliers
 Learns Conditional Median     Learns Conditional Mean
 Constant gradient magnitude   Gradient scales linearly with error
```

---

## 1. Mathematical Anatomy & Formulations

### 1.1 L1 Loss (Mean Absolute Error / Least Absolute Deviations)

L1 loss computes the mean of the absolute differences between targets and predictions.

* **Single Sample Formulation**:
  $$\mathcal{L}_1(y, \hat{y}) = |y - \hat{y}|$$

* **Dataset / Batch Formulation ($N$ samples)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

* **Vector Norm Perspective**:
  Given residual vector $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^N$:
  $$\text{L1 Error} = \frac{1}{N} \|\mathbf{e}\|_1 = \frac{1}{N} \sum_{i=1}^{N} |e_i|$$

---

### 1.2 L2 Loss (Mean Squared Error / Least Squares Error)

L2 loss computes the mean of the squared differences between targets and predictions. In mathematical derivations, a scaling factor of $\frac{1}{2}$ is often added to neatly cancel the power of $2$ during differentiation.

* **Single Sample Formulation**:
  $$\mathcal{L}_2(y, \hat{y}) = (y - \hat{y})^2 \quad \text{or} \quad \frac{1}{2}(y - \hat{y})^2$$

* **Dataset / Batch Formulation ($N$ samples)**:
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

* **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$
  > **Why RMSE?** MSE penalizes in squared units (e.g., $\text{dollars}^2$), making physical interpretation difficult. Taking the square root brings the error metric back to the original unit of measurement ($\text{dollars}$), while retaining L2's sensitivity to large errors.

* **Vector Norm Perspective**:
  $$\text{L2 Error} = \frac{1}{N} \|\mathbf{e}\|_2^2 = \frac{1}{N} \sum_{i=1}^{N} e_i^2$$

---

## 2. Gradient Derivations & Backpropagation Dynamics

To see how each loss guides model parameter updates, we compute the partial derivative with respect to the network's prediction $\hat{y}$.

```
                 Chain Rule in Backpropagation:
               ∂L/∂w = (∂L/∂y_pred) · (∂y_pred/∂w)
                              ▲
                 This term is dictated by the loss!
```

### 2.1 L2 Loss Gradient (Proportional & Smooth)

Let $\mathcal{L}_2 = \frac{1}{2}(\hat{y} - y)^2$:

$$\frac{\partial \mathcal{L}_2}{\partial \hat{y}} = (\hat{y} - y) = e$$

If using the unscaled version $\mathcal{L}_2 = (\hat{y} - y)^2$:

$$\frac{\partial \mathcal{L}_2}{\partial \hat{y}} = 2(\hat{y} - y) = 2e$$

#### Dynamics & Implications:
1. **Dynamic Step Size**: The gradient is strictly **proportional to the error magnitude**:
   - If error is huge ($e = 100$), gradient is massive ($\pm 200$).
   - If error is small ($e = 0.01$), gradient is tiny ($\pm 0.02$).
2. **Smooth Convergence**: As the prediction approaches the true target ($e \to 0$), the gradient naturally vanishes ($\frac{\partial \mathcal{L}}{\partial \hat{y}} \to 0$). The optimizer automatically decelerates, settling smoothly into the minimum without needing aggressive learning rate decay.
3. **Exploding Gradient Risk**: If an outlier produces an enormous residual, the gradient explodes, potentially destabilizing the entire network weights in a single batch.

---

### 2.2 L1 Loss Gradient (Constant & Non-Smooth)

Let $\mathcal{L}_1 = |\hat{y} - y|$:

$$\frac{\partial \mathcal{L}_1}{\partial \hat{y}} = \text{sign}(\hat{y} - y) = \begin{cases} +1 & \text{if } \hat{y} > y \\ -1 & \text{if } \hat{y} < y \\ \text{undefined} & \text{if } \hat{y} = y \end{cases}$$

*(In computational frameworks like PyTorch, the subgradient at $\hat{y} = y$ is conventionally defined as $0$.)*

#### Dynamics & Implications:
1. **Constant Step Size**: The gradient magnitude is always **$1$**, regardless of whether the error is $1000.0$ or $0.0001$.
2. **The "Bouncing" Problem (Oscillation at Minimum)**:
   - Because the gradient does not shrink near the optimum, a fixed learning rate $\eta$ causes predictions to repeatedly overshoot and oscillate around the minimum value $y$:
     $$\hat{y}_{t+1} = \hat{y}_t - \eta \cdot (+1) \quad \text{then} \quad \hat{y}_{t+2} = \hat{y}_{t+1} - \eta \cdot (-1)$$
   - To converge to high precision with L1 loss, a decaying learning rate schedule (e.g., Cosine Annealing, StepLR) is mandatory.
3. **Non-Differentiable Point**: At the point $\hat{y} = y$, the function forms a sharp $V$-shaped corner and is not formally differentiable in classical calculus, necessitating subgradient methods.

---

## 3. Loss Curves & Gradient Profiles Visualized

```text
       L2 Loss Curve: y = e²               L1 Loss Curve: y = |e|
           Loss                                Loss
            │       •                           │   •           •
            │      • •                          │    •         •
            │     •   •                         │     •       •
            │    •     •                        │      •     •
            │   •       •                       │       •   •
            │ •           •                     │        • •
            └───────────────► Error             └─────────•─────► Error
                   e=0                                 e=0
           (Smooth bowl, flat base)             (Sharp V-corner at e=0)

-------------------------------------------------------------------------

     L2 Gradient Curve: dL/de = 2e         L1 Gradient Curve: dL/de = sign(e)
          dL/de                               dL/de
            │        /                          │       ┌────── +1
            │       /                           │       │
            │      /                            │       │
      ──────┼─────/─────► Error           ──────┼───────┴──────► Error
           /│                                   │       │
          / │                                   │       │
         /  │                                -1 └───────┘
```

---

## 4. Probabilistic Interpretation: Maximum Likelihood Estimation (MLE)

A fundamental theoretical insight in machine learning is that **loss functions directly correspond to negative log-likelihood under specific noise assumptions**.

```
                Supervised Regression Setup:
                     y = f(x; θ) + ε
           where ε represents observation noise
```

### 4.1 L2 Loss $\Longleftrightarrow$ Gaussian (Normal) Noise

Assume the noise $\epsilon$ follows an independent zero-mean Gaussian distribution with variance $\sigma^2$:
$$\epsilon \sim \mathcal{N}(0, \sigma^2) \implies y \sim \mathcal{N}(\hat{y}, \sigma^2)$$

The likelihood of observing data point $y$ given model prediction $\hat{y} = f(x)$ is:
$$p(y \mid x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y - \hat{y})^2}{2\sigma^2} \right)$$

Taking the Negative Log-Likelihood ($-\log p(y \mid x)$):
$$-\log p(y \mid x) = \frac{1}{2\sigma^2}(y - \hat{y})^2 + \frac{1}{2}\log(2\pi\sigma^2)$$

Assuming constant variance $\sigma^2$, minimizing negative log-likelihood is **mathematically identical to minimizing $(y - \hat{y})^2$ (L2 Loss)**.

> **Key Statistical Consequence**:
> Minimizing L2 loss yields the **Conditional Mean** of the target distribution:
> $$\hat{y}^* = \mathbb{E}[Y \mid X]$$
> **Proof**:
> $$\frac{\partial}{\partial c} \mathbb{E}\left[(Y - c)^2\right] = -2\mathbb{E}[Y - c] = -2(\mathbb{E}[Y] - c) = 0 \implies c = \mathbb{E}[Y]$$

---

### 4.2 L1 Loss $\Longleftrightarrow$ Laplace (Double Exponential) Noise

Assume the noise $\epsilon$ follows an independent Laplace distribution with scale parameter $b$:
$$\epsilon \sim \text{Laplace}(0, b) \implies p(\epsilon) = \frac{1}{2b} \exp\left(-\frac{|\epsilon|}{b}\right)$$

The likelihood of observing data point $y$ is:
$$p(y \mid x) = \frac{1}{2b} \exp\left( -\frac{|y - \hat{y}|}{b} \right)$$

Taking the Negative Log-Likelihood:
$$-\log p(y \mid x) = \frac{1}{b}|y - \hat{y}| + \log(2b)$$

Minimizing negative log-likelihood is **mathematically identical to minimizing $|y - \hat{y}|$ (L1 Loss)**.

> **Key Statistical Consequence**:
> Minimizing L1 loss yields the **Conditional Median** of the target distribution:
> $$\hat{y}^* = \text{Median}(Y \mid X)$$
> **Proof**:
> $$\frac{\partial}{\partial c} \mathbb{E}\left[|Y - c|\right] = P(Y < c) - P(Y > c) = 0 \implies P(Y \le c) = 0.5$$

---

## 5. Outlier Sensitivity: The Numerical Walkthrough

Why is L2 so drastically affected by outliers compared to L1? Let us examine concrete numbers.

Suppose a model predicts $\hat{y} = 10.0$ for two data points:
- **Inlier**: True target $y_1 = 12.0$ $\implies$ Error $e_1 = 2.0$
- **Outlier**: True target $y_2 = 100.0$ $\implies$ Error $e_2 = 90.0$

| Metric | Inlier ($e = 2$) | Outlier ($e = 90$) | Outlier / Inlier Penalty Ratio |
| :--- | :--- | :--- | :--- |
| **L1 Penalty ($|e|$)** | $2.0$ | $90.0$ | **$45\times$** |
| **L2 Penalty ($e^2$)** | $4.0$ | $8,100.0$ | **$2,025\times$** |
| **L1 Gradient ($\text{sign}(e)$)** | $+1.0$ | $+1.0$ | **$1\times$ (Identical!)** |
| **L2 Gradient ($2e$)** | $+4.0$ | $+180.0$ | **$45\times$** |

### Why L2 Compromises on Clean Data
Because L2 penalizes large residuals quadratically, an error of $90$ creates a loss of $8,100$. During backpropagation, the optimizer tries desperately to reduce that $8,100$ penalty. 

To reduce the outlier error from $90$ to $80$ (saving $(90^2 - 80^2) = 1,700$ loss units), the model is willing to sacrifice many small inliers (increasing their errors from $1$ to $2$ only costs $(2^2 - 1^2) = 3$ units each). Consequently, **the regression line tilts heavily toward the outlier, destroying accuracy on genuine samples**.

Under L1, moving towards the outlier by $10$ units saves only $10$ loss units, while worsening $10$ inliers by $1$ unit costs $10$ loss units. The optimizer refuses to compromise the majority for the sake of the rogue point.

---

## 6. The Best of Both Worlds: Huber Loss & Smooth L1 Loss

Because L1 is non-smooth at zero and L2 explodes on outliers, practitioners designed hybrid loss functions that transition smoothly from quadratic to linear error penalization.

```text
                  Huber / Smooth L1 Loss Transition
                 
                   Loss
                     │         Linear (|e| - 0.5δ)
                     │        / 
                     │       /  Quadratic (0.5 e²)
                     │      (     for |e| ≤ δ
                     │     / \ 
                     │    /   \
                     └───┴──┴──┴──► Error
                        -δ  0  +δ
```

### 6.1 Mathematical Formulation of Huber Loss

Given a threshold hyperparameter $\delta > 0$:

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta \cdot \left(|y - \hat{y}| - \frac{1}{2}\delta\right) & \text{for } |y - \hat{y}| > \delta \end{cases}$$

### 6.2 Gradient of Huber Loss

$$\frac{\partial L_\delta}{\partial \hat{y}} = \begin{cases} \hat{y} - y & \text{for } |\hat{y} - y| \le \delta \\ \delta \cdot \text{sign}(\hat{y} - y) & \text{for } |\hat{y} - y| > \delta \end{cases}$$

* **Inside the band ($|e| \le \delta$)**: The loss is quadratic $\implies$ gradient smoothly scales to $0$, avoiding oscillations around the minimum.
* **Outside the band ($|e| > \delta$)**: The loss is linear $\implies$ gradient is capped at $\pm \delta$, preventing exploding gradients from outliers.

> [!TIP]
> **Industry Application: Object Detection Bounding Boxes**  
> In modern object detectors (e.g., Fast R-CNN, Faster R-CNN, SSD, YOLO), bounding box regression uses **Smooth L1 Loss** (`nn.SmoothL1Loss(beta=1.0)`). In the initial training stages, box offsets can be huge; linear penalization prevents gradient explosions. In the later stages, boxes are refined with sub-pixel precision; quadratic behavior ensures stable convergence.

---

## 7. ⚠️ Common Confusion: Loss Function vs. Regularization

One of the most frequent misconceptions in deep learning interviews and examinations is conflating **L1/L2 Loss** with **L1/L2 Regularization**.

| Attribute | **L1 / L2 Loss Functions** | **L1 / L2 Regularization (Weight Decay / Penalties)** |
| :--- | :--- | :--- |
| **What does it measure?** | Discrepancy between prediction $\hat{y}$ and target $y$. | Complexity/magnitude of the network weights $\mathbf{W}$. |
| **Where does it act?** | At the output layer: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$. | Inside the model parameters: $\mathcal{R}(\mathbf{W})$. |
| **Mathematical Term** | $\mathcal{L}_{\text{task}} = \frac{1}{N}\sum \|y_i - \hat{y}_i\|$ or $(y_i - \hat{y}_i)^2$ | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \sum |w_j|$ or $\frac{\lambda}{2}\sum w_j^2$ |
| **Primary Goal** | Minimize prediction error on the task. | Prevent overfitting, penalize extreme parameter values. |
| **Sparsity Effect** | L1 loss produces **robust regression** (median). | L1 regularization (Lasso) produces **sparse weights** ($w_j \to 0$). |
| **Weight Decay Effect** | L2 loss penalizes large prediction errors. | L2 regularization (Ridge) shrinks weights smoothly toward 0. |

> [!IMPORTANT]
> - **L1 Loss** makes the model **outlier-robust**. It does **not** make weights sparse.
> - **L1 Regularization** makes model weights **sparse**. It does not change the prediction loss geometry.

---

## 8. Comprehensive Comparison Matrix

| Property | L1 Loss (MAE) | L2 Loss (MSE) | Huber / Smooth L1 Loss |
| :--- | :--- | :--- | :--- |
| **Full Name** | Mean Absolute Error / LAD | Mean Squared Error / LSE | Huber Loss / Smooth L1 |
| **Formula** | $\frac{1}{N}\sum \|y_i - \hat{y}_i\|$ | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | Piecewise (quadratic near 0, linear afar) |
| **Gradient Magnitude** | Constant: $\pm 1$ | Proportional: $2\|e\|$ | Linear for $\|e\|\le\delta$, constant $\delta$ for $\|e\|>\delta$ |
| **Differentiability** | Everywhere except at $e = 0$ | Continuously differentiable everywhere | Continuously differentiable everywhere |
| **Sensitivity to Outliers** | **Low (Robust)** | **Extreme (Vulnerable)** | **Low (Robust)** |
| **Sensitivity to Small Errors** | Linear ($0.1 \to 0.1$) | Very weak ($0.1 \to 0.01$) | Controlled by $\delta$ |
| **Optimization Near 0** | Oscillates around minimum | Naturally slows and converges | Naturally slows and converges |
| **Convexity** | Convex (flat regions possible) | **Strictly convex** | Convex |
| **Uniqueness of Solution** | May have multiple solutions | **Unique global minimum** | Unique global minimum |
| **Target Statistic (MLE)** | **Conditional Median** | **Conditional Mean** | Robust hybrid |
| **Noise Assumption** | Laplace distribution | Gaussian (Normal) distribution | Heavy-tailed / contaminated Gaussian |
| **PyTorch Class** | `torch.nn.L1Loss()` | `torch.nn.MSELoss()` | `torch.nn.HuberLoss(delta=1.0)` |

---

## 9. PyTorch & NumPy Implementations

### 9.1 From Scratch: Forward and Backward Passes in NumPy

```python
import numpy as np

# 1. L1 Loss (MAE) Forward and Backward
def l1_loss_forward(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def l1_loss_backward(y_true, y_pred):
    """
    Subgradient of L1 loss with respect to y_pred.
    Returns +1/N if y_pred > y_true, -1/N if y_pred < y_true, 0 if equal.
    """
    N = len(y_true)
    grad = np.sign(y_pred - y_true) / N
    return grad

# 2. L2 Loss (MSE) Forward and Backward
def l2_loss_forward(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def l2_loss_backward(y_true, y_pred):
    """
    Derivative of L2 loss with respect to y_pred: 2 * (y_pred - y_true) / N
    """
    N = len(y_true)
    grad = 2.0 * (y_pred - y_true) / N
    return grad

# Verification on toy sample
y_true = np.array([10.0, 15.0, 20.0])
y_pred = np.array([12.0, 15.0, 30.0])  # errors: +2, 0, +10

print("L1 Loss:", l1_loss_forward(y_true, y_pred))     # (|2| + |0| + |10|) / 3 = 4.0
print("L2 Loss:", l2_loss_forward(y_true, y_pred))     # (4 + 0 + 100) / 3 = 34.6667
print("L1 Grad:", l1_loss_backward(y_true, y_pred))    # [+0.333, 0.0, +0.333]
print("L2 Grad:", l2_loss_backward(y_true, y_pred))    # [2*(2)/3, 2*(0)/3, 2*(10)/3] = [1.333, 0.0, 6.667]
```

---

### 9.2 PyTorch Standard Module Comparison

```python
import torch
import torch.nn as nn

# Target and predictions requiring autograd
y_true = torch.tensor([[3.0], [5.0], [100.0]], dtype=torch.float32)  # 100.0 is an outlier
y_pred = torch.tensor([[2.5], [5.5], [10.0]], dtype=torch.float32, requires_grad=True)

# Define PyTorch built-in loss functions
mae_criterion = nn.L1Loss()
mse_criterion = nn.MSELoss()
huber_criterion = nn.HuberLoss(delta=1.0)

# Compute Losses
loss_l1 = mae_criterion(y_pred, y_true)
loss_l2 = mse_criterion(y_pred, y_true)
loss_huber = huber_criterion(y_pred, y_true)

print(f"L1 Loss (MAE):    {loss_l1.item():.4f}")
print(f"L2 Loss (MSE):    {loss_l2.item():.4f}")
print(f"Huber Loss (δ=1): {loss_huber.item():.4f}")

# Backpropagation comparison
loss_l1.backward(retain_graph=True)
print("L1 Gradients on y_pred:\n", y_pred.grad)

y_pred.grad.zero_()
loss_l2.backward(retain_graph=True)
print("L2 Gradients on y_pred:\n", y_pred.grad)

y_pred.grad.zero_()
loss_huber.backward()
print("Huber Gradients on y_pred:\n", y_pred.grad)
```

**Expected Gradient Output**:
- Under **L1**, the outlier gradient is strictly bounded: $\frac{-1}{3} = -0.3333$.
- Under **L2**, the outlier gradient explodes: $\frac{2 \times (10 - 100)}{3} = -60.0$.
- Under **Huber**, the outlier gradient is safely capped at $\frac{-\delta}{3} = -0.3333$.

---

## 10. Practical Decision Guide: Which One Should You Use?

```
                               Start Here
                                    │
                  Are there severe anomalies / outliers
                  in the target measurements?
                                    │
                    ┌───────────────┴───────────────┐
                   YES                              NO
                    │                               │
       Does your model need to         Are target errors normally
       refine predictions with         distributed and large errors
       sub-pixel / smooth precision?   prohibitively dangerous?
                    │                               │
            ┌───────┴───────┐                       │
           YES              NO                      ▼
            │               │                  USE L2 LOSS
            ▼               ▼                    (MSE)
     USE HUBER LOSS     USE L1 LOSS         • Classical regression
    (Smooth L1 Loss)      (MAE)             • Variance prediction
    • Bounding boxes   • Financial data     • Temperature / sensors
    • Object detection • Median estimates
    • RL Bellman error • Robust baseline
```

### Choose L1 Loss (MAE) when:
1. The dataset contains **significant sensor noise, corruption, or extreme anomalies** that cannot easily be cleaned.
2. The business objective cares about the **typical/median** customer rather than the extreme average (e.g., house price pricing models, typical wait times).
3. Penalizing twice as much for a double error is adequate, rather than four times as much.

### Choose L2 Loss (MSE) when:
1. Target errors are cleanly **normally distributed**.
2. **Large errors are catastrophic** and must be penalized with exponential severity (e.g., safety-critical autopilots, structural load predictions).
3. You need a smooth, strictly convex loss landscape that guarantees a unique global minimum for linear architectures and smooth gradient flow to zero.

### Choose Huber / Smooth L1 Loss when:
1. You are predicting continuous coordinates (e.g., **Object Detection bounding boxes**, keypoint localization).
2. You want outlier robustness during early epochs, but quadratic convergence smoothness near the optimum.

---

## 11. High-Yield Interview & Exam Questions

### Q1: Why does L1 loss yield the median while L2 loss yields the mean?
**Answer**:  
Minimizing the expected error $\mathbb{E}[L(Y, c)]$ with respect to constant prediction $c$:
- For L2: $\frac{d}{dc}\mathbb{E}[(Y - c)^2] = -2\mathbb{E}[Y - c] = 0 \implies c = \mathbb{E}[Y]$ (the expected value / mean).
- For L1: $\frac{d}{dc}\mathbb{E}[|Y - c|] = P(Y < c) - P(Y > c) = 0 \implies P(Y \le c) = 0.5$ (the 50th percentile / median).

### Q2: Why can't we easily use Gradient Descent with a constant learning rate on L1 loss?
**Answer**:  
Because the derivative of L1 loss is $\pm 1$ everywhere except $0$. The gradient magnitude does not diminish as the error approaches zero. A fixed learning rate $\eta$ causes the weight update to maintain a constant step size $\eta \cdot 1$, causing the model to endlessly jump over the minimum and oscillate. A learning rate decay schedule or subgradient method is required.

### Q3: What is the breakdown point of L2 loss compared to L1 loss?
**Answer**:  
In robust statistics, the breakdown point is the proportion of contaminated data a model can handle before producing arbitrarily bad outputs. L2 loss has a breakdown point of **$0\%$** (a single point at $\infty$ pulls the mean to $\infty$). L1 loss has a much higher breakdown point (up to **$50\%$** for a 1D median estimate).

### Q4: How does L1 loss differ from L1 regularization?
**Answer**:  
L1 loss is a **task error function** measuring $|y - \hat{y}|$ to train regression outputs robust to label outliers. L1 regularization (Lasso) is a **model weight penalty** adding $\lambda \sum |w_i|$ to the loss to penalize parameter magnitude, driving non-informative weights strictly to zero to induce structural sparsity.

---

## 📚 References & Further Reading

1. **Course Repository Notebooks**:
   - [`Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb`](../04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb) — Practical implementation on California Housing dataset.
   - [`Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb`](../04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb) — 2D and 3D loss surface visualizations.
2. **Foundational Literature**:
   - Goodfellow, Bengio, Courville. *Deep Learning* (MIT Press), Chapter 6: Deep Feedforward Networks (Maximum Likelihood & Cost Functions).
   - Girshick, Ross. *"Fast R-CNN"* (ICCV 2015) — Introduces Smooth L1 loss for bounding box regression.
   - Huber, Peter J. *"Robust Estimation of a Location Parameter"* (Annals of Mathematical Statistics, 1964).

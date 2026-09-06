# Deep Learning Course Repository

This repository contains all organized lecture notes, worksheets, interactive HTML visualizations, and Jupyter/Google Colab lab notebooks for the **Deep Learning Course (Semester 5)**.

All materials have been categorized into dedicated folders and cross-referenced module-by-module so you can easily find everything without scrolling through Google Space or Drive.

---

## Directory Structure

```text
DL/
├── 01_Notes/                  # Lecture slides and comprehensive notes
├── 02_Worksheets/             # Worksheets 01 to 07 (PDFs & interactive HTML)
│   └── Solved/                # Solved worksheet answer keys
├── 03_HTML_Visualizations/    # Web-based explorables, interactive tools, & visual analogies
├── 04_Notebooks/              # Hands-on Jupyter & Google Colab lab notebooks (Lab 00 to 06)
│   ├── Lab_00_PyTorch_Basics/
│   ├── Lab_01_Perceptrons_and_XOR/
│   ├── Lab_02_Activation_Functions/
│   ├── Lab_03_Forward_Prop_and_Autograd/
│   ├── Lab_04_Loss_Functions/
│   ├── Lab_05_Gradient_Descent_and_Training/
│   └── Lab_06_Optimizers/
└── _archive_original_structure/  # Full backup archive of original unsorted files
```

---

## Course Module Map

### Module 1: Perceptrons, Linear Separability & Shallow Networks
*Understanding biological neuron analogies, the Perceptron learning rule, linear decision boundaries, and why hidden layers are needed to solve XOR.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Notes** | [Lecture_01_Perceptron_to_Shallow_Networks.pdf](01_Notes/Lecture_01_Perceptron_to_Shallow_Networks.pdf) | Slide deck covering neurons, weights, biases, and thresholding |
| **Worksheet** | [Worksheet_01_From_Perceptron_to_Shallow_Networks.pdf](02_Worksheets/Worksheet_01_From_Perceptron_to_Shallow_Networks.pdf) | Worksheet 01 on perceptrons and labelled shallow networks |
| **Lab Notebook** | [Lab_00_PyTorch_Coding_Demonstration.ipynb](04_Notebooks/Lab_00_PyTorch_Basics/Lab_00_PyTorch_Coding_Demonstration.ipynb) | Lab 0: PyTorch tensor fundamentals demonstration |
| **Lab Notebook** | [Lab_01_Part_1_XOR_Problem_Why_We_Need_Hidden_Layers.ipynb](04_Notebooks/Lab_01_Perceptrons_and_XOR/Lab_01_Part_1_XOR_Problem_Why_We_Need_Hidden_Layers.ipynb) | Demonstrating linear inseparability of XOR |
| **Lab Notebook** | [Lab_01_Part_2_Shallow_Neural_Networks_1D.ipynb](04_Notebooks/Lab_01_Perceptrons_and_XOR/Lab_01_Part_2_Shallow_Neural_Networks_1D.ipynb) | Shallow neural networks on 1D continuous functions |

---

### Module 2: Activation Functions & Update Rules
*Deep dive into non-linear activations: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and Softmax, alongside weight update math.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_02_Activation_Function_and_Update_Rule.pdf](02_Worksheets/Worksheet_02_Activation_Function_and_Update_Rule.pdf) | Worksheet 02: Mathematical derivations of activation gradients |
| **Worksheet** | [Worksheet_03_ReLU_Leaky_ReLU_ELU_and_Softmax.pdf](02_Worksheets/Worksheet_03_ReLU_Leaky_ReLU_ELU_and_Softmax.pdf) | Worksheet 03: Modern activations and Dying ReLU problem |
| **Visualization** | [Module_02_Activation_Functions_Explorer.html](03_HTML_Visualizations/Module_02_Activation_Functions_Explorer.html) | Interactive grapher and derivative plotter for activations |
| **Visualization** | [Module_02_Perceptron_and_Network_Activations_Playground.html](03_HTML_Visualizations/Module_02_Perceptron_and_Network_Activations_Playground.html) | Interactive sandbox: decision boundary, loss bowl, presets |
| **Lab Notebook** | [Lab_02_Part_1_Activation_Functions_Live_Demo.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_1_Activation_Functions_Live_Demo.ipynb) | Live coding: 1-3-1 toy network comparing activation effects |
| **Lab Notebook** | [Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab.ipynb) | 1-hour structured lab on activation functions |
| **Lab Notebook** | [Lab_02_Part_3_NN_Forward_Prop_and_Loss_Breast_Cancer.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_3_NN_Forward_Prop_and_Loss_Breast_Cancer.ipynb) | Forward pass and binary cross-entropy on Breast Cancer dataset |
| **Lab Notebook** | [Lab_03_Part_1_Activation_Functions_Simple_Demo.ipynb](04_Notebooks/Lab_03_Forward_Prop_and_Autograd/Lab_03_Part_1_Activation_Functions_Simple_Demo.ipynb) | Student version: step-by-step activation intuition |

---

### Module 3: Forward Propagation, Loss Functions & Autograd
*Building networks from scratch in PyTorch, implementing loss functions (MSE, MAE, BCE, CCE), and computing gradients via computational graphs.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Lab Notebook** | [Lab_03_Part_2_NN_Forward_Prop_and_Loss_Bank_Churn.ipynb](04_Notebooks/Lab_03_Forward_Prop_and_Autograd/Lab_03_Part_2_NN_Forward_Prop_and_Loss_Bank_Churn.ipynb) | Forward prop and loss evaluation on Bank Customer Churn data |
| **Lab Notebook** | [Lab_03_Part_3_PyTorch_Autograd_From_First_Principles.ipynb](04_Notebooks/Lab_03_Forward_Prop_and_Autograd/Lab_03_Part_3_PyTorch_Autograd_From_First_Principles.ipynb) | Reverse-mode automatic differentiation from first principles |
| **Lab Notebook** | [Lab_04_Loss_Functions_MSE_MAE_BCE_CCE.ipynb](04_Notebooks/Lab_04_Loss_Functions/Lab_04_Loss_Functions_MSE_MAE_BCE_CCE.ipynb) | Comprehensive 80-minute lab on standard loss functions |

---

### Module 4: Multilayer Perceptrons & Backpropagation
*Chain rule through composite functions, backpropagation error propagation from output layer to hidden units, and parameter counting.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_04_MLP_XOR_to_Backpropagation.pdf](02_Worksheets/Worksheet_04_MLP_XOR_to_Backpropagation.pdf) | Worksheet 04: Hand derivations of gradients and parameter counts |
| **Worksheet / Web** | [Worksheet_04_Backpropagation_Output_Layer_Interactive.html](02_Worksheets/Worksheet_04_Backpropagation_Output_Layer_Interactive.html) | Interactive output layer backprop step-by-step calculator |
| **Visualization** | [Module_04_Backpropagation_Interactive_Worksheet.html](03_HTML_Visualizations/Module_04_Backpropagation_Interactive_Worksheet.html) | Standalone interactive backprop explorer |

---

### Module 5: Gradient Descent & Deep Learning Training Pipeline
*Comparing Batch GD, Stochastic GD (SGD), and Mini-Batch GD, learning rate schedules, and complete modular PyTorch training loops.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_05_Gradient_Descent_and_Optimization.pdf](02_Worksheets/Worksheet_05_Gradient_Descent_and_Optimization.pdf) | Worksheet 05: Hand calculations for GD step updates (PDF) |
| **Worksheet / Web** | [Worksheet_05_Gradient_Descent_and_Optimization_Interactive.html](02_Worksheets/Worksheet_05_Gradient_Descent_and_Optimization_Interactive.html) | Worksheet 05: Interactive explorable notes & step-by-step calculations (HTML) |
| **Visualization** | [Module_05_Epoch_Circuit_Visual_Analogy.html](03_HTML_Visualizations/Module_05_Epoch_Circuit_Visual_Analogy.html) | Visual race track comparing Batch, Mini-batch, and SGD speeds |
| **Lab Notebook** | [Lab_05_Part_1_PyTorch_Autograd_Computational_Graph.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_1_PyTorch_Autograd_Computational_Graph.ipynb) | PyTorch Autograd computational graph inspection |
| **Lab Notebook** | [Lab_05_Part_1_Quiz_Autograd_in_AI_Town.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_1_Quiz_Autograd_in_AI_Town.ipynb) | Practical Autograd quiz and problem scenarios |
| **Lab Notebook** | [Lab_05_Part_2_Gradient_Descent_Types_Batch_vs_SGD.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_2_Gradient_Descent_Types_Batch_vs_SGD.ipynb) | Empirical comparison of Batch vs SGD with execution plots |
| **Lab Notebook** | [Lab_05_Part_2_PyTorch_Training_Pipeline.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_2_PyTorch_Training_Pipeline.ipynb) | End-to-end training pipeline with dataset loaders & evaluation |
| **Lab Notebook** | [Lab_05_Part_2_PyTorch_Dataset_DataLoader_Pipeline.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_2_PyTorch_Dataset_DataLoader_Pipeline.ipynb) | Deep dive into PyTorch Dataset and DataLoader machinery |
| **Lab Notebook** | [Lab_05_Part_3_In_Class_Diabetes_Prediction_Boilerplate.ipynb](04_Notebooks/Lab_05_Gradient_Descent_and_Training/Lab_05_Part_3_In_Class_Diabetes_Prediction_Boilerplate.ipynb) | In-class exercise boilerplate on Pima Indians Diabetes dataset |
| **Diagram** | [network_diagram.png](04_Notebooks/Lab_05_Gradient_Descent_and_Training/network_diagram.png) | Architecture diagram for Lab 5 neural network |

---

### Module 6: Optimizers with Memory (Momentum, NAG, RMSProp, Adam)
*Why raw SGD gets stuck in ravines, exponentially weighted moving averages (EWMA), Momentum, Nesterov Accelerated Gradient (NAG), and the Optimizer Zoo.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_06_Optimizers_With_Memory_Student.pdf](02_Worksheets/Worksheet_06_Optimizers_With_Memory_Student.pdf) | Worksheet 06: Student practice worksheet (unsolved) |
| **Worksheet (Solved)** | [Worksheet_06_Optimizers_With_Memory_Solved.pdf](02_Worksheets/Solved/Worksheet_06_Optimizers_With_Memory_Solved.pdf) | Worksheet 06: Full step-by-step solved key derivations |
| **Worksheet / Web** | [Worksheet_07_Momentum_EWMA_and_NAG_Interactive.html](02_Worksheets/Worksheet_07_Momentum_EWMA_and_NAG_Interactive.html) | Interactive Worksheet 07: SGD &rarr; EWMA &rarr; Momentum &rarr; NAG |
| **Visualization** | [Module_06_Gradient_Descent_Interactive_Lecture.html](03_HTML_Visualizations/Module_06_Gradient_Descent_Interactive_Lecture.html) | Explorable gradient descent trajectory visualizer |
| **Visualization** | [Module_06_Momentum_and_NAG_Interactive_Worksheet.html](03_HTML_Visualizations/Module_06_Momentum_and_NAG_Interactive_Worksheet.html) | Interactive momentum and NAG comparison tool |
| **Lab Notebook** | [Lab_06_Part_A_Optimizers_Zoo_Ravine_Trajectories.ipynb](04_Notebooks/Lab_06_Optimizers/Lab_06_Part_A_Optimizers_Zoo_Ravine_Trajectories.ipynb) | Lab 6 Part A: 2D quadratic ravine loss optimizer trajectories |
| **Lab Notebook** | [Lab_06_The_Optimizer_Zoo_SGD_to_Adam.ipynb](04_Notebooks/Lab_06_Optimizers/Lab_06_The_Optimizer_Zoo_SGD_to_Adam.ipynb) | Lab 6: 80-minute PyTorch lab navigating SGD, Momentum, and Adam |

---

## How to Use These Materials

1. **Viewing HTML Visualizations**:
   Double click any `.html` file inside `03_HTML_Visualizations/` to open it in Chrome, Safari, or your default web browser. All visualizers are self-contained with no server required.
2. **Running Notebooks**:
   Open notebooks locally using:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
   Alternatively, upload the `.ipynb` file to [Google Colab](https://colab.research.google.com).
3. **Restoring Original Structure**:
   If you ever need the original unsorted structure, the full copy is preserved in `_archive_original_structure/`.

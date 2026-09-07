# Deep Learning Course Repository

This repository contains all organized lecture notes, worksheets, interactive HTML visualizations, and Jupyter/Google Colab lab notebooks for the **Deep Learning Course (Semester 5)**.

All materials have been categorized into dedicated folders and cross-referenced module-by-module so you can easily find everything without scrolling through Google Space or Drive.

### 🔗 Official Google Drive Folders
* 📂 **[DL Notes Google Drive](https://drive.google.com/drive/folders/1dFwGfgOvE9a-1veKjbuvSMYV3qb5Swe5)** — Lecture slides, reading notes, and worksheets.
* 📓 **[DL Lab Notebooks Google Drive](https://drive.google.com/drive/folders/1cb0M-aseFx7djB3bF0OrtyB14QLSfaaV)** — Official lab notebooks and reference solutions.

---

### Directory Structure

```text
DL/
├── 01_Notes/                  # Lecture slides and comprehensive notes
├── 02_Worksheets/             # Worksheets 01 to 08 (PDFs & interactive HTML)
│   └── Solved/                # Solved worksheet answer keys
├── 03_HTML_Visualizations/    # Web-based explorables, interactive tools, & visual analogies
├── 04_Notebooks/              # Hands-on Jupyter & Google Colab lab notebooks (Lab 00 to 08)
│   ├── Lab_00_PyTorch_Basics/
│   ├── Lab_01_Perceptrons_and_XOR/
│   ├── Lab_02_Activation_Functions/
│   ├── Lab_03_ReLU_ELU_Softmax_and_Autograd/
│   ├── Lab_04_Loss_Functions/
│   ├── Lab_05_nn_Module_and_Pipelines/
│   ├── Lab_06_Gradient_Descent_and_DataLoaders/
│   ├── Lab_07_The_Optimizer_Zoo/
│   ├── Lab_08_Face_Recognition_MLP_Pipeline/
│   └── Pratice/
└── Deep Learning Lab Notebooks/ # Clean curriculum mirror (LAB_0 through LAB_8)
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
| **Lab Notebook (Solved)** | [Lab_00_PyTorch_Coding_Demonstration_Solved.ipynb](04_Notebooks/Lab_00_PyTorch_Basics/Lab_00_PyTorch_Coding_Demonstration_Solved.ipynb) | Lab 0: Fully solved and executed key |
| **Lab Notebook** | [Lab_00_PyTorch_Classroom_Demonstration.ipynb](04_Notebooks/Lab_00_PyTorch_Basics/Lab_00_PyTorch_Classroom_Demonstration.ipynb) | Lab 0: Classroom PyTorch tensor creation and GPU acceleration |
| **Lab Notebook** | [Lab_01_Part_1_XOR_Problem_Why_We_Need_Hidden_Layers.ipynb](04_Notebooks/Lab_01_Perceptrons_and_XOR/Lab_01_Part_1_XOR_Problem_Why_We_Need_Hidden_Layers.ipynb) | Demonstrating linear inseparability of XOR |
| **Lab Notebook** | [Lab_01_Part_2_Shallow_Neural_Networks_1D.ipynb](04_Notebooks/Lab_01_Perceptrons_and_XOR/Lab_01_Part_2_Shallow_Neural_Networks_1D.ipynb) | Shallow neural networks on 1D continuous functions |
| **Lab Notebook** | [Lab_01_Part_3_XOR_Problem_Perceptron_to_MLP.ipynb](04_Notebooks/Lab_01_Perceptrons_and_XOR/Lab_01_Part_3_XOR_Problem_Perceptron_to_MLP.ipynb) | Lab 1: AND/OR gates with single perceptron and XOR with MLP |

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
| **Lab Notebook** | [Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab.ipynb) | 1-hour structured lab on activation functions (Student) |
| **Lab Notebook (Solved)** | [Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab_Solved.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_2_Activation_Functions_PyTorch_1hr_Lab_Solved.ipynb) | 1-hour lab with instructor solutions and exit ticket key |
| **Lab Notebook** | [Lab_02_Part_3_NN_Forward_Prop_and_Loss_Breast_Cancer.ipynb](04_Notebooks/Lab_02_Activation_Functions/Lab_02_Part_3_NN_Forward_Prop_and_Loss_Breast_Cancer.ipynb) | Forward pass and binary cross-entropy on Breast Cancer dataset |

---

### Module 3: Forward Propagation, Loss Functions & Autograd
*Building networks from scratch in PyTorch, implementing loss functions (MSE, MAE, BCE, CCE), and computing gradients via computational graphs.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Lab Notebook** | [Lab_03_Part_1_Activation_Functions_Simple_Demo.ipynb](04_Notebooks/Lab_03_ReLU_ELU_Softmax_and_Autograd/Lab_03_Part_1_Activation_Functions_Simple_Demo.ipynb) | Student version: step-by-step activation intuition |
| **Lab Notebook** | [Lab_03_Part_2_NN_Forward_Prop_and_Loss_Bank_Churn.ipynb](04_Notebooks/Lab_03_ReLU_ELU_Softmax_and_Autograd/Lab_03_Part_2_NN_Forward_Prop_and_Loss_Bank_Churn.ipynb) | Forward prop and loss evaluation on Bank Customer Churn data |
| **Lab Notebook** | [Lab_03_Part_3_PyTorch_Autograd_From_First_Principles.ipynb](04_Notebooks/Lab_03_ReLU_ELU_Softmax_and_Autograd/Lab_03_Part_3_PyTorch_Autograd_From_First_Principles.ipynb) | Reverse-mode automatic differentiation from first principles |
| **Lab Notebook** | [Lab_03_Part_4_ReLU_ELU_Softmax_and_Dying_Neuron.ipynb](04_Notebooks/Lab_03_ReLU_ELU_Softmax_and_Autograd/Lab_03_Part_4_ReLU_ELU_Softmax_and_Dying_Neuron.ipynb) | Lab 3: ReLU, Leaky ReLU, ELU, Dying ReLU detection, and Softmax |
| **Lab Notebook** | [Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb](04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_1_Loss_Functions_MSE_MAE_BCE_CCE.ipynb) | Comprehensive 80-minute lab on standard loss functions |
| **Lab Notebook** | [Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb](04_Notebooks/Lab_04_Loss_Functions/Lab_04_Part_2_Loss_Functions_Practice_and_Landscapes.ipynb) | Lab 4: Loss landscapes, California Housing (MSE/MAE), BCE with logits, and CCE |

---

### Module 4: Multilayer Perceptrons & Backpropagation
*Chain rule through composite functions, backpropagation error propagation from output layer to hidden units, and parameter counting.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_04_MLP_XOR_to_Backpropagation.pdf](02_Worksheets/Worksheet_04_MLP_XOR_to_Backpropagation.pdf) | Worksheet 04: Hand derivations of gradients and parameter counts |
| **Worksheet / Web** | [Worksheet_04_Backpropagation_Output_Layer_Interactive.html](02_Worksheets/Worksheet_04_Backpropagation_Output_Layer_Interactive.html) | Interactive output layer backprop step-by-step calculator |
| **Visualization** | [Module_04_Backpropagation_Interactive_Worksheet.html](03_HTML_Visualizations/Module_04_Backpropagation_Interactive_Worksheet.html) | Standalone interactive backprop explorer |
| **Lab Notebook (Boilerplate)** | [Lab_08_Face_Recognition_MLP_Boilerplate.ipynb](04_Notebooks/Lab_08_Face_Recognition_MLP_Pipeline/Lab_08_Face_Recognition_MLP_Boilerplate.ipynb) | Lab 8: End-to-End MLP Pipeline on Olivetti Faces (Student Boilerplate) |
| **Lab Notebook (Solved)** | [Lab_08_Face_Recognition_MLP_Solved.ipynb](04_Notebooks/Lab_08_Face_Recognition_MLP_Pipeline/Lab_08_Face_Recognition_MLP_Solved.ipynb) | Lab 8: End-to-End MLP Pipeline on Olivetti Faces (Complete Solved Key) |

---

### Module 5: Gradient Descent & Deep Learning Training Pipeline
*Comparing Batch GD, Stochastic GD (SGD), and Mini-Batch GD, learning rate schedules, and complete modular PyTorch training loops.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_05_Gradient_Descent_and_Optimization.pdf](02_Worksheets/Worksheet_05_Gradient_Descent_and_Optimization.pdf) | Worksheet 05: Hand calculations for GD step updates (PDF) |
| **Worksheet / Web** | [Worksheet_05_Gradient_Descent_and_Optimization_Interactive.html](02_Worksheets/Worksheet_05_Gradient_Descent_and_Optimization_Interactive.html) | Worksheet 05: Interactive explorable notes & step-by-step calculations (HTML) |
| **Visualization** | [Module_05_Epoch_Circuit_Visual_Analogy.html](03_HTML_Visualizations/Module_05_Epoch_Circuit_Visual_Analogy.html) | Visual race track comparing Batch, Mini-batch, and SGD speeds |
| **Lab Notebook** | [Lab_05_Part_1_PyTorch_Autograd_Computational_Graph.ipynb](04_Notebooks/Lab_05_nn_Module_and_Pipelines/Lab_05_Part_1_PyTorch_Autograd_Computational_Graph.ipynb) | PyTorch Autograd computational graph inspection |
| **Lab Notebook** | [Lab_05_Part_1_Quiz_Autograd_in_AI_Town.ipynb](04_Notebooks/Lab_05_nn_Module_and_Pipelines/Lab_05_Part_1_Quiz_Autograd_in_AI_Town.ipynb) | Practical Autograd quiz and problem scenarios |
| **Lab Notebook** | [Lab_05_Part_2_nn_Module_Breast_Cancer_Pipeline.ipynb](04_Notebooks/Lab_05_nn_Module_and_Pipelines/Lab_05_Part_2_nn_Module_Breast_Cancer_Pipeline.ipynb) | Lab 5: Building a binary classifier with PyTorch `nn.Module` on Breast Cancer data |
| **Lab Notebook** | [Lab_05_Part_3_In_Class_Diabetes_Prediction_Boilerplate.ipynb](04_Notebooks/Lab_05_nn_Module_and_Pipelines/Lab_05_Part_3_In_Class_Diabetes_Prediction_Boilerplate.ipynb) | In-class exercise boilerplate on Pima Indians Diabetes dataset |
| **Lab Notebook (Solved)** | [Lab_05_Part_3_In_Class_Diabetes_Prediction_Solved.ipynb](04_Notebooks/Lab_05_nn_Module_and_Pipelines/Lab_05_Part_3_In_Class_Diabetes_Prediction_Solved.ipynb) | In-class exercise solved key with executed loss curves |
| **Diagram** | [network_diagram.png](04_Notebooks/Lab_05_nn_Module_and_Pipelines/network_diagram.png) | Architecture diagram for Lab 5 neural network |
| **Lab Notebook** | [Lab_06_Part_1_Gradient_Descent_Batch_vs_SGD.ipynb](04_Notebooks/Lab_06_Gradient_Descent_and_DataLoaders/Lab_06_Part_1_Gradient_Descent_Batch_vs_SGD.ipynb) | Lab 6 Part 1: Empirical comparison of Batch vs SGD with execution plots |
| **Lab Notebook** | [Lab_06_Part_2_PyTorch_Dataset_DataLoader_Pipeline.ipynb](04_Notebooks/Lab_06_Gradient_Descent_and_DataLoaders/Lab_06_Part_2_PyTorch_Dataset_DataLoader_Pipeline.ipynb) | Lab 6 Part 2: Deep dive into PyTorch `Dataset` and `DataLoader` machinery |
| **Lab Notebook** | [Lab_06_Part_3_PyTorch_Training_Pipeline.ipynb](04_Notebooks/Lab_06_Gradient_Descent_and_DataLoaders/Lab_06_Part_3_PyTorch_Training_Pipeline.ipynb) | Lab 6 Part 3: End-to-end training pipeline with dataset loaders & evaluation |

---

### Module 6: Optimizers with Memory (Momentum, NAG, RMSProp, Adam)
*Why raw SGD gets stuck in ravines, exponentially weighted moving averages (EWMA), Momentum, Nesterov Accelerated Gradient (NAG), and the Optimizer Zoo.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheet** | [Worksheet_06_Optimizers_With_Memory_Student.pdf](02_Worksheets/Worksheet_06_Optimizers_With_Memory_Student.pdf) | Worksheet 06: Student practice worksheet (unsolved) |
| **Worksheet (Solved)** | [Worksheet_06_Optimizers_With_Memory_Solved.pdf](02_Worksheets/Solved/Worksheet_06_Optimizers_With_Memory_Solved.pdf) | Worksheet 06: Full step-by-step solved key derivations |
| **Worksheet / Web** | [Worksheet_07_Momentum_EWMA_and_NAG_Interactive.html](02_Worksheets/Worksheet_07_Momentum_EWMA_and_NAG_Interactive.html) | Interactive Worksheet 07: SGD &rarr; EWMA &rarr; Momentum &rarr; NAG |
| **Worksheet / Web** | [Worksheet_08_Adaptive_Learning_AdaGrad_RMSProp_and_Adam_Interactive.html](02_Worksheets/Worksheet_08_Adaptive_Learning_AdaGrad_RMSProp_and_Adam_Interactive.html) | Interactive Worksheet 08: Adaptive Learning Rates &middot; AdaGrad &rarr; RMSProp &rarr; Adam |
| **Visualization** | [Module_06_Gradient_Descent_Interactive_Lecture.html](03_HTML_Visualizations/Module_06_Gradient_Descent_Interactive_Lecture.html) | Explorable gradient descent trajectory visualizer |
| **Visualization** | [Module_06_Momentum_and_NAG_Interactive_Worksheet.html](03_HTML_Visualizations/Module_06_Momentum_and_NAG_Interactive_Worksheet.html) | Interactive momentum and NAG comparison tool |
| **Visualization** | [Module_06_Adaptive_Learning_Rates_Interactive_Worksheet.html](03_HTML_Visualizations/Module_06_Adaptive_Learning_Rates_Interactive_Worksheet.html) | Interactive adaptive learning rates comparison tool (AdaGrad, RMSProp, Adam) |
| **Lab Notebook** | [Lab_07_Part_A_Optimizers_Zoo_Ravine_Trajectories.ipynb](04_Notebooks/Lab_07_The_Optimizer_Zoo/Lab_07_Part_A_Optimizers_Zoo_Ravine_Trajectories.ipynb) | Lab 7 Part A: 2D quadratic ravine loss optimizer trajectories |
| **Lab Notebook** | [Lab_07_The_Optimizer_Zoo_SGD_to_Adam.ipynb](04_Notebooks/Lab_07_The_Optimizer_Zoo/Lab_07_The_Optimizer_Zoo_SGD_to_Adam.ipynb) | Lab 7: The complete Optimizer Zoo navigating SGD, Momentum, NAG, AdaGrad, RMSProp, and Adam |

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
3. **Lab Notebooks Mirror**:
   The `Deep Learning Lab Notebooks/` directory maintains a streamlined 1-to-1 mirror (`LAB_0` through `LAB_8`) matching official course lab submissions.
4. **Accessing Cloud Sources**:
   - Slides, Notes & Worksheets: [DL Notes Drive](https://drive.google.com/drive/folders/1dFwGfgOvE9a-1veKjbuvSMYV3qb5Swe5)
   - Online Colab Notebooks: [DL Lab Notebooks Drive](https://drive.google.com/drive/folders/1cb0M-aseFx7djB3bF0OrtyB14QLSfaaV)

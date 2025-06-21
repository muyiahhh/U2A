### Bridging the Gap Between Preference Alignment and Machine Unlearning

This guide describes how to set up the environment, prepare datasets, fine-tune the original model, run the U2A method, and perform evaluation. The goal is to facilitate reproducible research and enable users to apply U2A to alignment tasks on large language models.

---

### 1. Environment Setup

First, create a Conda environment and install the necessary dependencies:

```bash
conda create -n u2a python=3.10
conda activate u2a
pip install -r requirements.txt
```

---

### 2. Dataset Download and Preprocessing

We use two public datasets in this project:

* [`PKU-Alignment/PKU-SafeRLHF`](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
* [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)

After downloading, run the following script to preprocess and split the data:

```bash
python create_datasets.py
```

This script processes the raw datasets and generates task-specific subsets, including `PA`, `original`, `forget`, `remain`, and `test` splits. A dataset split diagram is provided below to illustrate the structure:
![Dataset Split](./dataset_split.pdf)

---

### 3. Fine-tuning the Original Model

Using the `original` subset, we fine-tune a base model using the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository. Follow the instructions provided there to configure and launch training.

---

### 4. Running the U2A Method

Once the original model is trained, execute `u2a.py` to perform alignment-aware fine-tuning. We support three variants of the U2A method:

* **U2A + GA**: using a Genetic Algorithm-based strategy
* **U2A + GradDiff**: using Gradient Difference-based optimization
* **U2A + NPO**: using a Near-Policy Optimization method

Run the following scripts to apply each variant:

```bash
bash run_u2a_ga_3_safe.sh      # U2A + GA
bash run_u2a_grad_3_safe.sh    # U2A + GradDiff
bash run_u2a_npo_3_safe.sh     # U2A + NPO
```

These experiments are conducted on the `PKU-SafeRLHF` dataset using the LLaMA 3 model.

---

### 5. Evaluation

# Masterarbeit
## make_medmnist_c.py
Using this file, the corruptions from MedMNIST-C can be effortlessly generated and stored for every resolution, including 28x28, 64x64, 128x128, and 224x224.
## Code Structure for mm-PT
The interaction between the code files in the mm-PT directory is structured to ensure modularity and clarity. While most files are interdependent, an essential separation exists between training and testing processes. During training, the focus is on producing training metrics and validation metrics for the models. Notably, no evaluation on the test set is conducted during this phase. The best model and the corresponding heads are saved for evaluation later.

The following files collaborate to ensure an efficient and seamless training pipeline:
- **`mm-pt_config.yaml`**: Configuration file specifying training parameters and settings.  
- **`main.py`**: Entry point for running the training process.  
- **`model_n_data.py`**: Handles dataset loading and preprocessing.  
- **`multi_head_multi_domain_pt.py`**: Implements multi-head and multi-domain training logic.  
- **`multi_head_multi_domain_pt_gradient_accumulation.py`**: Extends multi-head and multi-domain training logic with gradient accumulation.  
- **`backbones.py`**: Contains backbone architecture definitions used.  
- **`utils.py`**: Provides utility functions to support the training process.

For training, the core logic resides in multi_head_multi_domain_pt.py. This file implements the sampling procedure central to the multi-domain, multi-task training paradigm outlined by Woerner et al. (2024).

For testing there are 2 files: 
- **`evaluate_multi_head.py`**: Evaluates trained models on the test set of MedMNIST+.  
- **`evaluate_multi_head_corrupted.py`**: Evaluates trained models on the corresponding test set of MedMNIST-C.

## Code Structure for mm-PT aug
The code structure is the same for mm-PT aug in comparision with mm-PT. Following modifications where made

- **`model_n_data.py`**: Added data augmentation to breastmnist and retinamnist
- **`multi_head_multi_domain_pt.py`**: Modified the logic that the model sees every dataset the same number of times
- **`multi_head_multi_domain_pt_gradient_accumulation.py`**: Modified the logic that the model sees every dataset the same number of times

Testsing is the same to mm-PT.

## Code Structure for pre-trained-training
- **`config.yaml`**: Configuration file specifying training parameters and settings.  
- **`feature_extraction.py`**: Extracts the features from the underlying backbones so only the heads can be trained by using the image embeddings
- **`evaluate_with_embeddings_*.py`**: Every file trains 12 heads for the 12 datasets of MedMNIST+ using a different classifier (* = kNN,  lightGBM, Random Forest and SVM), saves the models and evaluates the models and saves the metrics.
- **`train_linear_with_embeddings.py`**: Trains 12 heads for the 12 datasets of MedMNIST+ using a linear classifier and saves the models.
- **`evaluate_with_embeddings_linear.py`**: Evaluates the trained models on train, validation and test, set.
- **`evaluate_mutli_head_corrupted_*.py`**: Evaluates the trained and saved heads on the corrupted MedMNIST-C.
Cited sources:

Woerner, S., Jaques, A., & Baumgartner, C.F. (2024). A comprehensive and easy-to-use multi-domain multi-task medical imaging meta-dataset (MedIMeta). ArXiv, abs/2404.16000.

The work and the code are based on the work of [Sebastian Dörrich](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus) and [Francesco di Salvo](https://github.com/francescodisalvo05/medmnistc-api) and their respective papers.


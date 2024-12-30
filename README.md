# Masterarbeit
## make_medmnist_c.py
Using this file, the corruptions from MedMNIST-C can be effortlessly generated and stored for every resolution, including 28x28, 64x64, 128x128, and 224x224.
## Code Structure for mm-PT
The interaction between the code files in the model-training directory is structured to ensure modularity and clarity. While most files are interdependent, an essential separation exists between training and testing processes. During training, the focus is on producing training metrics and validation metrics for the latest, best, and final models. Notably, no evaluation on the test set is conducted during this phase.

The following files collaborate to ensure an efficient and seamless training pipeline:
- **`mm-pt_config.yaml`**: Configuration file specifying training parameters and settings.  
- **`main.py`**: Entry point for running the training process.  
- **`model_n_data.py`**: Handles dataset loading and preprocessing.  
- **`multi_head_multi_domain_pt.py`**: Implements multi-head and multi-domain training logic.  
- **`multi_head_multi_domain_pt_gradient_accumulation.py`**: Extends multi-head and multi-domain training logic with gradient accumulation.  
- **`backbones.py`**: Contains backbone architecture definitions used in the models.  
- **`utils.py`**: Provides utility functions to support the training process.  


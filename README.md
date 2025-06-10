# Spectrogram Image Classification with ResNet-50 (PyTorch)

## Overview

This Jupyter Notebook implements a 2D-CNN using **PyTorch** for multi-class classification of **spectrogram images**. It makes use of  **transfer learning** with a modified **ResNet-50** architecture.

---

## Data handling


Organize your dataset into separate folders for each class, with images stored inside their respective class directories. For example:

        ```
        dataset/
        ├── class_1/
        │   ├── img1.png
        │   ├── img2.png
        │   └── ...
        ├── class_2/
        │   ├── img1.png
        │   ├── img2.png
        │   └── ...
        └── class_3/
            ├── img1.png
            └── ...
        ```

Each subfolder under `dataset/` should be named after the class label and contain all images belonging to that class.

Then the `split_data()` function divides your dataset into training, validation, and test sets, with custom ratios. 

Data is then transformed using `torchvision.transforms` to include resizing (to the inout size used by ResNet50), normalization, and optional grayscale conversion. The images are then loaded into PyTorch datasets and dataloaders for training and evaluation.

## Model architecture

The model is based on a modified ResNet-50 architecture, which is adapted for image classification tasks. The model includes:
- Transfer Learning - Uses a pre-trained ResNet-50 model, allowing the model to use learned features from a large dataset.
- Final Fully Connected Layer- The final layer is replaced with a fully connected layer that matches the number of classes in your dataset. There is added dropout.

By default the model is configured to use a grayscale input, by averaging out the weights in the first layer across the 3 channels, but this can be changed by setting `grayscale=False` in the configuration.

The model is defined in the class SpectrogramResNet50. It takes the number of classes as an argument and initializes the ResNet-50 model with the appropriate modifications. Other parameters such as dropout rate and optimizer type can also be configured.

## Model Configuration
Model configurations can be adjusted to optimize performance. The following parameters can be modified:
batch_size, learning rate (lr), dropout rate, optimizer type (Adam or SGD), grayscale input, and whether to freeze the backbone layers of the ResNet-50 model.

Different configurations can be tested using stratified k-fold cross-validation to find the best performing model.

This is handled by the function `k_fold_cv()`.

The function then outputs a DataFrame object containing the results of each fold, including mean and standard deviation for each of the tested configurations.

From the results, the best performing configuration can be selected based on validation accuracy and can be used to train the final model on the entire training dataset like we did in the notebook.

## Training and Evaluation
The model is trained using the `train_model()` function, which handles the training loop, validation, and evaluation of the model. The function is adapted from: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html ; with original author: Sasank Chilamkurthy, License: BSD ;

The function trains the model for a specified number of epochs, evaluates it on the validation set, and saves the best model based on validation accuracy. 

All models were trained with it for 25 epochs.

## Results
The model's performance is evaluated on the test set, and the results are printed, including accuracy, precision, recall, and F1-score for each class.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- pandas
- numpy
- matplotlib


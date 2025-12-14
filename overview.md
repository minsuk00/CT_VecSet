# VecSetX Codebase Overview

This document provides a comprehensive overview of the VecSetX codebase, aimed at beginners who are new to the project.

## Project Goal

VecSetX is a research-oriented framework for 3D shape representation learning based on the [VecSet](https://arxiv.org/abs/2301.11445) architecture. It aims to reconstruct 3D shapes, represented as point clouds, using a deep learning model. The core of the project is an autoencoder that learns a compressed representation (latent code) of a 3D shape, which can then be used to reconstruct the original shape. This project is specifically designed to regress Signed Distance Function (SDF) values, which allows for creating high-quality mesh reconstructions.

The codebase is structured to facilitate experimentation with different model configurations, training parameters, and dataset variations.

## Codebase Structure

The project is organized into the following main directories and files:

-   `vecset/`: The main Python package containing the core source code.
    -   `main_ae.py`: The main script for training the autoencoder model.
    -   `infer.py`: A script for running inference with a trained model to generate a 3D mesh from an input point cloud.
    -   `engines/engine_ae.py`: Contains the training logic for one epoch, including the loss calculation and optimization steps.
    -   `models/`: This directory contains the neural network model definitions.
        -   `autoencoder.py`: Defines the main `VecSetAutoEncoder` architecture.
        -   `bottleneck.py`: Implements different bottleneck layers for the autoencoder, which are crucial for learning a compressed representation.
        -   `utils.py`: Contains utility functions and classes for building the model, such as attention mechanisms and embedding layers.
    -   `utils/`: This directory contains various helper scripts.
        -   `objaverse.py`: Defines the dataset class for loading data from the Objaverse dataset.
        -   `lr_sched.py` and `lr_decay.py`: These files handle the learning rate scheduling during training.
        -   `misc.py`: A collection of miscellaneous utility functions, such as for distributed training and logging.
-   `README.md`: Provides instructions on how to set up the environment, train the model, and run inference.
-   `LICENSE`: The license file for the project.
-   `.gitignore`: Specifies which files and directories to ignore in Git.

## Core Concepts and Important Files

### `vecset/main_ae.py`

This is the entry point for training the model. It performs the following key functions:

-   **Argument Parsing:** It uses `argparse` to define and parse command-line arguments for training, such as batch size, learning rate, and model name.
-   **Distributed Training Setup:** It initializes the distributed training environment if multiple GPUs are used.
-   **Dataset and DataLoader:** It creates instances of the `Objaverse` dataset for training and validation and wraps them in `DataLoader` for efficient data loading.
-   **Model Initialization:** It dynamically creates the specified autoencoder model from `models/autoencoder.py`.
-   **Optimizer and Loss Function:** It sets up the AdamW optimizer and the L1Loss criterion.
-   **Training Loop:** It iterates through the specified number of epochs, calling the `train_one_epoch` function from `vecset/engines/engine_ae.py` in each epoch.
-   **Checkpointing:** It saves the model checkpoints periodically.

### `vecset/engines/engine_ae.py`

This file contains the core training logic within the `train_one_epoch` function. Here's how it works:

1.  **Iterates through Data:** It loops through the `data_loader` to get batches of training data.
2.  **Forward Pass:** For each batch, it passes the input point cloud and query points through the model to get the predicted SDF values.
3.  **Loss Calculation:** It calculates a composite loss that includes several components:
    -   `loss_vol`: The L1 loss between the predicted SDF and the ground truth SDF for points in the volume.
    -   `loss_near`: The L1 loss for points near the surface.
    -   `loss_eikonal`: An Eikonal loss that regularizes the gradients of the SDF, encouraging them to have a unit norm. This is important for generating a valid SDF.
    -   `loss_surface`: A loss that encourages the predicted SDF to be zero on the surface of the shape.
4.  **Backward Pass and Optimization:** It performs a backward pass to compute gradients and updates the model's weights using the optimizer.
5.  **Logging:** It logs various metrics like the different loss components and Intersection over Union (IoU) to monitor the training progress.

### `vecset/models/autoencoder.py`

This file defines the `VecSetAutoEncoder` model, which is a Transformer-based architecture. Here's a breakdown of its components:

-   **Encoder:** The `encode` method takes a point cloud as input and produces a set of latent vectors. It uses cross-attention to distill information from the input point cloud into a smaller set of latent queries.
-   **Bottleneck:** The output of the encoder is passed through a bottleneck layer (`bottleneck.py`), which can be a simple linear projection, a variational autoencoder (VAE) bottleneck with a KL-divergence loss, or a normalized bottleneck.
-   **Decoder:** The `decode` method takes the processed latent vectors and a set of query points as input and predicts the SDF value for each query point. It uses cross-attention to query the latent vectors for information at the given spatial locations.
-   **Model Variations:** The file also defines several factory functions (e.g., `learnable_vec1024x16_dim1024_depth24_nb`) that create `VecSetAutoEncoder` instances with specific configurations of depth, dimension, and bottleneck type.

### `vecset/infer.py`

This script demonstrates how to use a trained model for inference. It loads a checkpoint, prepares a grid of points, and then queries the model to get the SDF values for the entire grid. Finally, it uses the marching cubes algorithm (from the `mcubes` library) to extract a mesh from the SDF volume and saves it as an OBJ file.

## Adapting for 3D CT Volume Regression

The user asked how to adapt this codebase to regress intensity values of 3D CT volumes instead of reconstructing point clouds. Here's a breakdown of the necessary changes:

The current architecture is designed to take a sparse point cloud representing the surface of an object and learn a continuous SDF representation of the object's shape. To adapt this to a CT scan regression task (voxel-to-voxel), you would need to make the following conceptual and architectural changes:

### 1. **Dataset (`vecset/utils/objaverse.py`)**

-   You would need to replace the `Objaverse` dataset with a new dataset class for your CT scan data.
-   This new dataset class should handle loading your 3D CT volumes (e.g., from NIfTI or DICOM files).
-   The `__getitem__` method would need to be modified to return a pair of (input, target) volumes. For a voxel-to-voxel reconstruction task, the input and target might be the same CT volume, or the input could be a downsampled/corrupted version of the target.
-   Instead of sampling SDF points, you would be dealing with 3D grids of voxel intensities.

### 2. **Model Input and Output (`vecset/models/autoencoder.py`)**

-   The current model takes a `(B, N, 3)` tensor (a batch of N 3D points) as input. For CT scans, the input would be a 3D or 5D tensor, e.g., `(B, C, D, H, W)` where D, H, W are the dimensions of the volume and C is the number of channels (usually 1 for CT scans).
-   The `PointEmbed` layer, which uses sinusoidal embeddings for 3D coordinates, would no longer be suitable. You would need a different way to process the input volume. A 3D convolutional neural network (CNN) would be a more natural choice for the encoder to process the input grid.
-   The output of the model is currently a single SDF value per query point. For CT regression, the output of the decoder should be a 3D volume of the same dimensions as the target CT scan, with each voxel representing the predicted intensity.

### 3. **Model Architecture (`vecset/models/autoencoder.py`)**

-   **Encoder:** You would likely replace the Transformer-based encoder with a 3D CNN encoder. This encoder would downsample the input CT volume through a series of 3D convolutions and pooling layers to produce a flat latent vector or a set of latent vectors.
-   **Decoder:** The decoder would also need to be a 3D CNN, often with a symmetric architecture to the encoder. It would take the latent representation and upsample it using 3D transposed convolutions (or upsampling followed by convolutions) to reconstruct the full-resolution CT volume. This is a common architecture for volumetric autoencoders, like a 3D U-Net.
-   **Bottleneck:** The bottleneck part of the model could remain similar. You would still have a compressed latent representation between the encoder and the decoder.

### 4. **Loss Function (`vecset/engines/engine_ae.py`)**

-   The current loss function is tailored for SDF regression. You would need to replace it with a loss function suitable for image/volume regression.
-   A common choice is the **Mean Squared Error (MSE)** loss or **Mean Absolute Error (MAE/L1Loss)** between the predicted and target CT volumes.
-   You might also consider more advanced losses like SSIM (Structural Similarity Index) if preserving the structural details of the CT scan is important.
-   The Eikonal loss and other SDF-specific losses would no longer be relevant and should be removed.

### 5. **Inference (`vecset/infer.py`)**

-   The inference script would need to a class-conditional latent diffusion model (`EDMPrecond`) is implemented to generate latent vectors. |
| **Autoencoder Output**| Signed Distance Function (SDF) values. | Occupancy values (binary classification). |
| **Loss Function (AE)** | L1 Loss on SDF values, with an Eikonal regularization term. | Binary Cross-Entropy (BCE) with Logits Loss on occupancy values. Also supports KL-divergence for VAEs. |
| **Dataset** | Objaverse (`vecset/utils/objaverse.py`) | ShapeNet (`util/shapenet.py`) |
| **Model Architecture (AE)** | `VecSetAutoEncoder`. Transformer-based, with flexible bottleneck options (`Bottleneck`, `KLBottleneck`, `NormalizedBottleneck`). Encoder uses cross-attention between learnable/point queries and the input point cloud. | `AutoEncoder` / `KLAutoEncoder`. Also Transformer-based. The encoder uses FPS to sample points from the input, then uses cross-attention. Less flexible bottleneck. |
| **Training Scripts** | `main_ae.py` for the autoencoder. | `main_ae.py` for the autoencoder, `main_class_cond.py` for the diffusion model. |
| **Inference/Sampling**| `infer.py` for reconstructing a mesh from a point cloud using the trained autoencoder. | `eval.py` for evaluating the autoencoder. `sample_class_cond.py` for sampling from the diffusion model to generate new shapes. |
| **Dependencies** | `flash-attn`, `torch-cluster`, `trimesh`, `PyMCubes`. | `timm`, `torch-cluster`, `trimesh`, `mcubes`. |
| **Code Structure** | All code is under the `vecset` package. | Code is in the root directory, with utilities in the `util` directory. |

### Diffusion Logic for Latent VecSet Generation

-   **`VecSetX`**: There is **no diffusion logic** in this codebase. The primary focus is on the autoencoder for shape representation and reconstruction. The `README.md` file does not mention any plans to incorporate diffusion models.

-   **`3DShape2VecSet`**: This codebase has a **complete implementation of a class-conditional latent diffusion model**.
    -   **Core Implementation**: The `models_class_cond.py` file contains the core of the diffusion model. The `EDMPrecond` class implements the diffusion model itself, following the architecture from the paper "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM).
    -   **Sampling**: The `edm_sampler` function in `models_class_cond.py` implements the sampler from the EDM paper, which is used to generate samples from the diffusion model.
    -   **Training**: The `main_class_cond.py` script is used to train the diffusion model on the latent space of a pre-trained autoencoder.
    -   **Inference**: The `sample_class_cond.py` script demonstrates how to use the trained diffusion model to generate new shapes by sampling from the model and then decoding the generated latent vectors with the autoencoder.

### Summary of Differences

The most significant difference between the two codebases is the **presence of a generative model (a latent diffusion model) in `3DShape2VecSet`**, which is entirely absent in `VecSetX`. `VecSetX` is purely focused on representation learning with an autoencoder that regresses Signed Distance Functions (SDFs). In contrast, `3DShape2VecSet` not only includes an autoencoder (which predicts occupancy instead of SDFs) but also a state-of-the-art diffusion model for generating new shapes.

The choice of dataset (Objaverse vs. ShapeNet) and the output representation of the autoencoder (SDF vs. occupancy) are other key differences that reflect the distinct goals of the two projects. `VecSetX`'s focus on SDF regression is aimed at achieving high-quality surface reconstruction, whereas `3DShape2VecSet`'s approach is more geared towards generative tasks.

## Porting Features from 3DShape2VecSet to VecSetX

This section provides a high-level guide on how to transfer the key features from `3DShape2VecSet` to this `VecSetX` codebase.

### 1. Modifying the Autoencoder for Occupancy Prediction

To switch from SDF regression to occupancy prediction, you will need to make the following changes:

-   **Loss Function:** In `vecset/main_ae.py`, change the loss function from `torch.nn.L1Loss` to `torch.nn.BCEWithLogitsLoss`.
-   **Engine:** In `vecset/engines/engine_ae.py`, remove the SDF-specific losses (`loss_eikonal`, `loss_surface`). The loss calculation should be simplified to use only the `BCEWithLogitsLoss`.
-   **Dataset:** Modify the `vecset/utils/objaverse.py` dataset to return occupancy labels (0 or 1) instead of SDF values. This may require changing how the data is processed in the `__getitem__` method.

### 2. Integrating the Latent Diffusion Model

This is the most significant change and involves adding the diffusion model components to the codebase.

-   **Copy Diffusion Model Code:** Copy the `3DShape2VecSet/models_class_cond.py` file to `vecset/models/`. This file contains the `EDMPrecond` class, which is the core of the diffusion model, and the `edm_sampler` function.
-   **Create a New Training Script:** Create a new training script, `vecset/main_class_cond.py`, based on `3DShape2VecSet/main_class_cond.py`. This script will be responsible for training the diffusion model on the latent vectors generated by the autoencoder. You will need to adapt it to the `VecSetX` project structure.
-   **Create a New Training Engine:** Create a new engine file, `vecset/engines/engine_class_cond.py`, based on `3DShape2VecSet/engine_class_cond.py`. This file will contain the training logic for the diffusion model.
-   **Create a New Sampling Script:** Create a new sampling script, `vecset/sample_class_cond.py`, based on `3DShape2VecSet/sample_class_cond.py`. This script will allow you to generate new 3D shapes by sampling from the trained diffusion model and decoding the results with the autoencoder.

### 3. Switching to the ShapeNet Dataset

If you want to use the ShapeNet dataset instead of Objaverse, you will need to:

-   **Copy Dataset File:** Copy the `3DShape2VecSet/util/shapenet.py` file to `vecset/utils/`.
-   **Update Training Scripts:** In both `vecset/main_ae.py` and the new `vecset/main_class_cond.py`, change the dataset being used from `Objaverse` to `ShapeNet`. This will involve updating the dataset creation and data loading sections of the code.

By following these steps, you can integrate the generative capabilities of `3DShape2VecSet` into the `VecSetX` framework.

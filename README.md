# GAN RGB Image Generator (CIFAR-10)

This project implements a **Generative Adversarial Network (GAN)** using **TensorFlow / Keras** to generate **RGB images similar to CIFAR-10 images (32×32)**.

The model consists of two neural networks:

* **Generator** – creates fake images from random noise
* **Discriminator** – distinguishes between real and generated images

Both networks are trained **adversarially** so that the generator gradually learns to produce realistic images.

---

## Project Overview

The training pipeline:

1. Load the **CIFAR-10 dataset**
2. Normalize images to **[-1, 1]**
3. Train a **Generator** to create RGB images
4. Train a **Discriminator** to classify real vs fake
5. Save generated images each epoch
6. Save checkpoints to resume training

---

## Architecture

### Generator

The **Generator** converts a random noise vector into a **32×32 RGB image**.

#### Input

Random noise vector:

$$z \in \mathbb{R}^{100}$$

#### Architecture

```
Dense (8×8×256)
BatchNorm
LeakyReLU

Reshape → (8, 8, 256)

Conv2DTranspose
Filters: 128
Kernel: 4×4
Stride: 2
Output: 16×16×128
BatchNorm
LeakyReLU

Conv2DTranspose
Filters: 64
Kernel: 4×4
Stride: 2
Output: 32×32×64
BatchNorm
LeakyReLU

Conv2DTranspose
Filters: 3
Kernel: 3×3
Stride: 1
Activation: tanh
Output: 32×32×3 (RGB image)
```

#### Generator Summary

```
Noise (100)
   ↓
Dense
   ↓
8×8×256
   ↓
Deconvolution Layers
   ↓
32×32×3 RGB Image
```

The **tanh activation** is used because the dataset is normalized to **[-1, 1]**.

---

### Discriminator

The **Discriminator** is a CNN that classifies images as **real or fake**.

#### Input

```
Image: 32×32×3
```

#### Architecture

```
Conv2D
Filters: 64
Kernel: 4×4
Stride: 2
Output: 16×16×64
LeakyReLU
Dropout (0.3)

Conv2D
Filters: 128
Kernel: 4×4
Stride: 2
Output: 8×8×128
LeakyReLU
Dropout (0.3)

Flatten

Dense (1)
Output: Real / Fake score
```

---

## Loss Functions

Binary cross-entropy is used.

### Discriminator Loss

$$\mathcal{L}_D = \text{BCE}(D(x_{real}), 1) + \text{BCE}(D(G(z)), 0)$$

Where:
- $D(x_{real})$ is the discriminator output for real images
- $D(G(z))$ is the discriminator output for generated images
- $G(z)$ is the generator output

### Generator Loss

$$\mathcal{L}_G = \text{BCE}(D(G(z)), 1)$$

The generator tries to **fool the discriminator** by maximizing the probability that the discriminator classifies generated images as real.

---

## Training Details

| Parameter       | Value     |
| --------------- | --------- |
| Dataset         | CIFAR-10  |
| Image Size      | 32×32 RGB |
| Noise Dimension | 100       |
| Batch Size      | 128       |
| Epochs          | 50        |
| Optimizer       | Adam      |
| Learning Rate   | 0.0001    |

---

## Dataset

The dataset is automatically downloaded using TensorFlow:

```python
tf.keras.datasets.cifar10.load_data()
```

Images are normalized to:

$$[-1, 1]$$

---

## Checkpointing

The training automatically saves model checkpoints to **Google Drive**.

```
/content/drive/MyDrive/gan_rgb_checkpoints
```

If training stops, it will **resume automatically from the last checkpoint**.

---

## Generated Images

After every epoch, generated images are saved to:

```
/content/drive/MyDrive/gan_rgb_images
```

Example grid:

```
16 generated samples per epoch
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/gan-rgb-generator.git
cd gan-rgb-generator
```

Install dependencies:

```bash
pip install tensorflow matplotlib numpy
```

---

## Running the Project

This project is designed to run on **Google Colab**.

### Step 1 — Upload the script to Colab

Open **Google Colab** and upload the training file.

### Step 2 — Mount Google Drive

The script automatically runs:

```python
drive.mount('/content/drive')
```

Allow permission when prompted.

### Step 3 — Run the notebook

Execute the script:

```
Runtime → Run All
```

Training will begin.

---

## Training Output

Each epoch prints:

```
Epoch 5 | Gen Loss: 1.24 | Disc Loss: 0.85
```

And generates a **4×4 grid of fake images**.

---

## Example Training Pipeline

```
Noise (100)
     ↓
Generator
     ↓
Fake Image
     ↓
Discriminator
     ↓
Real / Fake
     ↓
Loss
     ↓
Backpropagation
```

---

## Future Improvements

Possible improvements include:

* Using **DCGAN architecture**
* Adding **Spectral Normalization**
* Using **WGAN-GP**
* Training on **larger datasets**
* Using **FID score evaluation**

---

## License

MIT License

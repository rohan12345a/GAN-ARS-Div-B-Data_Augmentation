### Conditional Generative Adversarial Nets/CGANs
![68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3639382f302a4c386c6f574251494a6f5572505230302e706e67](https://github.com/rohan12345a/GAN-ARS-Div-B-Data_Augmentation/assets/109196424/f9f66007-32ef-443c-9723-c441aebd91c4)


### OVERVIEW

1. **main.py:**
   - This file contains all the code related to CGAN implementation, from data loading to model training and evaluation.
   - It includes functions for data loading, defining the generator and discriminator networks, specifying the loss function and optimizer, and setting up the training process.
   - At the end of the file, evaluation metrics such as Intra-class Inception Distance (IID), Kernel Inception Distance (KID), and Inception Score (IS) are calculated to assess the performance of the trained model.

2. **deployment.py:**
   - This file contains the code for deploying the trained CGAN model using Streamlit.
   - It provides a user-friendly interface for data augmentation, allowing users to select a specific food category and specify the number of images to generate.
   - Users also have the option to apply various image augmentation techniques such as rotation, flips, cropping, scaling, and adjusting brightness/contrast.
   - The generated images can be downloaded as a zip file for further use in machine-learning/deep-learning tasks.


### Data Description

The dataset used in this project is the Food-101 dataset, which contains a diverse collection of food images spanning various categories. Each food category consists of high-resolution images showcasing different variations and presentations of the respective dishes. The dataset is organized into folders, with each folder representing a specific food category.

#### Selected Classes:

- Garlic Bread
- Chicken Wings
- Frozen Yogurt
- Dumplings
- French Fries
- Cheese Cake
- Donut
- Cupcakes

#### Dataset Composition:

- Each class folder contains approximately 1000 images.
- The images are resized to dimensions of 64x64 pixels for training the CGAN model.

DATASET LINK: https://www.kaggle.com/datasets/kmader/food41


### VIDEO DEMONSTRATION

https://github.com/rohan12345a/GAN-ARS-Div-B-Data_Augmentation/assets/109196424/b48fec95-021c-4426-b339-346196278b17


### Usage and Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   cd your-repository
   git clone https://github.com/rohan12345a/GAN-ARS-Div-B-Data_Augmentation.git
   python main.py
   streamlit run deployment.py
   



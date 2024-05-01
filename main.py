# LOADING DATA AND MAPING

%matplotlib inline
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# For all IMages
import os
from PIL import Image
from torch.utils.data import Dataset

class Foodimages(Dataset):
    def __init__(self, root_dir, transform=None, target_folders=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []  # List to store image paths
        self.labels = []  # List to store food labels
        self.target_folders = target_folders if target_folders else []

        # Map target folders to labels
        self.label_map = {folder: idx for idx, folder in enumerate(self.target_folders)}
        # Labeling of images
        for folder in self.target_folders:
            folder_path = os.path.join(root_dir, folder)
            if os.path.exists(folder_path):
                images_in_folder = os.listdir(folder_path)
                for image_name in images_in_folder:
                    image_path = os.path.join(folder_path, image_name)
                    self.images.append(image_path)
                    self.labels.append(self.label_map[folder])
    # Total no of images
    def __len__(self):
        return len(self.images)
    #getting images
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(image_path)

        if self.transform:
            img = self.transform(img)

        return img, label



from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize
])

target_folders = ['chicken_wings', 'cheesecake', 'cup_cakes', 'donuts', 'dumplings', 'french_fries', 'frozen_yogurt', 'garlic_bread']

dataset = Foodimages(root_dir='/home/student/FoodDataset/images', transform=transform, target_folders=target_folders)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in data_loader:
    pass




# Ploting the images:

import matplotlib.pyplot as plt
import numpy as np
import torchvision
# Function to display images in a grid
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

images, labels = next(iter(data_loader))
imshow(torchvision.utils.make_grid(images))



# Class Discriminator:

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes=8, img_channels=3):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.conv_layers = nn.Sequential(
            # Input: (img_channels) x 64 x 64
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 64 x 32 x 32

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 256 x 8 x 8

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: 512 x 4 x 4
        )

        self.fc = nn.Linear(512*4*4 + num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten convolutional features
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out.squeeze()



# Class Generator:

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, num_classes=8, img_channels=3, latent_dim=100):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.init_size = 64 // 4  # Initial size before upsampling
        self.img_channels = img_channels

        self.fc = nn.Linear(latent_dim + num_classes, 1024 * self.init_size ** 2)

        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), -1)  # Flatten input noise
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.fc(x)
        out = out.view(out.size(0), 1024, self.init_size, self.init_size)  # Reshape to feature map shape
        out = self.conv_layers(out)
        return out


# Initilizing Loss functions:

# Loss function
criterion = nn.BCELoss()  #binary cross entropy loss

# Optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))


# Traning loop and forward and backward propogation of Discriminator and Generator:

import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os

# Define the generator and discriminator architectures
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, num_classes=8):
    g_optimizer.zero_grad()

    # Generate random noise and random labels
    z = torch.randn(batch_size, 100)  # Random noise
    fake_labels = torch.randint(0, num_classes, (batch_size,))  # Random labels

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)
    fake_labels = fake_labels.to(device)

    # Generate fake images
    fake_images = generator(z, fake_labels)

    # Discriminator's prediction on fake images
    validity = discriminator(fake_images, fake_labels)

    # Generator loss
    g_loss = criterion(validity, torch.ones(batch_size, device=device))  # Target labels are 1 (indicating real)

    # Backpropagation
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()  # .item() is used to get a Python number from a tensor

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels, num_classes=8):
    d_optimizer.zero_grad()

    # Ensure labels are within the correct range
    labels = torch.clamp(labels, 0, num_classes - 1)

    # Train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, torch.ones(batch_size, device=real_images.device))  # Target labels are 1 (indicating real)

    # Train with fake images
    z = torch.randn(batch_size, 100, device=real_images.device)  # Random noise
    fake_labels = torch.randint(0, num_classes, (batch_size,), device=real_images.device)  # Random labels
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, torch.zeros(batch_size, device=real_images.device))  # Target labels are 0 (indicating fake)

    # Total discriminator loss
    d_loss = real_loss + fake_loss

    # Backpropagation
    d_loss.backward()
    d_optimizer.step()

    return d_loss.item()  # .item() is used to get a Python number from a tensor

# Define your generator and discriminator objects
generator = Generator()
discriminator = Discriminator()


# Define your DataLoader object
data_loader = data_loader  # Define your DataLoader object

# Define your training parameters
num_epochs = 3000
n_critic = 5
display_step = 300

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# Create the directory to save the models
model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Specify the file name pattern for saving models
model_file_pattern = os.path.join(model_dir, 'epoch_{}.pt')

# Training loop
for epoch in range(num_epochs):
    print('Starting epoch {}...'.format(epoch))

    for i, (images, labels) in enumerate(data_loader):
        real_images = images.to(device)
        labels = labels.to(device)

        # Train discriminator
        for _ in range(n_critic):
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                              generator, d_optimizer, criterion,
                                              real_images, labels)

        # Train generator
        g_loss = generator_train_step(len(real_images), discriminator,
                                      generator, g_optimizer, criterion)

        # Display losses
        if i % display_step == 0:
            print('[Epoch {}/{}][Batch {}/{}] => Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(
                epoch+1, num_epochs, i+1, len(data_loader), g_loss, d_loss))

    # Save the model at the end of each epoch
    torch.save(generator.state_dict(), model_file_pattern.format(epoch))
    print("Saved model for epoch", epoch)

    # Remove previous model file if it exists
    if epoch > 0:
        prev_model_file = model_file_pattern.format(epoch - 1)
        if os.path.exists(prev_model_file):
            os.remove(prev_model_file)
            print("Deleted previous model file:", prev_model_file)

    # Generate sample images at the end of each epoch
    with torch.no_grad():
        generator.eval()
        z = torch.randn(8, 100, device=device)
        labels = torch.arange(8, device=device)
        sample_images = generator(z, labels).unsqueeze(1)
        sample_images = sample_images.squeeze(1).cpu()  # Move tensor to CPU
        grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        plt.imshow(grid)
        plt.axis('off')
        plt.show()



# Code for calculating IS given generated and original image:

import torch
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3
from scipy.stats import entropy
import numpy as np
from PIL import Image

def compute_inception_score(original_path, generated_path, device=torch.device('cpu'), splits=10):
    inception_model = inception_v3(pretrained=True, transform_input=True).to(device).eval()
    original_image = Image.open(original_path)
    generated_image = Image.open(generated_path)

    # Transform images to tensors
    original_tensor = ToTensor()(original_image).unsqueeze(0).to(device)
    generated_tensor = ToTensor()(generated_image).unsqueeze(0).to(device)

    # Compute predictions for the original image
    with torch.no_grad():
        original_pred = inception_model(original_tensor)

    # Compute softmax of the predictions for the original image
    original_pred = torch.softmax(original_pred, dim=1)

    # Compute the KL divergence for the original image
    kl_original = original_pred * (torch.log(original_pred) - torch.log(torch.mean(original_pred, dim=1, keepdim=True)))
    kl_original = torch.mean(torch.sum(kl_original, dim=1))

    with torch.no_grad():
        generated_pred = inception_model(generated_tensor)

    generated_pred = torch.softmax(generated_pred, dim=1)

    kl_generated = generated_pred * (torch.log(generated_pred) - torch.log(torch.mean(generated_pred, dim=1, keepdim=True)))
    kl_generated = torch.mean(torch.sum(kl_generated, dim=1))

    # Compute the Inception Score
    scores = torch.exp(kl_original), torch.exp(kl_generated)
    is_mean = torch.mean(torch.stack(scores))
    is_std = torch.std(torch.stack(scores))
    return is_mean.item(), is_std.item()

# Paths to the original and generated images
image_pairs = [
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cheesecake_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cheesecake_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/chicken-wings_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/chicken-wings_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cupcakes_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cupcakes_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/donut_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/donut_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/dumplings_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/dumplings_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/french-fries_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/french-fries_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/frozen_youg_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/frozen_yogurt_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/garlic-bread_orig.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/garlic-bread_gen.png"),
]
for original_path, generated_path in image_pairs:
    is_mean, is_std = compute_inception_score(original_path, generated_path)
    image_name = original_path.split('/')[-1].split('_')[0]  # Extract the image name from the path
    print(f'Inception Score (IS) for {image_name} image: {is_mean:.2f} ± {is_std:.2f}')




# For IID:

import torch
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3
from PIL import Image

def inception_intra_class_distance(real_path, generated_path, device=torch.device('cpu')):
    # Load InceptionV3 model pretrained on ImageNet
    inception_model = inception_v3(pretrained=True, transform_input=True).to(device).eval()
    real_image = Image.open(real_path)
    generated_image = Image.open(generated_path)

    # Transform images to tensors
    real_tensor = ToTensor()(real_image).unsqueeze(0).to(device)
    generated_tensor = ToTensor()(generated_image).unsqueeze(0).to(device)

    # Compute activations for the real and generated images
    with torch.no_grad():
        real_activations = inception_model(real_tensor)[0].view(-1)
        generated_activations = inception_model(generated_tensor)[0].view(-1)

    # Compute mean and standard deviation for real and generated activations
    real_mean = torch.mean(real_activations, dim=0)
    real_std = torch.std(real_activations, dim=0)
    generated_mean = torch.mean(generated_activations, dim=0)
    generated_std = torch.std(generated_activations, dim=0)

    mean_distance = torch.norm(real_mean - generated_mean, p=2)

    std_difference = torch.abs(real_std - generated_std)

    return mean_distance.item(), std_difference.item()

image_pairs = [
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cheesecake_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cheesecake_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/chicken-wings_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/chicken-wings_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cupcakes_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/cupcakes_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/donut_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/donut_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/dumplings_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/dumplings_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/french-fries_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/french-fries_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/frozen_youg_original.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/frozen_yogurt_gen.png"),
    ("/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/garlic-bread_orig.jpg", "/content/drive/MyDrive/DATASETS/GANS_IMAGES_DATA/GANS_FINAL/garlic-bread_gen.png"),
]

for real_path, generated_path in image_pairs:
    mean_distance, std_difference = inception_intra_class_distance(real_path, generated_path)
    image_name = real_path.split('/')[-1].split('_')[0]  # Extract the image name from the path
    print(f'Inception Intra-class Distance (IID) for {image_name} image pair: Mean Distance={mean_distance:.2f}, Std Difference={std_difference:.2f}')





# For KID:

import torch
from torchvision.transforms import ToTensor
from torchvision.models import inception_v3
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a function to compute feature embeddings
def compute_embeddings(images, inception_model, device=device):
    embeddings = []
    for image_path in images:
        image = Image.open(image_path)
        image_tensor = ToTensor()(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = inception_model(image_tensor)[0].view(-1)  # Extract the feature embeddings
            embeddings.append(features)
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings

# Define a function to compute the Gram matrix
def compute_gram_matrix(embeddings):
    num_samples, num_features = embeddings.shape
    gram_matrix = torch.matmul(embeddings, embeddings.t()) / num_features
    return gram_matrix

def kernel_inception_distance(image_pairs, inception_model, device=device):
    kid_scores = []
    for real_path, generated_path in image_pairs:
        real_embeddings = compute_embeddings([real_path], inception_model, device=device)
        generated_embeddings = compute_embeddings([generated_path], inception_model, device=device)
        real_gram_matrix = compute_gram_matrix(real_embeddings)
        generated_gram_matrix = compute_gram_matrix(generated_embeddings)
        kid_score = torch.norm(real_gram_matrix - generated_gram_matrix, p='fro').item()
        kid_scores.append(kid_score)
    return kid_scores

inception_model = inception_v3(pretrained=True, transform_input=True).to(device).eval()

kid_scores = kernel_inception_distance(image_pairs, inception_model, device=device)

for i, (real_path, generated_path) in enumerate(image_pairs):
    image_name = real_path.split('/')[-1].split('_')[0]  # Extract the image name from the path
    print(f'Kernel Inception Distance (KID) for {image_name} image pair: {kid_scores[i]:.2f}')



# import streamlit as st

# # Title
# st.title("DATA AUGMENTATION USING FOOD IMAGES")

# # Subheader
# st.subheader("DATA AUGMENTATION")

# # List of applications
# st.write("""
# Data augmentation is a technique used to artificially increase the size and diversity of a dataset by applying various transformations to the existing data samples. These transformations can include rotations, translations, flips, crops, changes in brightness or contrast, and more. By augmenting the data, the model is exposed to a wider range of variations in the input data, which helps improve its generalization and robustness. Data augmentation is commonly used in deep learning tasks, especially when dealing with
# limited amounts of labeled data, to prevent overfitting and improve the performance of the model.
# """)

# # Additional content or functionalities can be added here


# import torch
# import torch.nn as nn
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt


# # Define your Generator class
# class Generator(nn.Module):
#     def __init__(self, num_classes=8, img_channels=3, latent_dim=100):
#         super(Generator, self).__init__()
#         self.label_emb = nn.Embedding(num_classes, num_classes)
#         self.init_size = 64 // 4  # Initial size before upsampling
#         self.img_channels = img_channels
#         self.fc = nn.Linear(latent_dim + num_classes, 1024 * self.init_size ** 2)
#         self.conv_layers = nn.Sequential(
#             nn.BatchNorm2d(1024),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, img_channels, kernel_size=3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, z, labels):
#         z = z.view(z.size(0), -1)  # Flatten input noise
#         c = self.label_emb(labels)
#         x = torch.cat([z, c], 1)
#         out = self.fc(x)
#         out = out.view(out.size(0), 1024, self.init_size, self.init_size)  # Reshape to feature map shape
#         out = self.conv_layers(out)
#         return out

# ## Define the path to the pretrained generator model file
# generator_model_path = 'C:\\Users\\Lenovo\\Downloads\\GansDeployment\\Assets\\epoch_1367.pt'

# # Instantiate the Generator
# generator = Generator()

# # Load the pretrained model weights and map them to CPU
# generator.load_state_dict(torch.load(generator_model_path, map_location=torch.device('cpu')))

# # Set the generator to evaluation mode
# generator.eval()

# # Specify the label options
# class_labels = ['garlic_bread', 'chicken_wings', 'frozen_yogurt', 'dumplings', 'french_fries', 'cheese_cake', 'donut', 'cupcakes']

# # Display a selectbox for choosing the label
# selected_label = st.selectbox("Select a class label:", class_labels, index=0)

# # Display a submit button
# if st.button('Generate Image'):
#     # Generate and visualize images
#     with torch.no_grad():
#         # Generate a random noise vector for each image
#         latent_dim = 100
#         noise = torch.randn(1, latent_dim)

#         # Map the selected label to its corresponding index
#         label_idx = class_labels.index(selected_label)

#         # Repeat the label for the number of images to generate
#         labels = torch.tensor([label_idx])

#         # Generate an image from the random noise vector and the specified label using the pretrained generator
#         generated_image = generator(noise, labels).squeeze()

#         # Normalize the pixel values to the range [0.0, 1.0]
#         generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())

#         # Visualize the generated image
#         st.image(generated_image.permute(1, 2, 0).cpu().numpy(), caption=selected_label, use_column_width=True)





import zipfile
from matplotlib.image import imsave
import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
st.set_page_config(page_title="Image Augmentation App",page_icon="🍔")

# Title

# Subheader
st.title("DATA AUGMENTATION USING FOOD IMAGES")

st.subheader("DATA AUGMENTATION")


page_by_img = """

<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(90deg, #F2F2F2, #EAEAEA);
    background-size: 300% 300%;
    animation: gradientAnimation 12s ease infinite;
    opacity: 1;
}

[data-testid="stAppViewContainer"] body {
    font-family: serif;
}

@keyframes gradientAnimation {
    0% {
        background-position: 0% 50%;
    }
    100% {
        background-position: 100% 50%;
    }
</style>

"""


# List of applications
st.write("""
Data augmentation is a technique used to artificially increase the size and diversity of a dataset by applying various transformations to the existing data samples. These transformations can include rotations, translations, flips, crops, changes in brightness or contrast, and more. By augmenting the data, the model is exposed to a wider range of variations in the input data, which helps improve its generalization and robustness. Data augmentation is commonly used in deep learning tasks, especially when dealing with
limited amounts of labeled data, to prevent overfitting and improve the performance of the model.
""")


st.subheader("FEATURES")


st.write("""
Our application for data augmentation using food images offers a user-friendly interface with several key features.
 Users can select a specific food category from a list and choose the number of images they want to generate. They also have the option to apply various image augmentation techniques such as rotation, horizontal or vertical flips, cropping, scaling, and adjusting brightness/contrast. The application leverages a pre-trained deep learning model to generatefood images based on the selected category and augmentation preferences. Additionally, users can download the generated images as a zip file for further use. With these features, our application
 provides a convenient and efficient way to augment food image datasets for training machine learning models.
""")

# # Define your Generator class
# class Generator(nn.Module):
#     def __init__(self, num_classes=8, img_channels=3, latent_dim=100):
#         super(Generator, self).__init__()
#         self.label_emb = nn.Embedding(num_classes, num_classes)
#         self.init_size = 64 // 4  # Initial size before upsampling
#         self.img_channels = img_channels
#         self.fc = nn.Linear(latent_dim + num_classes, 1024 * self.init_size ** 2)
#         self.conv_layers = nn.Sequential(
#             nn.BatchNorm2d(1024),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(256, img_channels, kernel_size=3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, z, labels):
#         z = z.view(z.size(0), -1)  # Flatten input noise
#         c = self.label_emb(labels)
#         x = torch.cat([z, c], 1)
#         out = self.fc(x)
#         out = out.view(out.size(0), 1024, self.init_size, self.init_size)  # Reshape to feature map shape
#         out = self.conv_layers(out)
#         return out

# ## Define the path to the pretrained generator model file
# generator_model_path = 'C:\\Users\\Lenovo\\Downloads\\GansDeployment\\Assets\\epoch_1367.pt'

# # Instantiate the Generator
# generator = Generator()

# # Load the pretrained model weights and map them to CPU
# generator.load_state_dict(torch.load(generator_model_path, map_location=torch.device('cpu')))

# # Set the generator to evaluation mode
# generator.eval()

# # Specify the label options
# class_labels = ['garlic_bread', 'chicken_wings', 'frozen_yogurt', 'dumplings', 'french_fries', 'cheese_cake', 'donut', 'cupcakes']

# # Display a selectbox for choosing the label
# selected_label = st.selectbox("Select a class label:", class_labels, index=0)

# # Ask user for the number of images to generate
# num_images = st.number_input("How many images do you want to generate?", min_value=1, max_value=10, value=1, step=1)

# # Display a submit button
# if st.button('Generate Images'):
#     # Generate and visualize images
#     with torch.no_grad():
#         # Generate a random noise vector for each image
#         latent_dim = 100
#         noise = torch.randn(num_images, latent_dim)

#         # Map the selected label to its corresponding index
#         label_idx = class_labels.index(selected_label)

#         # Repeat the label for the number of images to generate
#         labels = torch.tensor([label_idx] * num_images)

#         # Generate images from the random noise vectors and the specified label using the pretrained generator
#         generated_images = generator(noise, labels)

#         # Normalize the pixel values to the range [0.0, 1.0]
#         generated_images = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min())

#         # Visualize the generated images
#         for i in range(num_images):
#             # Set the width of the image to 300 pixels
#             st.image(generated_images[i].permute(1, 2, 0).cpu().numpy(), caption=f"{selected_label} - Image {i+1}", width=300)


import io
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Define your Generator class
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

# Define the path to the pretrained generator model file
generator_model_path = 'C:\\Users\\Lenovo\\Downloads\\GansDeployment\\Assets\\epoch_1367.pt'

# Instantiate the Generator
generator = Generator()

# Load the pretrained model weights and map them to CPU
generator.load_state_dict(torch.load(generator_model_path, map_location=torch.device('cpu')))

# Set the generator to evaluation mode
generator.eval()

# Specify the label options
class_labels = ['garlic_bread', 'chicken_wings', 'frozen_yogurt', 'dumplings', 'french_fries', 'cheese_cake', 'donut', 'cupcakes']

# Display a selectbox for choosing the label
selected_label = st.selectbox("Select a class label:", class_labels, index=0)

# Ask user for the number of images to generate
num_images = st.number_input("How many images do you want to generate?", min_value=1, max_value=10, value=1, step=1)

# Image augmentation options
augmentation_options = ['Rotation', 'Horizontal Flip', 'Vertical Flip', 'Crop', 'Scale', 'Brightness/Contrast', 'None']

# Display image augmentation options
selected_augmentation = st.selectbox('Select Image Augmentation Option:', augmentation_options)


# Function to perform image augmentation

def augment_image(image, augmentation):
    # Convert PIL image to torch tensor
    image = TF.to_pil_image(image)

    # Apply selected augmentation
    if augmentation == 'Rotation':
        # Rotate the image by 360 degrees
        augmented_image = TF.rotate(image, 360)
    elif augmentation == 'Horizontal Flip':
        # Flip the image horizontally by 180 degrees
        augmented_image = TF.rotate(image, 180)
    elif augmentation == 'Vertical Flip':
        # Flip the image vertically by 180 degrees
        augmented_image = TF.rotate(image, 180)
    elif augmentation == 'Crop':
        # Crop the image to a fixed shape (e.g., 100x100)
        augmented_image = TF.crop(image, 10, 10, 100, 100)
    elif augmentation == 'Scale':
        # Scale the image to a specific size (e.g., 200x200)
        augmented_image = TF.resize(image, (200, 200))
    elif augmentation == 'Brightness/Contrast':
        # Increase brightness by 0.5 and contrast by 0.5
        augmented_image = TF.adjust_brightness(image, 0.5)
        augmented_image = TF.adjust_contrast(augmented_image, 0.5)
    else:
        # No augmentation selected
        augmented_image = image
    
    # Convert augmented image back to torch tensor
    augmented_image = TF.to_tensor(augmented_image)
    
    return augmented_image



if st.button('Generate Images', key="generate_button"):
    # Generate and visualize images
    with torch.no_grad():
        # Generate a random noise vector for each image
        latent_dim = 100
        noise = torch.randn(num_images, latent_dim)

        # Map the selected label to its corresponding index
        label_idx = class_labels.index(selected_label)

        # Repeat the label for the number of images to generate
        labels = torch.tensor([label_idx] * num_images)

        # Generate images from the random noise vectors and the specified label using the pretrained generator
        generated_images = generator(noise, labels)

        # Normalize the pixel values to the range [0.0, 1.0]
        generated_images = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min())

        # Perform image augmentation
        if selected_augmentation != 'None':
            augmented_images = []
            for i in range(num_images):
                augmented_image = augment_image(generated_images[i], selected_augmentation)
                augmented_images.append(augmented_image)
        else:
            augmented_images = generated_images

        # Visualize the generated and augmented images
        for i in range(num_images):
            # Set the width of the image to 300 pixels
            st.image(augmented_images[i].permute(1, 2, 0).cpu().numpy(), caption=f"{selected_label} - Image {i+1}", width=300)

    # Download button for images
    def download_images():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for idx, img in enumerate(augmented_images):
                img = img.permute(1, 2, 0).cpu().numpy()
                img_name = f"{selected_label}_image_{idx + 1}.png"
                img_bytes = io.BytesIO()
                imsave(img_bytes, img, format="PNG")
                zip_file.writestr(img_name, img_bytes.getvalue())
        
        zip_buffer.seek(0)
        st.download_button(label="Download Images", data=zip_buffer, file_name="generated_images.zip")
    
    download_images()


st.markdown(page_by_img,unsafe_allow_html=True)


with st.container():
    with st.sidebar:
        members = [
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},
            {"name": "Saksham Jain", "email": "sakshamgr8online@gmail. com", "linkedin": "https://www.linkedin.com/in/saksham-jain-59b2241a4/"},
        ]

        # Define the page title and heading
        st.markdown("<h1 style='font-size:28px'>Team Members</h1>",unsafe_allow_html=True)

        # Iterate over the list of members and display their details
        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")
# import streamlit as st
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import io

# # Set page config
# st.set_page_config(page_title="Image Colorization", layout="wide")


# # Define the colorization model
# class ColorizationNet(nn.Module):
#     def __init__(self):
#         super(ColorizationNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
#         self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.relu(self.conv3(x))
#         x = torch.sigmoid(self.conv4(x))
#         return x


# # Function to load model
# @st.cache_resource
# def load_model():
#     # Check if CUDA is available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create model instance
#     model = ColorizationNet().to(device)

#     # Load the saved weights
#     try:
#         model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
#         model.eval()  # Set to evaluation mode
#         return model, device
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, device


# # Helper functions for color manipulation
# def torch_rgb_to_hsv(rgb):
#     """
#     Convert an RGB image tensor to HSV.
#     """
#     r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
#     max_val, _ = torch.max(rgb, dim=1)
#     min_val, _ = torch.min(rgb, dim=1)
#     diff = max_val - min_val

#     # Compute H
#     h = torch.zeros_like(r)
#     mask = (max_val == r) & (g >= b)
#     h[mask] = (g[mask] - b[mask]) / diff[mask]
#     mask = (max_val == r) & (g < b)
#     h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
#     mask = max_val == g
#     h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
#     mask = max_val == b
#     h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
#     h = h / 6.0
#     h[diff == 0.0] = 0.0

#     # Compute S
#     s = torch.zeros_like(r)
#     s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]

#     # V is just max_val
#     v = max_val

#     return torch.stack([h, s, v], dim=1)


# def torch_hsv_to_rgb(hsv):
#     """
#     Convert an HSV image tensor to RGB.
#     """
#     h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
#     i = (h * 6.0).floor()
#     f = h * 6.0 - i
#     p = v * (1.0 - s)
#     q = v * (1.0 - s * f)
#     t = v * (1.0 - s * (1.0 - f))

#     i_mod = i % 6
#     r = torch.zeros_like(h)
#     g = torch.zeros_like(h)
#     b = torch.zeros_like(h)

#     r[i_mod == 0.0] = v[i_mod == 0.0]
#     g[i_mod == 0.0] = t[i_mod == 0.0]
#     b[i_mod == 0.0] = p[i_mod == 0.0]

#     r[i_mod == 1.0] = q[i_mod == 1.0]
#     g[i_mod == 1.0] = v[i_mod == 1.0]
#     b[i_mod == 1.0] = p[i_mod == 1.0]

#     r[i_mod == 2.0] = p[i_mod == 2.0]
#     g[i_mod == 2.0] = v[i_mod == 2.0]
#     b[i_mod == 2.0] = t[i_mod == 2.0]

#     r[i_mod == 3.0] = p[i_mod == 3.0]
#     g[i_mod == 3.0] = q[i_mod == 3.0]
#     b[i_mod == 3.0] = v[i_mod == 3.0]

#     r[i_mod == 4.0] = t[i_mod == 4.0]
#     g[i_mod == 4.0] = p[i_mod == 4.0]
#     b[i_mod == 4.0] = v[i_mod == 4.0]

#     r[i_mod == 5.0] = v[i_mod == 5.0]
#     g[i_mod == 5.0] = p[i_mod == 5.0]
#     b[i_mod == 5.0] = q[i_mod == 5.0]

#     return torch.stack([r, g, b], dim=1)


# def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
#     """
#     Exaggerate the colors of RGB images.
#     """
#     # Convert RGB images to HSV
#     images_hsv = torch_rgb_to_hsv(images)

#     # Increase the saturation and value components
#     images_hsv[:, 1, :, :] = torch.clamp(
#         images_hsv[:, 1, :, :] * saturation_factor, 0, 1
#     )
#     images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)

#     # Convert the modified HSV images back to RGB
#     color_exaggerated_images = torch_hsv_to_rgb(images_hsv)

#     return color_exaggerated_images


# def process_image(
#     image, model, device, saturation=1.5, value=1.2, preserve_aspect=True
# ):
#     """
#     Process an image through the colorization model.
#     """
#     # Create transform pipeline
#     if preserve_aspect:
#         # Preserve aspect ratio by resizing the short side to 256
#         original_size = image.size
#         aspect_ratio = original_size[0] / original_size[1]

#         if aspect_ratio > 1:  # width > height
#             new_size = (int(256 * aspect_ratio), 256)
#         else:  # height > width
#             new_size = (256, int(256 / aspect_ratio))

#         # Resize image
#         resized_image = image.resize(new_size, Image.LANCZOS)
#     else:
#         # Simple resize to 256x256
#         resized_image = image.resize((256, 256), Image.LANCZOS)

#     # Convert to grayscale for model input
#     gray_image = resized_image.convert("L")

#     # Create a tensor transform for the grayscale image
#     transform_gray = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     # Create a tensor transform for the original color image
#     transform_color = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     # Convert grayscale image to tensor
#     gray_tensor = transform_gray(gray_image).unsqueeze(0).to(device)  # [1, 1, H, W]

#     # Convert original image to tensor for comparison
#     original_tensor = transform_color(resized_image).unsqueeze(0)  # [1, 3, H, W]

#     # Pass through model
#     with torch.no_grad():
#         colorized_tensor = model(gray_tensor)

#         # Apply color enhancement
#         enhanced_tensor = exaggerate_colors(
#             colorized_tensor, saturation_factor=saturation, value_factor=value
#         )

#     # Convert tensors to PIL images
#     to_pil = transforms.ToPILImage()

#     original_pil = to_pil(original_tensor.squeeze(0).cpu())
#     gray_pil = gray_image  # Already a PIL image
#     colorized_pil = to_pil(enhanced_tensor.squeeze(0).cpu())

#     # Resize back to original size if needed
#     if preserve_aspect and original_size != new_size:
#         original_pil = original_pil.resize(original_size, Image.LANCZOS)
#         gray_pil = gray_pil.resize(original_size, Image.LANCZOS)
#         colorized_pil = colorized_pil.resize(original_size, Image.LANCZOS)

#     return original_pil, gray_pil, colorized_pil


# # Main Streamlit app
# def main():
#     st.title("Image Colorization with Deep Learning")
#     st.write(
#         "Upload a color image or capture one with your camera, and watch it transform!"
#     )

#     # Load model
#     model, device = load_model()

#     if model is None:
#         st.error(
#             "Failed to load model. Please make sure 'colorization_model.pth' is available in the app directory."
#         )
#         return

#     # Create tabs for different input methods
#     tab1, tab2 = st.tabs(["Upload Image", "Capture from Camera"])

#     # Settings
#     with st.sidebar:
#         st.header("Settings")
#         saturation = st.slider("Color Saturation", 1.0, 3.0, 1.5, 0.1)
#         value = st.slider("Brightness", 1.0, 3.0, 1.2, 0.1)
#         preserve_aspect = st.checkbox("Preserve Aspect Ratio", value=True)

#         st.markdown("---")
#         st.subheader("Processing Quality")
#         quality = st.radio(
#             "Image Quality", ["Standard (256px)", "High (512px)"], index=0
#         )

#         # Additional info
#         st.markdown("---")
#         st.info("Higher quality processing takes longer but produces better results.")

#     # Process quality settings
#     size = 512 if quality == "High (512px)" else 256

#     # Process uploaded image
#     with tab1:
#         uploaded_file = st.file_uploader(
#             "Choose an image...", type=["jpg", "jpeg", "png"]
#         )

#         if uploaded_file is not None:
#             try:
#                 # Read image
#                 image = Image.open(uploaded_file).convert("RGB")
#                 process_and_display(
#                     image, model, device, saturation, value, preserve_aspect, size
#                 )
#             except Exception as e:
#                 st.error(f"Error processing uploaded image: {e}")

#     # Process camera input
#     with tab2:
#         camera_image = st.camera_input("Take a picture")

#         if camera_image is not None:
#             try:
#                 # Read image
#                 image = Image.open(camera_image).convert("RGB")
#                 process_and_display(
#                     image, model, device, saturation, value, preserve_aspect, size
#                 )
#             except Exception as e:
#                 st.error(f"Error processing camera image: {e}")

#     # Add information about the project
#     with st.expander("About this Project"):
#         st.write(
#             """
#         This is a Deep Learning mini-project that uses a Convolutional Neural Network to colorize grayscale images.

#         The model architecture consists of 4 convolutional layers:
#         - First layer: 1 input channel (grayscale) to 64 channels
#         - Second layer: 64 channels to 64 channels
#         - Third layer: 64 channels to 128 channels
#         - Fourth layer: 128 channels to 3 output channels (RGB)

#         The model was trained on a dataset of color images, where it learned to predict color values from grayscale input.
#         """
#         )


# def process_and_display(image, model, device, saturation, value, preserve_aspect, size):
#     """
#     Process an image and display the results.
#     """
#     # Show processing message
#     with st.spinner("Processing image..."):
#         # Resize image based on quality setting while maintaining aspect ratio
#         if preserve_aspect:
#             aspect_ratio = image.width / image.height
#             if aspect_ratio > 1:  # width > height
#                 new_size = (int(size * aspect_ratio), size)
#             else:  # height > width
#                 new_size = (size, int(size / aspect_ratio))
#         else:
#             new_size = (size, size)

#         # Resize image for processing
#         resized_image = image.resize(new_size, Image.LANCZOS)

#         # Process image
#         original, grayscale, colorized = process_image(
#             resized_image, model, device, saturation, value, preserve_aspect=False
#         )

#     # Display images
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.subheader("Original Image")
#         st.image(original, use_column_width=True)

#     with col2:
#         st.subheader("Grayscale Image")
#         st.image(grayscale, use_column_width=True)

#     with col3:
#         st.subheader("Colorized Image")
#         st.image(colorized, use_column_width=True)

#     # Add download button for colorized image
#     buf = io.BytesIO()
#     colorized.save(buf, format="PNG")
#     byte_im = buf.getvalue()

#     st.download_button(
#         label="Download Colorized Image",
#         data=byte_im,
#         file_name="colorized_image.png",
#         mime="image/png",
#     )


# if __name__ == "__main__":
#     main()


import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Set page config
st.set_page_config(page_title="Image Colorization", layout="wide")


# Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


# Function to load model
@st.cache_resource
def load_model():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = ColorizationNet().to(device)

    # Load the saved weights
    try:
        model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
        model.eval()  # Set to evaluation mode
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device


# Helper functions for color manipulation
def torch_rgb_to_hsv(rgb):
    """
    Convert an RGB image tensor to HSV.
    """
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    max_val, _ = torch.max(rgb, dim=1)
    min_val, _ = torch.min(rgb, dim=1)
    diff = max_val - min_val

    # Compute H
    h = torch.zeros_like(r)
    mask = (max_val == r) & (g >= b)
    h[mask] = (g[mask] - b[mask]) / diff[mask]
    mask = (max_val == r) & (g < b)
    h[mask] = (g[mask] - b[mask]) / diff[mask] + 6.0
    mask = max_val == g
    h[mask] = (b[mask] - r[mask]) / diff[mask] + 2.0
    mask = max_val == b
    h[mask] = (r[mask] - g[mask]) / diff[mask] + 4.0
    h = h / 6.0
    h[diff == 0.0] = 0.0

    # Compute S
    s = torch.zeros_like(r)
    s[diff != 0.0] = diff[diff != 0.0] / max_val[diff != 0.0]

    # V is just max_val
    v = max_val

    return torch.stack([h, s, v], dim=1)


def torch_hsv_to_rgb(hsv):
    """
    Convert an HSV image tensor to RGB.
    """
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    r[i_mod == 0.0] = v[i_mod == 0.0]
    g[i_mod == 0.0] = t[i_mod == 0.0]
    b[i_mod == 0.0] = p[i_mod == 0.0]

    r[i_mod == 1.0] = q[i_mod == 1.0]
    g[i_mod == 1.0] = v[i_mod == 1.0]
    b[i_mod == 1.0] = p[i_mod == 1.0]

    r[i_mod == 2.0] = p[i_mod == 2.0]
    g[i_mod == 2.0] = v[i_mod == 2.0]
    b[i_mod == 2.0] = t[i_mod == 2.0]

    r[i_mod == 3.0] = p[i_mod == 3.0]
    g[i_mod == 3.0] = q[i_mod == 3.0]
    b[i_mod == 3.0] = v[i_mod == 3.0]

    r[i_mod == 4.0] = t[i_mod == 4.0]
    g[i_mod == 4.0] = p[i_mod == 4.0]
    b[i_mod == 4.0] = v[i_mod == 4.0]

    r[i_mod == 5.0] = v[i_mod == 5.0]
    g[i_mod == 5.0] = p[i_mod == 5.0]
    b[i_mod == 5.0] = q[i_mod == 5.0]

    return torch.stack([r, g, b], dim=1)


def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    """
    Exaggerate the colors of RGB images.
    """
    # Convert RGB images to HSV
    images_hsv = torch_rgb_to_hsv(images)

    # Increase the saturation and value components
    images_hsv[:, 1, :, :] = torch.clamp(
        images_hsv[:, 1, :, :] * saturation_factor, 0, 1
    )
    images_hsv[:, 2, :, :] = torch.clamp(images_hsv[:, 2, :, :] * value_factor, 0, 1)

    # Convert the modified HSV images back to RGB
    color_exaggerated_images = torch_hsv_to_rgb(images_hsv)

    return color_exaggerated_images


def process_image(
    image, model, device, saturation=1.5, value=1.2, preserve_aspect=True
):
    """
    Process an image through the colorization model.
    """
    # Create transform pipeline
    if preserve_aspect:
        # Preserve aspect ratio by resizing the short side to 256
        original_size = image.size
        aspect_ratio = original_size[0] / original_size[1]

        if aspect_ratio > 1:  # width > height
            new_size = (int(256 * aspect_ratio), 256)
        else:  # height > width
            new_size = (256, int(256 / aspect_ratio))

        # Resize image
        resized_image = image.resize(new_size, Image.LANCZOS)
    else:
        # Simple resize to 256x256
        resized_image = image.resize((256, 256), Image.LANCZOS)

    # Convert to grayscale for model input
    gray_image = resized_image.convert("L")

    # Create a tensor transform for the grayscale image
    transform_gray = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Create a tensor transform for the original color image
    transform_color = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Convert grayscale image to tensor
    gray_tensor = transform_gray(gray_image).unsqueeze(0).to(device)  # [1, 1, H, W]

    # Convert original image to tensor for comparison
    original_tensor = transform_color(resized_image).unsqueeze(0)  # [1, 3, H, W]

    # Pass through model
    with torch.no_grad():
        colorized_tensor = model(gray_tensor)

        # Apply color enhancement
        enhanced_tensor = exaggerate_colors(
            colorized_tensor, saturation_factor=saturation, value_factor=value
        )

    # Convert tensors to PIL images
    to_pil = transforms.ToPILImage()

    original_pil = to_pil(original_tensor.squeeze(0).cpu())
    gray_pil = gray_image  # Already a PIL image
    colorized_pil = to_pil(enhanced_tensor.squeeze(0).cpu())

    # Resize back to original size if needed
    if preserve_aspect and original_size != new_size:
        original_pil = original_pil.resize(original_size, Image.LANCZOS)
        gray_pil = gray_pil.resize(original_size, Image.LANCZOS)
        colorized_pil = colorized_pil.resize(original_size, Image.LANCZOS)

    return original_pil, gray_pil, colorized_pil


# Main Streamlit app
def main():
    st.title("Image Colorization with Deep Learning")
    st.write(
        "Upload a color image or capture one with your camera, and watch it transform!"
    )

    # Load model
    model, device = load_model()

    if model is None:
        st.error(
            "Failed to load model. Please make sure 'colorization_model.pth' is available in the app directory."
        )
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Capture from Camera"])

    # Settings
    with st.sidebar:
        st.header("Settings")
        saturation = st.slider("Color Saturation", 1.0, 3.0, 1.5, 0.1)
        value = st.slider("Brightness", 1.0, 3.0, 1.2, 0.1)
        preserve_aspect = st.checkbox("Preserve Aspect Ratio", value=True)

        st.markdown("---")
        st.subheader("Processing Quality")
        quality = st.radio(
            "Image Quality", ["Standard (256px)", "High (512px)"], index=0
        )

        # Additional info
        st.markdown("---")
        st.info("Higher quality processing takes longer but produces better results.")

    # Process quality settings
    size = 512 if quality == "High (512px)" else 256

    # Process uploaded image
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                # Read image
                image = Image.open(uploaded_file).convert("RGB")
                process_and_display(
                    image,
                    model,
                    device,
                    saturation,
                    value,
                    preserve_aspect,
                    size,
                    key="upload",
                )
            except Exception as e:
                st.error(f"Error processing uploaded image: {e}")

    # Process camera input
    with tab2:
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            try:
                # Read image
                image = Image.open(camera_image).convert("RGB")
                process_and_display(
                    image,
                    model,
                    device,
                    saturation,
                    value,
                    preserve_aspect,
                    size,
                    key="camera",
                )
            except Exception as e:
                st.error(f"Error processing camera image: {e}")

    # Add information about the project
    with st.expander("About this Project"):
        st.write(
            """
        This is a Deep Learning mini-project that uses a Convolutional Neural Network to colorize grayscale images.
        
        The model architecture consists of 4 convolutional layers:
        - First layer: 1 input channel (grayscale) to 64 channels
        - Second layer: 64 channels to 64 channels
        - Third layer: 64 channels to 128 channels
        - Fourth layer: 128 channels to 3 output channels (RGB)
        
        The model was trained on a dataset of color images, where it learned to predict color values from grayscale input.
        """
        )


def process_and_display(
    image, model, device, saturation, value, preserve_aspect, size, key=None
):
    """
    Process an image and display the results.
    """
    # Show processing message
    with st.spinner("Processing image..."):
        # Resize image based on quality setting while maintaining aspect ratio
        if preserve_aspect:
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:  # width > height
                new_size = (int(size * aspect_ratio), size)
            else:  # height > width
                new_size = (size, int(size / aspect_ratio))
        else:
            new_size = (size, size)

        # Resize image for processing
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Process image
        original, grayscale, colorized = process_image(
            resized_image, model, device, saturation, value, preserve_aspect=False
        )

    # Display images
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_column_width=True)

    with col2:
        st.subheader("Grayscale Image")
        st.image(grayscale, use_column_width=True)

    with col3:
        st.subheader("Colorized Image")
        st.image(colorized, use_column_width=True)

    # Add download button for colorized image
    buf = io.BytesIO()
    colorized.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Colorized Image",
        data=byte_im,
        file_name="colorized_image.png",
        mime="image/png",
        key=f"download_btn_{key}",  # Add unique key based on the source
    )


if __name__ == "__main__":
    main()

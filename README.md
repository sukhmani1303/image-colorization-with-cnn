# Image Colorization with Deep Learning

This Streamlit app uses a deep learning model to colorize grayscale images, allowing you to upload color images or capture them with your camera, convert them to grayscale, and then colorize them again using AI.

## Features

- Upload images or capture them with your camera
- Convert color images to grayscale
- Apply AI-based colorization to grayscale images
- Adjust color saturation and brightness
- Choose between standard and high-quality processing
- Option to preserve aspect ratio
- Download colorized images

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sukhmani1303/image-colorization-with-cnn.git
   cd image-colorization
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the pre-trained model weights file `colorization_model.pth` and place it in the project directory.

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

This will start the app and open it in your default web browser. You can then:

1. Upload an image or take a picture with your camera
2. Adjust settings in the sidebar:
   - Color Saturation: Increase or decrease color intensity
   - Brightness: Adjust the brightness of colorized images
   - Preserve Aspect Ratio: Maintain original image proportions
   - Image Quality: Choose between standard (256px) or high (512px) quality

3. View the original, grayscale, and colorized versions of your image
4. Download the colorized image

## Model Architecture

The colorization model uses a Convolutional Neural Network with the following architecture:

- Input: Grayscale image (1 channel)
- First layer: 1 input channel (grayscale) to 64 channels
- Second layer: 64 channels to 64 channels
- Third layer: 64 channels to 128 channels
- Fourth layer: 128 channels to 3 output channels (RGB)

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- Pillow
- NumPy
- torchvision

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project demonstrates the application of deep learning for image colorization using PyTorch and Streamlit.

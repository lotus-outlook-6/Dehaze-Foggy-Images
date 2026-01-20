import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

def get_dark_channel(image, size=15):
    """
    Calculates the dark channel of the image.
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel):
    """
    Estimates the atmospheric light from the image using the dark channel.
    """
    h, w = image.shape[:2]
    image_size = h * w
    num_pixels = int(max(image_size * 0.001, 1))  # Top 0.1%

    dark_vec = dark_channel.reshape(image_size)
    image_vec = image.reshape(image_size, 3)

    indices = dark_vec.argsort()[-num_pixels:]
    
    # Pick the pixel with the highest intensity among the top 0.1% brightest in dark channel
    # A simple heuristic is to take the mean of these pixels or the brightest one.
    # Here we take the mean for stability.
    atmospheric_light = np.mean(image_vec[indices], axis=0)
    return atmospheric_light

def get_transmission(image, atmospheric_light, size=15, omega=0.95):
    """
    Estimates the transmission map.
    """
    norm_image = image / atmospheric_light
    dark_channel = get_dark_channel(norm_image, size)
    transmission = 1 - omega * dark_channel
    return transmission

def refine_transmission(image, transmission, r=40, eps=1e-3):
    """
    Refines the transmission map using Guided Filter.
    """
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64) / 255.0
    
    # Guided filter implementation
    mean_I = cv2.boxFilter(gray, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(transmission, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(gray * transmission, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(gray * gray, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    refined_transmission = mean_a * gray + mean_b
    return refined_transmission

def recover_scene(image, transmission, atmospheric_light, t0=0.1):
    """
    Recovers the scene radiance.
    """
    # Clip transmission to avoid division by zero
    transmission = np.maximum(transmission, t0)
    
    transmission_3c = np.dstack([transmission] * 3)
    
    dehazed = (image - atmospheric_light) / transmission_3c + atmospheric_light
    return np.clip(dehazed, 0, 1)

def dehaze_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert to float for processing
    I = img.astype('float64') / 255.0
    
    print("Processing...")

    # 1. Dark Channel
    dark = get_dark_channel(I)
    
    # 2. Atmospheric Light
    A = get_atmospheric_light(I, dark)
    print(f"Atmospheric Light estimated: {A}")
    
    # 3. Transmission
    te = get_transmission(I, A)
    
    # 4. Refine Transmission (Optional but recommended)
    t = refine_transmission(I, te)
    
    # 5. Recover Scene
    J = recover_scene(I, t, A)

    # Convert back to uint8 for display
    J_uint8 = (J * 255).astype(np.uint8)
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    J_rgb = cv2.cvtColor(J_uint8, cv2.COLOR_BGR2RGB)

    # Display
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Foggy Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Dehazed Image")
    plt.imshow(J_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Initialize Tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()

    print("Please select an image file...")
    file_path = filedialog.askopenfilename(
        title="Select a Foggy Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if file_path:
        print(f"Selected file: {file_path}")
        dehaze_image(file_path)
    else:
        print("No file selected.")

if __name__ == "__main__":
    main()

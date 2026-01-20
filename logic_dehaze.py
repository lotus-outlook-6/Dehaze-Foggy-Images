import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def get_dark_channel(image, window_size):
    b, g, r = cv2.split(image)
    min_intensity = np.minimum(np.minimum(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_intensity, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, percentile=0.001):
    num_pixels = dark_channel.size
    num_brightest = int(max(1, num_pixels * percentile))
    
    flat_dark_channel = dark_channel.flatten()
    flat_image = image.reshape(num_pixels, 3)
    
    indices = np.argsort(flat_dark_channel)[-num_brightest:]
    brightest_pixels = flat_image[indices]
    
    atmospheric_light = np.mean(brightest_pixels, axis=0)
    
    # Create a mask for visualization
    mask = np.zeros(num_pixels, dtype=np.uint8)
    mask[indices] = 255
    mask = mask.reshape(dark_channel.shape)
    
    return atmospheric_light.astype(np.float64), mask

def guided_filter(guidance_image, input_image, radius, epsilon):
    mean_I = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_p = cv2.boxFilter(input_image, -1, (radius, radius))

    mean_II = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + epsilon)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output = mean_a * guidance_image + mean_b
    return output

def get_transmission_map(image, atmospheric_light, window_size, omega=0.95, 
                         guided_filter_radius=60, guided_filter_epsilon=0.0001):
    
    normalized_image = np.empty_like(image, dtype=np.float64)
    for i in range(3):
        normalized_image[:, :, i] = image[:, :, i] / atmospheric_light[i]
        
    transmission_estimate = get_dark_channel(normalized_image, window_size)
    transmission_map = 1 - (omega * transmission_estimate)
    
    image_normalized_for_guidance = (image / 255.0).astype(np.float32)
    image_gray_for_guidance = cv2.cvtColor(image_normalized_for_guidance, cv2.COLOR_BGR2GRAY)

    transmission_map_float32 = transmission_map.astype(np.float32)
    
    refined_transmission_map = guided_filter(image_gray_for_guidance, 
                                             transmission_map_float32, 
                                             guided_filter_radius, 
                                             guided_filter_epsilon)
    
    return transmission_map, refined_transmission_map.astype(np.float64), image_gray_for_guidance

def dehaze_image(image, atmospheric_light, transmission_map, t0=0.1):
    clamped_transmission = np.maximum(transmission_map, t0)
    
    clamped_transmission_3d = cv2.merge([clamped_transmission, clamped_transmission, clamped_transmission])

    dehazed_image = np.empty_like(image, dtype=np.float64)
    
    for i in range(3):
        dehazed_image[:, :, i] = ((image[:, :, i] - atmospheric_light[i]) / 
                                  clamped_transmission_3d[:, :, i]) + atmospheric_light[i]
                                  
    dehazed_image = np.clip(dehazed_image, 0, 255)
    return dehazed_image.astype(np.uint8)

def main():
    print("Opening file dialog to select an image...")
    root = tk.Tk()
    root.withdraw()
    
    input_image_path = filedialog.askopenfilename(
        title="Select a Hazy Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
    )
    
    if not input_image_path:
        print("No image selected. Exiting.")
        return

    window_size = 15
    
    # --- TWEAKABLE VALUES ---
    omega = 0.95
    t0 = 0.3
    percentile = 0.001
    guided_filter_radius = 60
    guided_filter_epsilon = 0.0001
    # ---------------------

    print(f"Loading image: {input_image_path}")
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        print("Please make sure the file is a valid image.")
        return

    image_float = image.astype(np.float64)
    
    print("Calculating dark channel...")
    dark_channel = get_dark_channel(image_float, window_size)
    
    print("Estimating atmospheric light...")
    a_light, a_light_mask = get_atmospheric_light(image_float, dark_channel, percentile)
    print(f"  Estimated atmospheric light (BGR): {a_light}")
    
    print("Estimating transmission map...")
    raw_transmission, transmission, guidance_image = get_transmission_map(image_float / 255.0, a_light / 255.0, window_size, omega,
                                        guided_filter_radius, guided_filter_epsilon)
    
    print("Dehazing image...")
    dehazed_image = dehaze_image(image_float, a_light, transmission, t0)
    
    print("Displaying images...")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dehazed_image_rgb = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB)

    # Figure 1: Processing Steps
    plt.figure(figsize=(18, 10))
    plt.suptitle("Processing Steps")

    # Row 1
    plt.subplot(2, 3, 1)
    plt.title("Dark Channel")
    plt.imshow(dark_channel, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Atmospheric Light Mask (Top 0.1%)")
    plt.imshow(a_light_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Raw Transmission")
    plt.imshow(raw_transmission, cmap='gray')
    plt.axis('off')

    # Row 2
    plt.subplot(2, 3, 4)
    plt.title("Guidance Image (Gray)")
    plt.imshow(guidance_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Refined Transmission")
    plt.imshow(transmission, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()

    # Figure 2: Final Result
    plt.figure(figsize=(12, 6))
    plt.suptitle("Final Result")

    plt.subplot(1, 2, 1)
    plt.title("Original Foggy Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Dehazed Image")
    plt.imshow(dehazed_image_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    print("Done.")

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import sys
# --- NEW IMPORTS ---
import tkinter as tk
from tkinter import filedialog
# --- MATPLOTLIB IMPORT ---
import matplotlib.pyplot as plt
# --- END NEW IMPORTS ---

def get_dark_channel(image, window_size):
    """
    Calculates the Dark Channel of an image.
    The dark channel is a concept based on the observation that in most non-sky
    patches, at least one color channel has some pixels whose intensity is very low.
    
    Args:
        image (numpy.ndarray): The input image (should be in BGR format).
        window_size (int): The size of the window (kernel) for the minimum filter.
    
    Returns:
        numpy.ndarray: The dark channel of the image.
    """
    # Split the image into its B, G, R components
    b, g, r = cv2.split(image)
    
    # Find the minimum intensity in each pixel across all color channels
    min_intensity = np.minimum(np.minimum(b, g), r)
    
    # Define the kernel for the minimum filter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    
    # Apply the minimum filter to the min_intensity image
    dark_channel = cv2.erode(min_intensity, kernel)
    
    return dark_channel

def get_atmospheric_light(image, dark_channel, percentile=0.001):
    """
    Estimates the atmospheric light (A) from the image.
    This is done by finding the brightest pixels in the dark channel and
    selecting the corresponding pixel in the original image with the highest intensity.
    
    Args:
        image (numpy.ndarray): The original BGR image.
        dark_channel (numpy.ndarray): The dark channel of the image.
        percentile (float): The percentage of brightest pixels to consider.
    
    Returns:
        list: The estimated atmospheric light [B, G, R].
    """
    # Get the total number of pixels
    num_pixels = dark_channel.size
    
    # Calculate the number of pixels to consider based on the percentile
    num_brightest = int(max(1, num_pixels * percentile))
    
    # Reshape the dark channel to a 1D array
    flat_dark_channel = dark_channel.flatten()
    
    # Reshape the original image to a 2D array (num_pixels x 3 channels)
    flat_image = image.reshape(num_pixels, 3)
    
    # Get the indices of the pixels with the highest intensities in the dark channel
    indices = np.argsort(flat_dark_channel)[-num_brightest:]
    
    # Find the pixel in the original image that has the highest intensity among these candidates
    brightest_pixels = flat_image[indices]
    
    # Average the intensities of these brightest pixels to get a more stable estimate
    atmospheric_light = np.mean(brightest_pixels, axis=0)
    
    return atmospheric_light.astype(np.float64)

# --- NEW FUNCTION FOR GUIDED FILTER ---
def guided_filter(guidance_image, input_image, radius, epsilon):
    """
    Applies a guided filter to the input_image, guided by the guidance_image.
    This is used to refine the transmission map to reduce halos.
    
    Args:
        guidance_image (numpy.ndarray): The image to guide the filter (usually the original image).
                                        Should be normalized to [0, 1] and in float type.
        input_image (numpy.ndarray): The image to be filtered (e.g., the transmission map).
                                     Should be normalized to [0, 1] and in float type.
        radius (int): The radius of the guiding window.
        epsilon (float): Regularization parameter.
    
    Returns:
        numpy.ndarray: The filtered output image.
    """
    # --- CHANGE 1: REMOVED THIS BLOCK ---
    # The guided_image should be 1-channel (grayscale) for this implementation,
    # so we no longer convert it to 3-channel.
    #
    # if guidance_image.ndim == 2:
    #     guidance_image = cv2.cvtColor(guidance_image, cv2.COLOR_GRAY2BGR)
    # --- END OF CHANGE 1 ---

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
                         guided_filter_radius=60, guided_filter_epsilon=0.0001): # Added guided filter params
    """
    Estimates the transmission map (t(x)).
    The transmission map represents the portion of light that is not scattered
    and reaches the camera.
    
    Args:
        image (numpy.ndarray): The original BGR image.
        atmospheric_light (list): The estimated atmospheric light [B, G, R].
        window_size (int): The size of the window (kernel) for the minimum filter.
        omega (float): A constant factor (between 0 and 1) to keep a small
                       amount of haze for a more natural look.
        guided_filter_radius (int): Radius for the guided filter.
        guided_filter_epsilon (float): Epsilon for the guided filter.
    
    Returns:
        numpy.ndarray: The estimated transmission map.
    """
    # Normalize the image by the atmospheric light
    # We do this for each channel
    normalized_image = np.empty_like(image, dtype=np.float64)
    for i in range(3):
        normalized_image[:, :, i] = image[:, :, i] / atmospheric_light[i]
        
    # Get the dark channel of the normalized image
    # This gives us an estimate of 1 - t(x)
    transmission_estimate = get_dark_channel(normalized_image, window_size)
    
    # Calculate the transmission map: t(x) = 1 - omega * (dark channel of normalized image)
    transmission_map = 1 - (omega * transmission_estimate)
    
    # --- THIS IS THE NEW PART: REFINE WITH GUIDED FILTER ---
    # --- CHANGE 2: Convert guidance image to GRAYSCALE ---
    # Convert original image to float32, normalize, and convert to GRAYSCALE for guidance
    image_normalized_for_guidance = (image / 255.0).astype(np.float32)
    image_gray_for_guidance = cv2.cvtColor(image_normalized_for_guidance, cv2.COLOR_BGR2GRAY)
    # --- END OF CHANGE 2 ---

    # Convert transmission map to float32 for guided filter input
    transmission_map_float32 = transmission_map.astype(np.float32)
    
    # Apply the guided filter (using the new grayscale guide)
    refined_transmission_map = guided_filter(image_gray_for_guidance, 
                                             transmission_map_float32, 
                                             guided_filter_radius, 
                                             guided_filter_epsilon)
    
    # Ensure output is float64 as expected by later functions
    return refined_transmission_map.astype(np.float64)
    # --- END NEW PART ---

def dehaze_image(image, atmospheric_light, transmission_map, t0=0.1):
    """
    Recovers the dehazed image (J(x)) using the atmospheric scattering model:
    J(x) = (I(x) - A) / max(t(x), t0) + A
    
    Args:
        image (numpy.ndarray): The original (hazy) BGR image.
        atmospheric_light (list): The estimated atmospheric light [B, G, R].
        transmission_map (numpy.ndarray): The estimated transmission map.
        t0 (float): A minimum threshold for the transmission map to avoid
                    division by zero or excessively bright pixels.
    
    Returns:
        numpy.ndarray: The dehazed BGR image.
    """
    # Ensure transmission_map has a minimum value of t0
    clamped_transmission = np.maximum(transmission_map, t0)
    
    # Expand transmission_map to 3 channels (to match the image)
    # We use cv2.merge to combine three copies of the 1-channel map
    clamped_transmission_3d = cv2.merge([clamped_transmission, clamped_transmission, clamped_transmission])

    # Allocate memory for the dehazed image
    dehazed_image = np.empty_like(image, dtype=np.float64)
    
    # Apply the dehazing formula for each color channel
    for i in range(3):
        dehazed_image[:, :, i] = ((image[:, :, i] - atmospheric_light[i]) / 
                                  clamped_transmission_3d[:, :, i]) + atmospheric_light[i]
                                  
    # Clip the values to the valid range [0, 255]
    dehazed_image = np.clip(dehazed_image, 0, 255)
    
    # Convert back to uint8
    return dehazed_image.astype(np.uint8)

def main():
    """
    Main function to run the dehazing process.
    """
    # --- Configuration ---
    # --- MODIFICATION: Use Tkinter to ask for input file ---
    print("Opening file dialog to select an image...")
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    input_image_path = filedialog.askopenfilename(
        title="Select a Hazy Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
    )
    
    if not input_image_path:
        print("No image selected. Exiting.")
        return
    # --- END MODIFICATION ---

    output_image_path = 'dehazed_image.jpg' # As requested
    window_size = 15      # Kernel size for dark channel and transmission map
    
    # --- TWEAKABLE VALUES ---
    omega = 0.95          # Haze-keeping factor (0.0 to 1.0)
    t0 = 0.3              # Minimum transmission threshold (0.0 to 1.0)
    percentile = 0.001    # Percentile for atmospheric light
    
    # --- NEW GUIDED FILTER PARAMETERS ---
    # radius: Larger radius makes the smoothing effect broader.
    # epsilon: Controls edge preservation. Smaller epsilon means sharper edges (less smoothing)
    #          but might leave more halos. Larger epsilon means more smoothing.
    guided_filter_radius = 60 # Try 30-80. Larger = more smoothing.
    guided_filter_epsilon = 0.0001 # Try 0.0001 - 0.01. Smaller = sharper edges.
    # ---------------------

    print(f"Loading image: {input_image_path}")
    
    # Load the image
    image = cv2.imread(input_image_path)
    
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        # --- MODIFICATION: Updated error message ---
        print("Please make sure the file is a valid image.")
        # --- END MODIFICATION ---
        return

    # Convert image to float64 for calculations
    image_float = image.astype(np.float64)
    
    print("Calculating dark channel...")
    dark_channel = get_dark_channel(image_float, window_size)
    
    print("Estimating atmospheric light...")
    a_light = get_atmospheric_light(image_float, dark_channel, percentile)
    print(f"  Estimated atmospheric light (BGR): {a_light}")
    
    print("Estimating transmission map...")
    # Convert image_float to [0, 1] range for transmission calculation, and pass guided filter params
    transmission = get_transmission_map(image_float / 255.0, a_light / 255.0, window_size, omega,
                                        guided_filter_radius, guided_filter_epsilon)
    
    print("Dehazing image...")
    dehazed_image = dehaze_image(image_float, a_light, transmission, t0)
    
    print(f"Saving dehazed image to: {output_image_path}")
    cv2.imwrite(output_image_path, dehazed_image)
    
    print("Displaying images...")

    # --- MODIFICATION: Display images using Matplotlib ---
    
    # Convert images from BGR (OpenCV format) to RGB (Matplotlib format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dehazed_image_rgb = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    ax1.imshow(image_rgb)
    ax1.set_title('Original')
    ax1.axis('off') # Hide axes

    # Display the dehazed image
    ax2.imshow(dehazed_image_rgb)
    ax2.set_title('Dehazed')
    ax2.axis('off') # Hide axes

    plt.tight_layout() # Adjust layout to prevent title overlap
    plt.show() # Show the plot window

    # --- END MODIFICATION ---

    # --- REMOVED OLD DISPLAY CODE ---
    # print("Displaying images... Press any key to close.")
    
    # # Create a side-by-side comparison
    # h, w, _ = image.shape
    # comparison = np.concatenate((image, dehazed_image), axis=1)
    
    # # Resize if too large to fit on screen
    # max_width = 1600
    # if comparison.shape[1] > max_width:
    #     scale = max_width / comparison.shape[1]
    #     comparison = cv2.resize(comparison, (0, 0), fx=scale, fy=scale)

    # cv2.imshow('Original (Left)  |  Dehazed (Right)', comparison)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --- END REMOVED CODE ---
    
    print("Done.")

if __name__ == '__main__':
    main()
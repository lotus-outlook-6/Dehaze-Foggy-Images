# Image Dehazing Tool

This project implements a Single Image Haze Removal algorithm using the **Dark Channel Prior** method. It effectively removes haze from images to recover the clear scene radiance.

The core logic is implemented in `logic_dehaze.py`, which not only dehazes the image but also visualizes every stage of the process, helping you understand how the algorithm treats the input image.

## How It Works

The dehazing process involves several key mathematical and image processing steps:

### 1. Dark Channel Prior Estimation
The algorithm starts by calculating the **Dark Channel** of the image. The Dark Channel Prior suggests that in most non-sky patches of a haze-free outdoor image, at least one color channel has very low intensity at some pixels.
- **Visual Output**: You will see a dark, grayscale version of the image where hazy areas appear brighter.

### 2. Atmospheric Light Estimation
Using the Dark Channel, the algorithm estimates the **Atmospheric Light** (the ambient light color). It picks the top 0.1% brightest pixels in the dark channel and uses the corresponding pixels in the original image to estimate the atmospheric light.
- **Visual Output**: A mask showing the pixels selected for this estimation is displayed.

### 3. Transmission Map Estimation
The **Transmission Map** describes the portion of the light that is not scattered and reaches the camera. It is derived from the Dark Channel.
- **Visual Output**: A "Raw Transmission" map is generated, showing a rough estimate of haze density (darker means more haze/less transmission).

### 4. Transmission Refinement (Guided Filter)
The raw transmission map often has blocky artifacts. We use a **Guided Filter** to refine this map using the original image as a guide. This preserves edges and details while smoothing the map.
- **Visual Output**: A "Refined Transmission" map that looks much smoother and aligns with image edges.

### 5. Scene Recovery
Finally, the algorithm recovers the scene radiance using the refined transmission map and the estimated atmospheric light, following the atmospheric scattering model.
- **Result**: The final dehazed image with restored colors and contrast.

## Visualization

When you run `logic_dehaze.py`, it calculates these stages and presents them in two specific figures:

**Figure 1: Processing Steps**
This figure displays the intermediate outputs:
1.  **Dark Channel**
2.  **Atmospheric Light Mask**
3.  **Raw Transmission**
4.  **Guidance Image**
5.  **Refined Transmission**

**Figure 2: Final Result**
This figure shows a side-by-side comparison:
-   **Original Foggy Image**
-   **Dehazed Image**

*(A sample of this output would show the progression from a gray/foggy analysis to a clear, high-contrast final image)*

## How to Run

Follow these steps to run the dehazing tool on your local machine.

### Prerequisites

You need Python installed along with the following libraries:
-   `opencv-python` (cv2)
-   `numpy`
-   `matplotlib`
-   `tkinter` (usually included with Python)

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-python numpy matplotlib
```

### Running the Script

1.  Navigate to the project directory:
    ```bash
    cd "dehaze foggy images"
    ```

2.  Run the logic script:
    ```bash
    python logic_dehaze.py
    ```

3.  A file dialog will open. Select a hazy image (supported formats: JPG, PNG, BMP) from your computer (you can find samples in the `Sample Images` folder).

4.  The script will process the image and open two windows showing the processing stages and the final result.

---
*Note: `dehaze.py` is an alternative script in the folder, but `logic_dehaze.py` provides the detailed visualization of the dehazing stages.*

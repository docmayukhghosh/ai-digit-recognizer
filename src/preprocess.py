import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_image_for_inference(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocesses a raw input image (RGB/BGR/Grayscale, any size) to match MNIST format:
    - Grayscale conversion
    - Noise removal via Gaussian Blur
    - Thresholding (Otsu's binarization)
    - Auto-inversion to ensure white digit on black background
    - Contour detection to crop the bounding box of the digit
    - Resize digit to fit 20x20 pixels while preserving aspect ratio
    - Center the 20x20 digit inside a 28x28 black canvas
    - Normalize pixel values to [0.0, 1.0]
    
    Args:
        image (np.ndarray): The input image array (height, width, channels) or (height, width).
        debug (bool): If True, returns a visualization image of the steps.
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]:
            - preprocessed_image: A (28, 28, 1) float32 numpy array.
            - debug_image: A visualization of the preprocessing steps (or None if debug=False).
    """
    # 1. Convert to grayscale if it has multiple channels
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            # Convert alpha channel transparency to white background
            alpha = image[:, :, 3] / 255.0
            image_rgb = image[:, :, :3]
            white_bg = np.ones_like(image_rgb) * 255
            image = (image_rgb * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image.copy()
        
    # Save original grayscale for debugging visualization
    orig_gray = image_gray.copy()

    # 2. Noise reduction using Gaussian Blur
    # Kernels size (5,5) removes small grains, paper texture, and sensor noise.
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # 3. Binarization (Otsu's thresholding)
    # Automatically calculates optimal threshold value separating foreground and background.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Auto-inversion
    # We want a white digit on a black background.
    # Check the borders of the image (first/last row, first/last col).
    # If the average border value is > 127 (closer to white), the background is white, so we invert.
    border_pixels = np.concatenate([
        thresh[0, :],          # Top border
        thresh[-1, :],         # Bottom border
        thresh[:, 0],          # Left border
        thresh[:, -1]          # Right border
    ])
    
    is_inverted = False
    if np.mean(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)
        is_inverted = True

    # 5. Find contours of the digit
    # Contours are curves joining all the continuous points along a boundary.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    
    if len(contours) > 0:
        # Assume the largest contour is the digit
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the digit
        digit_crop = thresh[y:y+h, x:x+w]
        
        # 6. Resize keeping aspect ratio
        # MNIST digits are centered inside a 20x20 bounding box.
        if w > h:
            # Width is the larger dimension
            new_w = 20
            new_h = int(h * (20.0 / w))
            if new_h < 1: new_h = 1
        else:
            # Height is the larger dimension
            new_h = 20
            new_w = int(w * (20.0 / h))
            if new_w < 1: new_w = 1
            
        digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 7. Center the resized digit on the 28x28 canvas
        # Calculate padding to center it
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_resized
    else:
        # If no contours found (empty image), resize whole thresh to 20x20 and center it
        digit_resized = cv2.resize(thresh, (20, 20), interpolation=cv2.INTER_AREA)
        canvas[4:24, 4:24] = digit_resized

    # 8. Normalize pixel values to [0.0, 1.0] and add channel dimension
    preprocessed_image = canvas.astype(np.float32) / 255.0
    preprocessed_image = np.expand_dims(preprocessed_image, axis=-1)  # Shape: (28, 28, 1)

    debug_image = None
    if debug:
        # Create a visual grid of the steps
        # Resize intermediate steps for side-by-side visualization
        h_orig, w_orig = image_gray.shape[:2]
        size = 200
        vis_orig = cv2.resize(orig_gray, (size, size))
        vis_thresh = cv2.resize(thresh, (size, size))
        
        # Draw bounding box on the original grayscale image if contours exist
        vis_contour = cv2.cvtColor(vis_orig.copy(), cv2.COLOR_GRAY2RGB)
        if len(contours) > 0:
            # Map coordinates back to visual size
            rx = int(x * size / w_orig)
            ry = int(y * size / h_orig)
            rw = int(w * size / w_orig)
            rh = int(h * size / h_orig)
            cv2.rectangle(vis_contour, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            
        vis_canvas = cv2.resize(canvas, (size, size))
        vis_canvas = cv2.cvtColor(vis_canvas, cv2.COLOR_GRAY2RGB)
        
        # Combine visual panels
        panel1 = cv2.putText(cv2.cvtColor(vis_orig, cv2.COLOR_GRAY2RGB), "1. Grayscale Input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        panel2 = cv2.putText(cv2.cvtColor(vis_thresh, cv2.COLOR_GRAY2RGB), "2. Thresholded", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        panel3 = cv2.putText(vis_contour, "3. Bounding Box", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        panel4 = cv2.putText(vis_canvas, "4. Resized & Centered", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Stack horizontal
        debug_image = np.hstack([panel1, panel2, panel3, panel4])

    return preprocessed_image, debug_image


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Applies optional random augmentation to a preprocessed image (28, 28, 1) to simulate 
    real-world variations:
    - Slight rotation (up to 10 degrees)
    - Translation/Shifting (up to 2 pixels)
    - Slight scaling/zoom (90% to 110%)
    - Gaussian noise addition
    
    Args:
        image (np.ndarray): (28, 28, 1) float32 image.
        
    Returns:
        np.ndarray: Augmented (28, 28, 1) image.
    """
    img = (image * 255.0).astype(np.uint8).squeeze()
    h, w = img.shape
    
    # 1. Rotation & Scaling
    angle = np.random.uniform(-10.0, 10.0)
    scale = np.random.uniform(0.9, 1.1)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 2. Translation (Shifting)
    tx = np.random.uniform(-2.0, 2.0)
    ty = np.random.uniform(-2.0, 2.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    augmented = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 3. Add light Gaussian noise
    noise = np.random.normal(0, 5, augmented.shape).astype(np.int16)
    augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Normalize back
    return np.expand_dims(augmented.astype(np.float32) / 255.0, axis=-1)

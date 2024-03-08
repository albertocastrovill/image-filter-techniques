"""cvlib.py"""

# Import standard libraries
import cv2
import numpy as np
import sys
from numpy.typing import NDArray


def load_image(filename:str)->NDArray:
    """
    Load an image from the specified filename using OpenCV.

        Parameters:
            filename (str): The path to the image file to be loaded.

        Returns:
            img (cv2 or None): The loaded image represented as a cv2 image.
      
    This function uses OpenCV's imread function to read the image file.
    If the image is not loaded successfully, the function prints an error 
    message to the console and exits.)
    """

    # Modify the following lines so that 'img' holds the input image
    img = cv2.imread(filename)

    # Add below the required lines of codes to check whether or not 'img'
    # was loaded correctly. If it was not, your script should print 
    # an error message on the Windows/Linux terminal and should finish.
    if img is None:
        sys.exit(f"\nERROR! - the image {filename} could not be read \n")   
    
    return img


def apply_rotation(img:NDArray)->NDArray:

    # Modify the following lines so that 'img_rotated' is a copy of 'img'
    # rotated by 45deg
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
        
    # Return rotated image
    return img_rotated


def apply_translation(img:NDArray)->NDArray:

    # Modify the following lines so that 'img_translated' is a copy of 'img'
    # translated 50 pixels to the right.
    rows, cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])  
    img_translated = cv2.warpAffine(img, translation_matrix, (cols, rows))
    
    # Return translated image
    return img_translated


def apply_reflection(img:NDArray)->NDArray:

    # Modify the following lines so that 'img_reflected' is a copy of 'img'
    # reflected vertically.
    img_reflected =  cv2.flip(img, 1)  

    # Return reflected image
    return img_reflected


def apply_multiple_transformations(img:NDArray)->NDArray:
    
    # Apply rotation transformation to input image
    img = apply_rotation(img)
    
    # Apply translation to input image
    img = apply_translation(img)

    # Apply vertical reflection to input image
    img = apply_reflection(img)

    return img



def visualise_image(img:NDArray, title:str)->None:

    # Create a new window for visualisation purposes
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Visualise image
    cv2.imshow(title, img)

def close_windows()->None:

    # Wait for the user to press any key
    cv2.waitKey(0)

    # Destroy all windows freeing memory
    cv2.destroyAllWindows()
#Import standard libraries
import numpy as np
import cv2
import sys
import argparse
import cvlib as cvl


def parse_user_data():
    """
    Parse user data entered by the user.

        Returns:
            args (argparse.Namespace): Parsed data entered by the user.
    """

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Apply image '
                                     'filtering')
    
    # Add arguments
    parser.add_argument('-i','--input_image', 
                        type=str, 
                        required=True, 
                        help='Input image to be filtered')
    
    parser.add_argument('-f','--filter_name',
                        type=str,
                        required=False,
                        help="Filter name used as Kernel"
                        " [average, gaussian, median, bilateral]")
    args = parser.parse_args()
    
    # Return parsed data entered by the user
    return args

# Apply median filter
def apply_median_filter(img:cv2):
    """
    Apply median filter to the input image.

        Parameters:
            img (cv2): The input image represented as a cv2 image.

        Returns:
            img_filtered (cv2): The filtered image represented as a cv2 image.
    """

    img_filtered = cv2.medianBlur(img, 9)  # 9x9 kernel size

    return img_filtered


# Apply average filter
def apply_average_filter(img:cv2):
    """
    Apply average filter to the input image.

        Parameters:
            img (cv2): The input image represented as a cv2 image.

        Returns:
            img_filtered (cv2): The filtered image represented as a cv2 image.
    """

    img_filtered = cv2.blur(img, (5,5))  # 5x5 kernel size

    return img_filtered


# Apply gaussian filter
def apply_gaussian_filter(img):
    """
    Apply gaussian filter to the input image.

        Parameters:
            img (cv2): The input image represented as a cv2 image.

        Returns:
            img_filtered (cv2): The filtered image represented as a cv2 image.
    """

    
    img_filtered = cv2.GaussianBlur(img, (5,5), 0)  # image , kernel size, sigma

    return img_filtered

def run_pipeline(args:argparse.Namespace)->None:

    # Load image
    img = cvl.load_image(args.input_image)

    print("Image shape: ", img.shape)

    # Apply median filter
    if args.filter_name == "median":
        img_filtered = apply_median_filter(img)

    # Apply average filter
    elif args.filter_name == "average":
        img_filtered = apply_average_filter(img)

    # Apply gaussian filter
    elif args.filter_name == "gaussian":
        img_filtered = apply_gaussian_filter(img)

    # Visualise image
    cvl.visualise_image(img_filtered, "Filtered image")

    # Close windows
    cvl.close_windows()


if __name__ == "__main__":
    
    #Ask the user to enter the input image name
    args = parse_user_data()

    #Run pipeline
    run_pipeline(args= args)

from streetview import get_panorama
import cv2
import numpy as np
from PIL import Image
import os
import math

class NormalStreetViewGenor:
    def __init__(self):
        pass

    def _auto_crop_black_edges(self, 
                              image:Image.Image, 
                              threshold=10) -> Image.Image:
        """
        Automatically crop black edges from an image.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the cropped image. If None, overwrites original.
            threshold (int): Threshold for what is considered "black" (0-255)
        
        Returns:
            numpy.ndarray: Cropped image array
        """
        
        # Convert PIL Image to cv2 format
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError("Could not convert image")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create a binary mask where non-black pixels are True
        mask = gray > threshold
        
        # Find coordinates of non-black pixels
        coords = np.argwhere(mask)
        
        if len(coords) == 0:
            print("Warning: Image appears to be entirely black")
            return img
        
        # Get bounding box of non-black content
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop the image
        cropped = img[y_min:y_max+1, x_min:x_max+1]
        
        # Convert back to PIL Image
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_pil = Image.fromarray(cropped_rgb)
        
        return cropped_pil


    def _acquire_panorama(self, pano_id:str) -> Image.Image:
        image = get_panorama(pano_id=pano_id,
        zoom=1)
        return image
    
    def __call__(self, pano_id:str) -> Image.Image:
        """
        Acquire a panorama image by its ID and crop black edges.
        
        Args:
            pano_id (str): Panorama ID to fetch the image.
        
        Returns:
            Image.Image: Cropped panorama image.
        """
        img = self._acquire_panorama(pano_id)
        if img is None:
            raise ValueError("Failed to acquire panorama image")
        
        cropped_img = self._auto_crop_black_edges(img)
        return cropped_img

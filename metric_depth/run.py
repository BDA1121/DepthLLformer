import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from .depth_anything_v2.dpt import DepthAnythingV2

def load_depth_model(encoder='vitl', load_from='metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', max_depth=80):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  
  model_configs = {
      'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
      'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
      'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
      'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
  }
  
  depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
  depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'))
  depth_anything = depth_anything.to(DEVICE).eval()
  
  return depth_anything

def depth_estimation(raw_image, model, input_size=518, pred_only=False, grayscale=False):
  cmap = matplotlib.colormaps.get_cmap('Spectral')

  depth = model.infer_image(raw_image, input_size=input_size)
      
  depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
  depth = depth.astype(np.uint8)

  depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
  return depth

# Example usage
# Load the model once
# depth_model = load_depth_model()

# Use the model for depth estimation
# Assuming `raw_image` is your input image
# depth_map = depth_estimation(raw_image, depth_model)
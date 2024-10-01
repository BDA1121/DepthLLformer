## Modified By- Harshil Bhojwani and Dhanush Adithya

import cv2
import glob
import matplotlib
import numpy as np
import os
import warnings
import torch


from .depth_anything_v2.dpt import DepthAnythingV2
# from depth_anything_v2.dpt import DepthAnythingV2
# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

def load_model(encoder='vitl', load_from='metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth', max_depth=80):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  print(f'loading{encoder}')
  model_configs = {
      'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
      'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
      'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
      'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
  }
  
  depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
  depth_anything.load_state_dict(torch.load(load_from, map_location=DEVICE))
  depth_anything = depth_anything.to(DEVICE).eval()
  
  return depth_anything
def depth_estimation(x, model,input_size= 128, pred_only=False, grayscale=False):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  
  # Ensure x is on the correct device
  x = x.to(DEVICE)
  
  # Check if x is a batch
  if x.dim() != 4:
      raise ValueError("Input tensor must have 4 dimensions (batch_size, channels, height, width)")
  
  batch_size, channels, height, width = x.shape
  
  # Convert to numpy and change to uint8 if necessary
  x_numpy = x.permute(0, 2, 3, 1).cpu().numpy()
  if x_numpy.dtype != np.uint8:
      x_numpy = (x_numpy * 255).astype(np.uint8)
  
  
  # Process each image in the batch
  depth_list = []
  for i in range(batch_size):
      depth = model.infer_image(x_numpy[i], input_size)
      depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
      depth = depth.astype(np.uint8)
      
      depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #   cv2.imwrite('testttt.png',depth)
      
      depth_list.append(depth)
  
  # Stack the processed depths back into a batch
  depth_batch = np.stack(depth_list, axis=0)
  
  # Convert back to tensor
  depth_tensor = torch.from_numpy(depth_batch).permute(0, 3, 1, 2).float().to(DEVICE)
  
  # Normalize to [0, 1] if necessary
  if depth_tensor.dtype == torch.uint8:
      depth_tensor = depth_tensor.float() / 255.0
  
  if pred_only:
      return depth_tensor
  else:
      return depth_tensor

# Example usage:
# Assuming x is your input tensor with shape [32, 3, H, W]
# model = load_model()  # Load your model
# result = depth_estimation(x, model)
# print(result.shape)  # Should be [32, 3, H, W] or [32, 1, H, W] if grayscale
        #   split_region = np.ones((raw_image.shape[1], 50, 3), dtype=np.uint8) * 255
        #   combined_result = cv2.hconcat([raw_image, split_region, depth])
        #   cv2.imwrite(output_path_image, depth)   
        #   return depth     
#   for k, filename in enumerate(filenames):
#       print(f'Progress {k+1}/{len(filenames)}: {filename}')
      
#       raw_image = cv2.imread(filename)
      
#       if raw_image is None:
#           print(f"Error: Unable to read image {filename}")
#           continue
      
      # Convert image to tensor and move to GPU if available
    #   raw_image_tensor = torch.from_numpy(raw_image).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
      
    #   depth = depth_anything.infer_image(raw_image_tensor, input_size)
      
    #   if save_numpy:
    #       output_path = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
    #       np.save(output_path, depth.cpu().numpy())
      
    #   depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    #   depth = depth.cpu().numpy().astype(np.uint8)  # Move back to CPU for further processing
      
    #   if grayscale:
    #       depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #   else:
    #       depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
      
    #   output_path = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
    #   output_path_image = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0] + (model if model else '') + '.png')
    #   if pred_only:
    #       cv2.imwrite(output_path, depth)
    #   else:
    #       split_region = np.ones((raw_image.shape[1], 50, 3), dtype=np.uint8) * 255
    #       combined_result = cv2.hconcat([raw_image, split_region, depth])
    #       cv2.imwrite(output_path_image, depth)            
          # cv2.imwrite(output_path, combined_result)

# Example usage:
# depth_estimation('path/to/image_or_directory', input_size=518, outdir='./vis_depth', encoder='vitl', load_from='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', max_depth=20, save_numpy=False, pred_only=False, grayscale=False, model='outdoorL')

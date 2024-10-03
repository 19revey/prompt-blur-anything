import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def view_image(image,results,labels=None):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, label in zip(results['bboxes'], results['labels']):
        if label == labels: 
            x1, y1, x2, y2 = bbox
            rect_box = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,
            edgecolor='r', facecolor='none')
            ax.add_patch(rect_box)
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        else:
            x1, y1, x2, y2 = bbox
            rect_box = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1,
            edgecolor='b', facecolor='none')
            ax.add_patch(rect_box)
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='blue', alpha=0.5))
    ax.axis('off')
    plt.show()

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def pixelate_region(image, masks, pixelation_size=10):
  """
  Apply a pixelation effect to the regions of the image specified by the masks.

  :param image: Original image.
  :param masks: Array of boolean masks indicating regions to pixelate.
  :param pixelation_size: Size of the pixel blocks.
  :return: Image with pixelated regions.
  """
  # Ensure the masks are boolean and have the same spatial dimensions as the image
  masks = masks.astype(bool)

  # Get the dimensions of the image
  height, width = image.shape[:2]

  # Create a copy of the image to modify
  pixelated_image = image.copy()

  # Loop over the image in blocks of pixelation_size
  for y in range(0, height, pixelation_size):
      for x in range(0, width, pixelation_size):
          # Define the block region
          block_y_end = min(y + pixelation_size, height)
          block_x_end = min(x + pixelation_size, width)
          block = image[y:block_y_end, x:block_x_end]

          # Create a combined block mask from all individual masks
          combined_block_mask = np.zeros(block.shape[:2], dtype=bool)
          for mask in masks:
              block_mask = mask[y:block_y_end, x:block_x_end]
              combined_block_mask = np.logical_or(combined_block_mask, block_mask)

          # Apply pixelation only to masked regions
          if combined_block_mask.any():
              # Compute the average color of the block
              average_color = [int(np.mean(channel[combined_block_mask])) for channel in cv2.split(block)]

              # Apply the average color to the entire block
              for c in range(3):  # For each color channel
                  block[:, :, c][combined_block_mask] = average_color[c]

              pixelated_image[y:block_y_end, x:block_x_end] = block

  return pixelated_image
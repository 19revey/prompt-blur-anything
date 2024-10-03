import numpy as np
from src.utils import DEVICE, pixelate_region
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv
import torch
from PIL import Image

class ObjectSegmenter:
    def __init__(self):

        # CHECKPOINT = "segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
        # CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
        # self.predictor = SAM2ImagePredictor(sam2_model)
        self.predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-large',
                                                             cache_dir=f"./my_models/sam2")
    def segment(self, image, boxes):

        if len(boxes) == 0:
            return image
        
        else:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

                self.predictor.set_image(image)
                masks, scores, logits = self.predictor.predict(
                    box=boxes,
                    multimask_output=False
                )            


        masks = np.squeeze(masks)
        image_array = np.array(image)
        masks_array = np.array(masks)

        if masks_array.ndim == 2:
            masks_array = np.expand_dims(masks_array, axis=0)

        # detections = sv.Detections(
        #     xyxy=sv.mask_to_xyxy(masks=masks_array),
        #     mask=masks_array.astype(bool)
        # )

        # mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        # segmented_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        return masks, image_array, masks_array
    
    def segment_and_show(self, image, boxes):

        _,_,masks_array = self.segment(image, boxes)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks_array),
            mask=masks_array.astype(bool)
        )

        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        segmented_image = mask_annotator.annotate(scene=image.copy(), detections=detections)

        return segmented_image
    

    def segment_and_pixelate(self,image,boxes,pixelation_size=10):
        masks, image_array, masks_array = self.segment(image, boxes)

        pixelated_image = pixelate_region(image_array, masks_array, pixelation_size)

        pil_image = Image.fromarray(pixelated_image)
        return pil_image
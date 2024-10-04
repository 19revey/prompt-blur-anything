from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from src.obj_detection import ObjectDetector
from src.obj_segment import ObjectSegmenter
import os
import cv2

class ModelDeplyPipeline:
    def __init__(self):

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft",
                                                    cache_dir="./my_models/Florence_2",
                                                    device_map="cuda",
                                                    trust_remote_code=True)

        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft",
                                                    cache_dir=f"./my_models/Florence_2",
                                                    trust_remote_code=True)

        self.dector = ObjectDetector(model=self.model, processor=self.processor)
        self.segmenter = ObjectSegmenter()

    def run(self, image_path,prompt=None,pixelation_size =20):

        image_path = Path(image_path)

        if os.path.isfile(image_path):
            try:
                images = [image_path]
            except:
                raise ValueError("Image not found")
        elif os.path.exists(image_path):
            images = sorted(image_path.glob("*"))
        else:
            raise ValueError("Image not found")
        
        if prompt == None:
            raise ValueError("Prompt not found")
        
        prompt = '<CAPTION_TO_PHRASE_GROUNDING>' + prompt

        for image_p in images:
            
            image = Image.open(image_p)
            if image is None:
                print(f"Could not read image {image_p}")
                continue
            

            boxes = self.dector.detect(image,prompt)

            output_path = Path("images/output_segments") / image_p.name
            segmented_image = self.segmenter.segment_and_show(image, boxes)
            segmented_image.save(output_path.as_posix())


            output_path = Path("images/output_images") / image_p.name
            segmented_image = self.segmenter.segment_and_pixelate(image, boxes,pixelation_size=pixelation_size)
            segmented_image.save(output_path.as_posix())

from src.utils import view_image, DEVICE


class ObjectDetector:
    def __init__(self, model, processor, task_type="<OD>"):
        self.model = model
        self.processor = processor
        
        self.task_type = task_type

    def detect(self, image, prompt='<OD>', labels = None, plot = False):
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            do_sample=False,
        )

        text_generations = self.processor.batch_decode(generated_ids,skip_special_tokens=False)[0]
        
        results = self.processor.post_process_generation(text_generations,task=self.task_type, image_size=(image.width, image.height))
    
        raw_lists = []

        for bbox, label in zip(results[self.task_type]['bboxes'], results[self.task_type]['labels']):
            if label == labels: 
                raw_lists.append(bbox)
            elif labels == None:
                raw_lists.append(bbox)
            else:
                raise ValueError(f"Label {labels} not found in the detected labels")
        
        
        if plot:
            view_image(image,results[self.task_type],labels)

        return raw_lists
    

    
    
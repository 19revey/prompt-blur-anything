from src.pipeline import ModelDeplyPipeline
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Deployment Pipeline')
    parser.add_argument('--image_path', type=str, help='Path to image')
    parser.add_argument('--prompt', type=str, help='Prompt for the image')
    parser.add_argument('--pixelation_size', type=int, help='Size of pixelation')
    args = parser.parse_args()

    pipeline = ModelDeplyPipeline()
    pipeline.run(args.image_path,args.prompt,args.pixelation_size)
from caricature_generator import CaricatureGenerator
import cv2
import logging
import os


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    input_path = "assets/images/photo1.jpg"
    output_path = "assets/images/caricature_result.jpg"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        print("Please place an image named 'photo1.jpg' in assets/images/ folder")
        exit(1)
    
    generator = CaricatureGenerator(device='cpu', lora_weight=0.9)
    
    image = cv2.imread(input_path)
    
    result = generator.generate(
        image=image,
        prompt="caricature style, cartoon character, exaggerated features, big head",
        strength=0.75,
        guidance_scale=7.5,
        num_inference_steps=20
    )
    
    cv2.imwrite(output_path, result)
    print(f"Result saved to {output_path}")
    
    cv2.imshow('Original', image)
    cv2.imshow('Caricature', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

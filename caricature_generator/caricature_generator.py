import numpy as np
import cv2
import torch
from typing import Optional
import logging
from PIL import Image


class CaricatureGenerator:
    
    def __init__(self, device: str = 'cpu', lora_weight: float = 0.9, logger: Optional[logging.Logger] = None):
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.lora_weight = lora_weight
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.pipe = None
        
        self.load_model()
    
    def load_model(self):
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            
            if self.logger:
                self.logger.info(f"Loading Stable Diffusion XL on {self.device}...")
            
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                use_safetensors=True
            )
            
            lora_model_id = "ProomptEngineer/pe-caricature-style"
            
            if self.logger:
                self.logger.info(f"Loading LoRA: {lora_model_id}")
            
            self.pipe.load_lora_weights(lora_model_id)
            self.pipe.fuse_lora(lora_scale=self.lora_weight)
            
            self.pipe = self.pipe.to(self.device)
            
            if self.device.type == 'cpu':
                self.pipe.enable_attention_slicing()
            else:
                self.pipe.enable_model_cpu_offload()
            
            if self.logger:
                self.logger.info("Caricature Generator ready")
                
        except ImportError:
            raise RuntimeError(
                "diffusers library not found. Install with:\n"
                "pip install diffusers transformers accelerate safetensors"
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(self, image: np.ndarray, prompt: str, 
                 strength: float = 0.75, 
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 20) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("Model not loaded")
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            original_size = pil_image.size
            target_size = (1024, 1024)
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
            
            if self.logger:
                self.logger.info(f"Generating caricature with prompt: '{prompt}'")
            
            result = self.pipe(
                prompt=prompt,
                image=pil_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
            output_pil = result.images[0]
            output_pil = output_pil.resize(original_size, Image.LANCZOS)
            
            output_rgb = np.array(output_pil)
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            
            return output_bgr
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Generation failed: {e}")
            raise
    
    def generate_from_frame(self, frame: np.ndarray, 
                            base_prompt: str = "caricature style, cartoon character",
                            strength: float = 0.75) -> np.ndarray:
        return self.generate(
            frame, 
            base_prompt,
            strength=strength,
            guidance_scale=7.0,
            num_inference_steps=15
        )


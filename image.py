from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np

# Load model on CPU
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir="./models")

pipe.to("cpu")  # Force CPU usage

# Generate image
prompt = "parent booking kids activities on a mobile app illustration"
result = pipe(prompt, num_inference_steps = 200)

# Handle different output types
if isinstance(result, dict):
    image = result["images"][0]
elif hasattr(result, "images") and not isinstance(result, tuple):
    image = result.images[0]
elif isinstance(result, tuple):
    images = result[0]
    image = images[0] if isinstance(images, (list, tuple)) else images
else:
    image = result

# Convert to PIL.Image if needed
if isinstance(image, torch.Tensor):
    image = image.detach().cpu().numpy()
if not isinstance(image, Image.Image):
    image = Image.fromarray(np.asarray(image))

# Save it
image.save("output.png")

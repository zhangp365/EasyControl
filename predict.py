# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import time
import torch
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download
import yaml
# Assuming src is in the same directory or PYTHONPATH is set correctly
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

# Define cache directory if needed, or rely on default huggingface_hub cache
MODEL_CACHE = "./checkpoints/models/"  # Local cache for LoRA

torch.set_grad_enabled(False)


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


# Helper function to clear attention cache (adapted from inference.py)
def clear_cache(transformer):
    if hasattr(transformer, "attn_processors"):
        for name, attn_processor in transformer.attn_processors.items():
            if hasattr(attn_processor, "bank_kv") and hasattr(
                attn_processor.bank_kv, "clear"
            ):
                attn_processor.bank_kv.clear()
            # Add checks for other potential cache attributes if necessary


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start_time = time.time()
        self.device = torch.device("cuda")

        # --- Model Paths ---
        base_path = "black-forest-labs/FLUX.1-dev"

        self.lora_dict = {}
        try:
            with open("loras.yaml", "r") as f:
                self.lora_dict = yaml.safe_load(f)
            print(f"all loras: {self.lora_dict}")
        except Exception as e:
            print(f"Error loading loras.yaml: {e}")
            raise

        lora_local_dir = MODEL_CACHE
        
        # --- Downlaod base model ---
        if not os.path.exists(base_path):
            # Thanks to [replicate/cog-flux](https://github.com/replicate/cog-flux)
            base_url = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
            # Already contains the FLUX.1-dev directory like `FLUX.1-dev/text_encoder_2/model.safetensors.index.json`
            download_weights(base_url, os.path.dirname(base_path))

        # --- Download LoRA ---
        print(f"Ensuring LoRA directory exists: {lora_local_dir}")
        os.makedirs(lora_local_dir, exist_ok=True)

        for lora_name, lora_info in self.lora_dict.items():
            lora_filename = lora_info["file"]
            lora_repo_id = lora_info["repo_id"]
            print(f"Downloading LoRA: {lora_filename} from {lora_repo_id}")
            try:
                hf_hub_download(
                    repo_id=lora_repo_id,
                    filename=lora_filename,
                    local_dir=lora_local_dir,
                    local_dir_use_symlinks=False,  # Recommended for Cog compatibility
                )
                print("LoRA download complete.")
                self.lora_dict[lora_name]["full_path"] = os.path.join(lora_local_dir, lora_filename)
            except Exception as e:
                print(f"Error downloading LoRA: {e}")
                raise

        # --- Load Base Model ---
        print(f"Loading base pipeline from {base_path}")
        self.pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
        print(f"Loading transformer from {base_path}")
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.pipe.transformer = transformer

        # --- Move to Device ---
        print(f"Moving pipeline to {self.device}")
        self.pipe.to(self.device)
        print("Pipeline moved to device.")

        end_time = time.time()
        print(f"Setup completed in {end_time - start_time:.2f} seconds")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Enter a text prompt to guide image generation.",
            default="",
        ),
        height: int = Input(
            description="Set the height of the generated image (pixels).",
            ge=256,
            le=1024,
            default=768,
        ),
        width: int = Input(
            description="Set the width of the generated image (pixels).",
            ge=256,
            le=1024,
            default=768,
        ),
        seed: int = Input(
            description="Set a random seed for generation (-1 for random).", default=42
        ),
        use_zero_init: bool = Input(description="Use CFG zero star.", default=True),
        zero_steps: int = Input(description="Zero init steps.", default=1, ge=0),
        spatial_img: Path = Input(description= "spatial control image."),
        control_type: str = Input(
            description="Control type (now only support two types, pose and mine_craft).",
            choices=["Ghibli", "mine_craft", "pose", "canny", "depth"],
            default="Ghibli",
        ),
        control_weight: float = Input(
            description="Control weights (must correspond to the control type).",
            default=1.0,
        ),
        extra_spatial_img: Path = Input(description= "extra spatial control image.", default=None),
        extra_control_type: str = Input(
            description="Extra control type (now only support two types, pose and mine_craft).",
            choices=["Ghibli", "mine_craft", "pose", "canny", "depth"],
            default=None,
        ),
        extra_control_weight: float = Input(
            description="Extra control weights (must correspond to the extra control type).",
            default=1.0,
        ),

        output_format: str = Input(
            description="Choose the format of the output images.",
            choices=["png", "jpg", "webp"],
            default="png",
        ),
        output_quality: int = Input(
            description="Set the quality of the output image for jpg and webp (1-100).",
            ge=1,
            le=100,
            default=90,  # Default to higher quality
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        predict_start_time = time.time()

        if extra_control_type and extra_spatial_img is None:
            print(f"Extra control type must be provided with extra spatial image, now is {extra_control_type} and {extra_spatial_img}, please check your input")
            raise ValueError(f"Extra control type must be provided with extra spatial image, now is {extra_control_type} and {extra_spatial_img}, please check your input")

        if extra_control_type and extra_control_type not in self.lora_dict:
            print(f"Extra control type '{extra_control_type}' not found in loras.yaml, can't support this control type")
            raise ValueError(f"Extra control type '{extra_control_type}' not found in loras.yaml")

        if control_type not in self.lora_dict:
            print(f"Control type '{control_type}' not found in loras.yaml, can't support this control type")
            raise ValueError(f"Control type '{control_type}' not found in loras.yaml")
        
        # --- Apply LoRA ---
        control_type_num = 0
        if control_type and not extra_control_type:
            lora_full_path = self.lora_dict[control_type]["full_path"]
            lora_weights = [control_weight] if control_weight else [1.0]
            try:
                set_single_lora(
                    self.pipe.transformer, lora_full_path, lora_weights=lora_weights, cond_size=512
                )  # Ensure weight is float
                print("LoRA applied successfully.")
                control_type_num = 1
            except Exception as e:
                print(f"Error applying LoRA: {e}")
                # Decide if this should raise or continue without LoRA
                raise
        elif control_type and extra_control_type:
            lora_full_paths = [self.lora_dict[control_type]["full_path"] for control_type in [control_type, extra_control_type]]
            lora_weights =[ w for w in [control_weight, extra_control_weight] if w]
            if len(lora_weights) < 2:
                print(f"Warning: Not enough control weights provided, add default weights [1.0]")
                lora_weights = lora_weights + [1.0] * (2 - len(lora_weights))
            try:
                set_multi_lora(
                    self.pipe.transformer, lora_full_paths, lora_weights=lora_weights, cond_size=512
                )  # Ensure weight is float
                print("Multi LoRA applied successfully.")
                control_type_num = 2
            except Exception as e:
                print(f"Error applying LoRA: {e}")
                raise

        self.pipe.to(self.device)
        print("Pipeline moved to device.")

        # --- Handle Seeds ---
        if seed == -1:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")

        # --- Load Spatial Image ---
        spatial_imgs_list = []
        spatial_imgs = [img for img in [spatial_img, extra_spatial_img] if img]
        if len(spatial_imgs) != control_type_num:
            print(f"spatial image number {len(spatial_imgs)} is not equal to control type number {control_type_num}, please check your input")
            raise ValueError(f"spatial image number {len(spatial_imgs)} is not equal to control type number {control_type_num}, please check your input")

        for spatial_img in spatial_imgs:
            try:
                loaded_spatial_img = Image.open(str(spatial_img)).convert("RGB")
                spatial_imgs_list.append(loaded_spatial_img)
                print(f"Loaded spatial image from: {spatial_img}")
            except Exception as e:
                print(f"Error loading spatial image: {e}")
                raise ValueError(
                    f"Could not load spatial image from path: {spatial_img}"
                ) from e

        output_paths = []

        image = None
        if use_zero_init:
            # --- Generate Image with use_zero_init=True ---
            print("Generating image with use_zero_init=True...")
            gen_start = time.time()
            try:
                image = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=25,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    subject_images=[],
                    spatial_images=spatial_imgs_list,
                    cond_size=512,
                    use_zero_init=True,
                    zero_steps=zero_steps,
                ).images[0]
                print(f"Generated 'zeroinit' image in {time.time() - gen_start:.2f}s")
            except Exception as e:
                print(f"Error during generation (use_zero_init=True): {e}")
            finally:
                # Clear cache regardless of success/failure if pipe was called
                clear_cache(self.pipe.transformer)
        else:
            # --- Generate Image with use_zero_init=False ---
            print("Generating image with use_zero_init=False...")
            gen_start = time.time()
            try:
                image = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=3.5,
                    num_inference_steps=25,
                    max_sequence_length=512,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    subject_images=[],
                    spatial_images=spatial_imgs_list,
                    cond_size=512,
                    use_zero_init=False,
                    # zero_steps is likely ignored here by the pipeline, but included in common args
                ).images[0]
                print(
                    f"Generated 'no zeroinit' image in {time.time() - gen_start:.2f}s"
                )
            except Exception as e:
                print(f"Error during generation (use_zero_init=False): {e}")
            finally:
                clear_cache(self.pipe.transformer)

        # --- Save Outputs ---
        save_start_time = time.time()
        output_base = f"output_seed{seed}"
        output_path = f"{output_base}.{output_format}"

        save_params = {}
        if output_format in ["jpg", "jpeg", "webp"]:
            save_params["quality"] = output_quality
        if output_format == "jpg":
            save_params["optimize"] = True  # Small optimization for JPG

        try:
            image.save(output_path, **save_params)
            output_paths.append(Path(output_path))
            print(f"Saved: {output_path}")
            print(f"Saved output in {time.time() - save_start_time:.2f}s")

        except Exception as e:
            print(f"Error saving images: {e}")
            # Remove potentially partially saved paths if error occurs during saving one of them
            if Path(output_path) in output_paths:
                output_paths.remove(Path(output_path))

        predict_end_time = time.time()
        print(
            f"Total prediction time: {predict_end_time - predict_start_time:.2f} seconds"
        )
        return output_paths

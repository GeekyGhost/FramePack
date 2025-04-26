from diffusers_helper.hf_login import login

import os
import cv2
import numpy as np
import tempfile
import subprocess
import sys
import importlib
import importlib.util

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Check if sageattention module exists
try:
    sage_spec = importlib.util.find_spec("sageattention")
    if sage_spec is None:
        print("SageAttention not found. Continuing without SageAttention...")
    else:
        print("SageAttention is already installed.")
        # Manually set SAGEATTENTION_ENABLED for diffusers_helper/models/hunyuan_video_packed.py
        # This ensures SageAttention is properly used if available
        os.environ["SAGEATTENTION_ENABLED"] = "1"
except Exception as e:
    print(f"Error checking SageAttention: {e}")
    print("Continuing without SageAttention...")

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import argparse

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Force reload the module to apply the environment variable change
import diffusers_helper.models.hunyuan_video_packed
importlib.reload(diffusers_helper.models.hunyuan_video_packed)

# Check and inform if SageAttention is actually being used
try:
    from sageattention import sageattn_varlen, sageattn
    print("SageAttention imported successfully and will be used for attention operations.")
    sage_available = True
except ImportError:
    print("SageAttention module cannot be imported. Will use other attention mechanisms.")
    sage_available = False

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')
print(f'SageAttention available: {sage_available}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


def extract_last_frame(video_path):
    """Extract the last frame from a video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video has {total_frames} frames")
        
        # If video is empty, return None
        if total_frames == 0:
            cap.release()
            print(f"Video has 0 frames: {video_path}")
            return None
        
        # Set position to the last frame
        last_frame_pos = total_frames - 1
        print(f"Seeking to frame {last_frame_pos}")
        success = cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_pos)
        if not success:
            print(f"Failed to seek to last frame, trying to read all frames")
            # If seeking fails, read all frames to get to the last one
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(total_frames - 1):
                cap.read()  # Skip frames until we reach the last one
        
        # Read the last frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Failed to read last frame from {video_path}")
            return None
            
        # Convert BGR to RGB (OpenCV uses BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"Successfully extracted last frame with shape {frame.shape}")
        
        return frame
    except Exception as e:
        print(f"Error extracting frame from video: {e}")
        traceback.print_exc()
        return None


def process_input(input_media):
    """Process either image or video input"""
    if input_media is None:
        return None
    
    try:
        # Check if input is a video file
        if isinstance(input_media, str) and input_media.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            # Extract the last frame from the video
            return extract_last_frame(input_media)
        elif isinstance(input_media, dict) and 'video' in input_media:
            video_path = input_media['video']
            # Extract the last frame from the video
            return extract_last_frame(video_path)
        elif isinstance(input_media, str) and input_media.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            # Load as image directly
            return np.array(Image.open(input_media))
        else:
            # Try to handle it as an image directly
            if hasattr(input_media, 'name') and isinstance(input_media.name, str):
                # It's likely a named file object
                if input_media.name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    return np.array(Image.open(input_media.name))
                elif input_media.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    return extract_last_frame(input_media.name)
            
            # If all else fails, try to convert it directly
            return np.array(Image.open(input_media))
    except Exception as e:
        print(f"Error processing input: {e}")
        print(f"Input type: {type(input_media)}")
        if hasattr(input_media, 'name'):
            print(f"Input name: {input_media.name}")
        traceback.print_exc()
        return None


def process_media_preview(file):
    if file is None:
        return None, gr.update(visible=False)
    
    try:
        print(f"Processing preview for file: {file}")
        print(f"File type: {type(file)}")
        
        # Check if it's a file path string
        if isinstance(file, str):
            ext = os.path.splitext(file)[1].lower()
            
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                # Process video
                frame = extract_last_frame(file)
                if frame is not None:
                    print(f"Extracted video frame with shape: {frame.shape}")
                    return file, gr.update(visible=True, value=frame)
                return None, gr.update(visible=False)
            else:
                # Process image
                try:
                    img = np.array(Image.open(file))
                    print(f"Loaded image with shape: {img.shape}")
                    return file, gr.update(visible=True, value=img)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    traceback.print_exc()
                    return None, gr.update(visible=False)
        # Handle named file objects or other types
        elif hasattr(file, 'name') and isinstance(file.name, str):
            ext = os.path.splitext(file.name)[1].lower()
            print(f"Processing file with name: {file.name} and extension: {ext}")
            
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                # Process video
                frame = extract_last_frame(file.name)
                if frame is not None:
                    return file, gr.update(visible=True, value=frame)
                return None, gr.update(visible=False)
            else:
                # Process image
                try:
                    img = np.array(Image.open(file.name))
                    return file, gr.update(visible=True, value=img)
                except Exception as e:
                    print(f"Error loading image: {e}")
                    traceback.print_exc()
                    return None, gr.update(visible=False)
        else:
            print(f"Unrecognized file type: {type(file)}")
            return None, gr.update(visible=False)
    except Exception as e:
        print(f"Error in process_media_preview: {e}")
        traceback.print_exc()
        return None, gr.update(visible=False)


@torch.no_grad()
def worker(input_media, end_media, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, 
           gpu_memory_preservation, use_teacache, mp4_crf, fps, resolution, video_quality):
    # Process input (image or video) for start frame
    input_image = process_input(input_media)
    
    if input_image is None:
        stream.output_queue.push(('error', "Failed to process input media. Please try again with a different file."))
        return
    
    # Process end frame if provided
    end_image = None
    if end_media is not None:
        end_image = process_input(end_media)
    
    has_end_image = end_image is not None
    
    total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image (start frame)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))

        # Parse resolution
        if isinstance(resolution, str):
            try:
                width, height = map(int, resolution.split('x'))
            except:
                # Fall back to calculating from image dimensions
                H, W, C = input_image.shape
                height, width = find_nearest_bucket(H, W, resolution=640)
        else:
            # Fall back to calculating from image dimensions
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)

        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_start.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        
        # Processing end image (if provided)
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
            
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            
            end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

        # Adjust CRF based on video quality if selected
        if video_quality == "high":
            mp4_crf = 15  # Lower CRF for higher quality
        elif video_quality == "medium":
            mp4_crf = 23
        elif video_quality == "low":
            mp4_crf = 30
        # If "web_compatible" or custom, use the value from the slider

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        
        end_latent = None
        if has_end_image:
            end_latent = vae_encode(end_image_pt, vae)

        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings with equal weighting
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / fps) :.2f} seconds (FPS-{fps}). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except Exception as e:
        traceback.print_exc()
        stream.output_queue.push(('error', f"An error occurred: {str(e)}"))

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_media, end_media, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, 
           gpu_memory_preservation, use_teacache, mp4_crf, fps, resolution, video_quality):
    global stream
    assert input_media is not None, 'No input media!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_media, end_media, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, 
              gpu_memory_preservation, use_teacache, mp4_crf, fps, resolution, video_quality)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'error':
            yield None, gr.update(visible=False), data, '', gr.update(interactive=True), gr.update(interactive=False)
            break

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_media = gr.File(label="Upload Start Frame (Image/Video)", file_types=["image", "video"], type="filepath")
                    start_preview_box = gr.Image(label="Start Preview", visible=False, type="numpy")
                with gr.Column():
                    end_media = gr.File(label="Upload End Frame (Optional)", file_types=["image", "video"], type="filepath")
                    end_preview_box = gr.Image(label="End Preview", visible=False, type="numpy")
            
            # Add event handlers for previewing start and end frames
            input_media.change(process_media_preview, inputs=[input_media], outputs=[input_media, start_preview_box])
            end_media.change(process_media_preview, inputs=[end_media], outputs=[end_media, end_preview_box])
            
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                
                with gr.Row():
                    with gr.Column():
                        total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                    with gr.Column():
                        fps = gr.Slider(label="FPS", minimum=10, maximum=60, value=30, step=1)

                with gr.Row():
                    resolution = gr.Dropdown(
                        label="Resolution",
                        choices=["512x512", "768x512", "512x768", "640x640", "768x768", "1024x576", "576x1024", "1024x1024"],
                        value="640x640"
                    )
                
                # Video quality radio buttons
                video_quality = gr.Radio(
                    label="Video Quality",
                    choices=["high", "medium", "low", "web_compatible"],
                    value="medium",
                    info="Higher quality means larger file size but better visual quality."
                )

                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Only applied if 'web_compatible' is selected.")

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            
            with gr.Group(visible=True):
                gr.Markdown('### Generation Notes')
                gr.Markdown('When using only a start frame, the ending actions will be generated before the starting actions due to the inverted sampling. If using both start and end frames, the model will try to create a smooth transition between them.')
                if sage_available:
                    gr.Markdown('✅ SageAttention is activated and being used for faster processing.')
            
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_media, end_media, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, 
           gpu_memory_preservation, use_teacache, mp4_crf, fps, resolution, video_quality]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


def process_media_preview(file):
    if file is None:
        return None, gr.update(visible=False)
    
    # Check file extension
    ext = os.path.splitext(file)[1].lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Process video
        frame = extract_last_frame(file)
        if frame is not None:
            return file, gr.update(visible=True, value=frame)
        return None, gr.update(visible=False)
    else:
        # Process image
        try:
            img = np.array(Image.open(file))
            return file, gr.update(visible=True, value=img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, gr.update(visible=False)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=False,  # Enable sharing
    inbrowser=args.inbrowser,
)

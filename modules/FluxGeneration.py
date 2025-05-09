import sys

if "/content/CreativeAI/totoro4" not in sys.path:
  sys.path.append("/content/CreativeAI/totoro4")

import random
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Any

# Importações locais
import totoro4.nodes
from totoro4.nodes import NODE_CLASS_MAPPINGS
from totoro4.totoro_extras import nodes_custom_sampler, nodes_flux
from totoro4.totoro import model_management

# Carregando módulos necessários
CheckpointLoaderSimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()
FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# Modelo carregado globalmente (exemplo)
with torch.inference_mode():
    unet, clip, vae = CheckpointLoaderSimple.load_checkpoint("flux1-dev-fp8-all-in-one.safetensors")


def closestNumber(n: int, m: int) -> int:
    """Calcula o número mais próximo divisível por m."""
    q = n // m
    n1 = m * q
    n2 = m * (q + 1) if (n * m) > 0 else m * (q - 1)
    return n1 if abs(n - n1) < abs(n - n2) else n2


def encode_prompt(prompt: str, guidance_scale: float):
    """Codifica o prompt usando CLIP."""
    cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    return FluxGuidance.append(cond, guidance_scale)[0]


def generate_latent(width: int, height: int):
    """Gera imagem latente com dimensões ajustadas."""
    return EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]


def sample_latents(noise, guider, sampler_name: str, scheduler: str, steps: int, latent_image):
    """Realiza a amostragem customizada."""
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
    return SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)


def decode_latents(latents):
    """Decodifica os latents em imagem final."""
    decoded = VAEDecode.decode(vae, latents)[0].detach()
    return Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])


def clear_gpu():
    """Limpa memória da GPU."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@torch.inference_mode()
def generate(
    positive_prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    steps: int = 20,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    guidance: float = 3.5,
    *args,
) -> str:
    """Função principal de geração de imagem."""
    # Gera seed aleatória se não for fornecida
    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(f"Seed: {seed}")

    # Codificação do prompt
    cond = encode_prompt(positive_prompt, guidance)

    # Geração do ruído e guia
    noise = RandomNoise.get_noise(seed)[0]
    guider = BasicGuider.get_guider(unet, cond)[0]

    # Amostragem
    latent_image = generate_latent(width, height)
    sample, _ = sample_latents(noise, guider, sampler_name, scheduler, steps, latent_image)

    # Decodificação & retorno
    return decode_latents(sample)
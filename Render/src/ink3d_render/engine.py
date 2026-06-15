from typing import Literal

import bpy


def init_render_engine(
    engine: Literal["CYCLES", "CYCLES_GPU", "BLENDER_EEVEE"], render_samples: int = 64
):
    """Initialize the rendering engine.

    Args:
        engine: The rendering engine to use.
        render_samples: Number of samples to render. Defaults to 64.
    """
    if engine == "CYCLES_GPU":
        _cycles_gpu_init(render_samples)
    elif engine == "BLENDER_EEVEE":
        _eevee_init(render_samples)
    else:
        _cycles_init(render_samples)


def _eevee_init(render_samples: int):
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = render_samples
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.render.use_high_quality_normals = True


def _cycles_init(render_samples: int):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = True


def _cycles_gpu_init(render_samples: int):
    bpy.context.scene.render.engine = "CYCLES"

    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    cycles_prefs.compute_device_type = "CUDA"
    cycles_prefs.get_devices()

    for device in cycles_prefs.devices:
        if device.type in ("CUDA", "OPTIX"):
            device.use = True
        print(f"Device: {device.name}, Type: {device.type}, Use: {device.use}")

    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = render_samples
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3
    bpy.context.scene.cycles.transmission_bounces = 3
    bpy.context.scene.cycles.filter_width = 0.01
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.render.film_transparent = True

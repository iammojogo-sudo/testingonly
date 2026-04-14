"""
Hunyuan3D-2mv - Modly extension generator
Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2

Pipeline:
  1. Remove background from each provided view image with rembg
  2. Run Hunyuan3DDiTFlowMatchingPipeline with front/left/back/right inputs
  3. Export GLB

Paint pipeline (Hunyuan3D2mvPaintGenerator):
  1. Accept a reference image (front view) + path to an untextured GLB
  2. Run Hunyuan3DPaintPipeline from hy3dgen.texgen
  3. Export textured GLB
"""
import io
import os
import sys

# Redirect print to stderr so stdout stays clean for JSON protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)
import time
import uuid
import threading
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID       = "tencent/Hunyuan3D-2mv"
_HF_PAINT_REPO_ID = "tencent/Hunyuan3D-2"   # paint weights live here

_SUBFOLDERS = {
    "hunyuan3d-dit-v2-mv-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "hunyuan3d-dit-v2-mv-fast":  "hunyuan3d-dit-v2-mv-fast",
    "hunyuan3d-dit-v2-mv":       "hunyuan3d-dit-v2-mv",
}


def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val, default=True):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == "true"
    if val is None:
        return default
    return bool(val)


# ======================================================================== #
#  Shape generators (unchanged from original)
# ======================================================================== #

class Hunyuan3D2mvGenerator(BaseGenerator):
    MODEL_ID       = "hunyuan3d2mv"
    DISPLAY_NAME   = "Hunyuan3D-2mv"
    VRAM_GB        = 8
    MODEL_VARIANT  = "hunyuan3d-dit-v2-mv-turbo"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self):
        marker = self.model_dir / "hunyuan3d-dit-v2-mv" / "model.fp16.safetensors"
        return marker.exists()

    def _ensure_hy3dgen_on_path(self):
        repo_dir = Path(__file__).parent / "Hunyuan3D-2"
        if not repo_dir.exists():
            raise RuntimeError(
                "Hunyuan3D-2 source not found at %s. "
                "Please reinstall the extension." % repo_dir
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen_on_path()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.rembg import BackgroundRemover

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._rembg = BackgroundRemover()

        self._loaded_variant = None
        self._pipeline = None
        self._torch = torch
        self._Pipeline = Hunyuan3DDiTFlowMatchingPipeline

        self._model = True
        print("[Hunyuan3D2mvGenerator] Ready on %s." % device)

    def _load_variant(self, variant):
        if self._loaded_variant == variant:
            return
        import torch
        print("[Hunyuan3D2mvGenerator] Loading variant: %s ..." % variant)
        if self._pipeline is not None:
            del self._pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        subfolder = _SUBFOLDERS.get(variant, "hunyuan3d-dit-v2-mv-turbo")
        self._pipeline = self._Pipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=self._device,
        )
        self._loaded_variant = variant
        print("[Hunyuan3D2mvGenerator] Variant loaded: %s" % variant)

    def unload(self):
        self._pipeline = None
        self._loaded_variant = None
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes,
        params,
        progress_cb=None,
        cancel_event=None,
    ):
        import torch

        variant        = params.get("model_variant") or self.MODEL_VARIANT
        steps          = _safe_int(params.get("num_inference_steps"), 30)
        octree_res     = _safe_int(params.get("octree_resolution"), 380)
        seed           = _safe_int(params.get("seed"), 42)
        guidance_scale = _safe_float(params.get("guidance_scale"), 5.0)
        num_chunks     = _safe_int(params.get("num_chunks"), 8000)
        box_v          = _safe_float(params.get("box_v"), 1.01)
        mc_level       = _safe_float(params.get("mc_level"), 0.0)
        remove_bg      = _safe_bool(params.get("remove_bg"), True)

        print("[Hunyuan3D2mvGenerator] Parsed params: steps=%s octree=%s guidance=%.2f "
              "chunks=%s box_v=%.3f mc_level=%.4f remove_bg=%s seed=%s" % (
              steps, octree_res, guidance_scale, num_chunks, box_v, mc_level, remove_bg, seed))

        def _decode_param(key):
            val = params.get(key)
            if val is None:
                return None
            if params.get(key + "_is_b64"):
                return base64.b64decode(val)
            return val

        import base64
        left_bytes  = _decode_param("left_image")
        back_bytes  = _decode_param("back_image")
        right_bytes = _decode_param("right_image")

        # -- Background removal & preprocessing --
        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess_bytes(image_bytes, remove_bg=remove_bg)
        self._check_cancelled(cancel_event)

        image_dict = {"front": front_image}

        if left_bytes:
            self._report(progress_cb, 10, "Preprocessing left view...")
            image_dict["left"] = self._preprocess_bytes(left_bytes, remove_bg=remove_bg)
            self._check_cancelled(cancel_event)

        if back_bytes:
            self._report(progress_cb, 14, "Preprocessing back view...")
            image_dict["back"] = self._preprocess_bytes(back_bytes, remove_bg=remove_bg)
            self._check_cancelled(cancel_event)

        if right_bytes:
            self._report(progress_cb, 18, "Preprocessing right view...")
            image_dict["right"] = self._preprocess_bytes(right_bytes, remove_bg=remove_bg)
            self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvGenerator] image_dict keys: %s" % list(image_dict.keys()))

        # -- Load variant --
        self._report(progress_cb, 22, "Loading model variant...")
        self._load_variant(variant)
        self._check_cancelled(cancel_event)

        # -- Shape generation --
        self._report(progress_cb, 30, "Generating mesh...")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 92, "Generating mesh...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.no_grad():
                mesh = self._pipeline(
                    image=image_dict,
                    num_inference_steps=steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=num_chunks,
                    box_v=box_v,
                    mc_level=mc_level,
                    generator=generator,
                    output_type="trimesh",
                )[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # -- Export GLB --
        self._report(progress_cb, 94, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = "%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = self.outputs_dir / name
        mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess_bytes(self, image_bytes, remove_bg=True):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._remove_bg(img) if remove_bg else img

    def _preprocess_path(self, path, remove_bg=True):
        img = Image.open(path).convert("RGB")
        return self._remove_bg(img) if remove_bg else img

    def _remove_bg(self, img):
        try:
            result = self._rembg(img)
            return result
        except Exception:
            return img

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D2mvGenerator] Downloading weights from %s ..." % _HF_REPO_ID)
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("[Hunyuan3D2mvGenerator] Weights downloaded.")

    @classmethod
    def params_schema(cls):
        return [
            {
                "id":      "model_variant",
                "label":   "Model Variant",
                "type":    "select",
                "default": "hunyuan3d-dit-v2-mv-turbo",
                "options": [
                    {"value": "hunyuan3d-dit-v2-mv-turbo", "label": "Turbo (faster)"},
                    {"value": "hunyuan3d-dit-v2-mv",       "label": "Standard (better quality)"},
                ],
                "tooltip": "Turbo is distilled and roughly 2x faster.",
            },
            {
                "id":      "num_inference_steps",
                "label":   "Inference Steps",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast (10)"},
                    {"value": 30, "label": "Balanced (30)"},
                    {"value": 50, "label": "Quality (50)"},
                ],
                "tooltip": "Number of diffusion steps.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low (256)"},
                    {"value": 380, "label": "Medium (380)"},
                    {"value": 512, "label": "High (512)"},
                ],
                "tooltip": "Octree resolution. Higher = more detail but more VRAM.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "select",
                "default": "5.0",
                "options": [
                    {"value": "1.0",  "label": "1.0 — loose"},
                    {"value": "3.0",  "label": "3.0"},
                    {"value": "5.0",  "label": "5.0 — default"},
                    {"value": "7.5",  "label": "7.5"},
                    {"value": "10.0", "label": "10.0 — tight"},
                ],
                "tooltip": "Classifier-free guidance strength. Higher = closer to input image.",
            },
            {
                "id":      "num_chunks",
                "label":   "Decode Chunks",
                "type":    "select",
                "default": 8000,
                "options": [
                    {"value": 2000,  "label": "Low (2000) — less VRAM"},
                    {"value": 8000,  "label": "Medium (8000)"},
                    {"value": 20000, "label": "High (20000) — faster"},
                ],
                "tooltip": "VAE decode chunk size. Lower saves VRAM; higher is faster.",
            },
            {
                "id":      "box_v",
                "label":   "Bounding Box Scale",
                "type":    "select",
                "default": "1.01",
                "options": [
                    {"value": "0.75", "label": "0.75 — small"},
                    {"value": "1.01", "label": "1.01 — default"},
                    {"value": "1.25", "label": "1.25 — large"},
                    {"value": "1.5",  "label": "1.5 — extra large"},
                ],
                "tooltip": "Bounding box scale for mesh extraction. Increase if mesh edges are being clipped.",
            },
            {
                "id":      "mc_level",
                "label":   "Surface Level",
                "type":    "select",
                "default": "0.0",
                "options": [
                    {"value": "-0.05", "label": "-0.05 — thicker"},
                    {"value": "-0.02", "label": "-0.02"},
                    {"value": "0.0",   "label": "0.0 — default"},
                    {"value": "0.02",  "label": "0.02"},
                    {"value": "0.05",  "label": "0.05 — thinner"},
                ],
                "tooltip": "Marching cubes iso-surface level. Increase to thin the mesh; decrease to thicken.",
            },
            {
                "id":      "remove_bg",
                "label":   "Remove Background",
                "type":    "select",
                "default": "true",
                "options": [
                    {"value": "true",  "label": "Yes — auto remove background"},
                    {"value": "false", "label": "No — images already masked"},
                ],
                "tooltip": "Run rembg on input images. Disable if images already have a transparent background.",
            },
            {
                "id":      "left_image",
                "label":   "Left View Image (optional)",
                "type":    "image",
                "default": None,
                "tooltip": "Optional left-side view image.",
            },
            {
                "id":      "back_image",
                "label":   "Back View Image (optional)",
                "type":    "image",
                "default": None,
                "tooltip": "Optional back view image.",
            },
            {
                "id":      "right_image",
                "label":   "Right View Image (optional)",
                "type":    "image",
                "default": None,
                "tooltip": "Optional right-side view image.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": 42,
                "min":     0,
                "max":     4294967295,
                "tooltip": "Change if result is unsatisfying.",
            },
        ]


# Variant subclasses — same as before, just lock model_variant default
class Hunyuan3D2mvTurboGenerator(Hunyuan3D2mvGenerator):
    MODEL_ID      = "hunyuan3d2mv-turbo"
    DISPLAY_NAME  = "Hunyuan3D-2mv Turbo"
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-turbo"


class Hunyuan3D2mvFastGenerator(Hunyuan3D2mvGenerator):
    MODEL_ID      = "hunyuan3d2mv-fast"
    DISPLAY_NAME  = "Hunyuan3D-2mv Fast"
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-fast"


class Hunyuan3D2mvStandardGenerator(Hunyuan3D2mvGenerator):
    MODEL_ID      = "hunyuan3d2mv-standard"
    DISPLAY_NAME  = "Hunyuan3D-2mv Standard"
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv"


# ======================================================================== #
#  Paint generator  (texture synthesis via hy3dgen.texgen)
# ======================================================================== #

class Hunyuan3D2mvPaintGenerator(BaseGenerator):
    """
    Applies RGB textures to an untextured GLB using Hunyuan3D-Paint.

    Inputs (via generate()):
      image_bytes  — front view reference image (required, passed by Modly)
      params:
        mesh_path    — absolute path to the untextured .glb file (required)
        left_image   — left view bytes  (optional, base64 if _is_b64 flag set)
        back_image   — back view bytes  (optional)
        right_image  — right view bytes (optional)
        max_num_view — number of paint views (6–9)
        resolution   — texture resolution (512 / 1024)
        remove_bg    — whether to strip background from all view images

    The pipeline accepts a list of PIL images for multi-view texturing:
      pipeline(mesh, image=[front, left, back, right])
    Each extra view gives the model more colour information to paint the
    sides and back of the mesh, dramatically improving texture coverage.
    """

    MODEL_ID     = "hunyuan3d2mv/paint-texture"
    DISPLAY_NAME = "Hunyuan3D-2mv Paint"
    VRAM_GB      = 16   # paint pipeline needs ~16-21 GB; warn user

    # Paint weights live in Hunyuan3D-2 (not 2mv) on HF
    _PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0"

    def _ensure_hy3dgen_on_path(self):
        repo_dir = Path(__file__).parent / "Hunyuan3D-2"
        if not repo_dir.exists():
            raise RuntimeError(
                "Hunyuan3D-2 source not found at %s. "
                "Please reinstall the extension." % repo_dir
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def is_downloaded(self):
        # Check that the paint model subfolder exists inside the shared model_dir
        paint_dir = self.model_dir / self._PAINT_SUBFOLDER
        return paint_dir.exists() and any(paint_dir.iterdir())

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_paint_weights()

        self._ensure_hy3dgen_on_path()

        import torch
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.rembg import BackgroundRemover

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device  = device
        self._rembg   = BackgroundRemover()
        self._Pipeline = Hunyuan3DPaintPipeline

        # Lazy-load the pipeline on first generate() call so we can
        # pass the correct pretrained path at that point
        self._paint_pipeline = None
        self._model = True
        print("[Hunyuan3D2mvPaintGenerator] Ready on %s." % device)

    def _load_paint_pipeline(self):
        if self._paint_pipeline is not None:
            return
        print("[Hunyuan3D2mvPaintGenerator] Loading paint pipeline...")
        self._paint_pipeline = self._Pipeline.from_pretrained(
            str(self.model_dir),
            subfolder=self._PAINT_SUBFOLDER,
        )
        print("[Hunyuan3D2mvPaintGenerator] Paint pipeline loaded.")

    def unload(self):
        self._paint_pipeline = None
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes,
        params,
        progress_cb=None,
        cancel_event=None,
    ):
        # -- Parse params --
        mesh_path    = params.get("mesh_path", "").strip()
        max_num_view = _safe_int(params.get("max_num_view"), 6)
        resolution   = _safe_int(params.get("resolution"), 512)
        remove_bg    = _safe_bool(params.get("remove_bg"), True)

        if not mesh_path:
            raise ValueError(
                "mesh_path is required. Paste the full path to your untextured .glb file."
            )
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError("Mesh file not found: %s" % mesh_path)

        # -- Decode optional extra view bytes (same pattern as shape generators) --
        import base64

        def _decode_param(key):
            val = params.get(key)
            if val is None:
                return None
            if params.get(key + "_is_b64"):
                return base64.b64decode(val)
            return val

        left_bytes  = _decode_param("left_image")
        back_bytes  = _decode_param("back_image")
        right_bytes = _decode_param("right_image")

        print("[Hunyuan3D2mvPaintGenerator] mesh=%s views=%d res=%d remove_bg=%s "
              "extra_views=[left=%s back=%s right=%s]" % (
              mesh_path, max_num_view, resolution, remove_bg,
              bool(left_bytes), bool(back_bytes), bool(right_bytes)))

        # -- Preprocess all view images --
        # Front is always first; extras follow in order left→back→right.
        # The pipeline uses image order for view weighting so front must be index 0.
        self._report(progress_cb, 5, "Preprocessing front view...")
        images = [self._preprocess_bytes(image_bytes, remove_bg=remove_bg)]
        self._check_cancelled(cancel_event)

        if left_bytes:
            self._report(progress_cb, 7, "Preprocessing left view...")
            images.append(self._preprocess_bytes(left_bytes, remove_bg=remove_bg))
            self._check_cancelled(cancel_event)

        if back_bytes:
            self._report(progress_cb, 9, "Preprocessing back view...")
            images.append(self._preprocess_bytes(back_bytes, remove_bg=remove_bg))
            self._check_cancelled(cancel_event)

        if right_bytes:
            self._report(progress_cb, 11, "Preprocessing right view...")
            images.append(self._preprocess_bytes(right_bytes, remove_bg=remove_bg))
            self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvPaintGenerator] Texturing with %d reference view(s)." % len(images))
        # Pass a single PIL image when only front is provided, list for multi-view
        image_input = images if len(images) > 1 else images[0]

        # -- Load mesh --
        import trimesh
        self._report(progress_cb, 13, "Loading mesh...")
        mesh = trimesh.load(str(mesh_path), force="mesh")

        # -- Load pipeline --
        self._report(progress_cb, 15, "Loading paint pipeline...")
        self._load_paint_pipeline()
        self._check_cancelled(cancel_event)

        # -- Run texture synthesis --
        self._report(progress_cb, 20, "Applying textures...")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 20, 90, "Applying textures...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            import torch
            with torch.no_grad():
                textured_mesh = self._paint_pipeline(
                    mesh,
                    image=image_input,
                    max_num_view=max_num_view,
                    resolution=resolution,
                )
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # -- Export textured GLB --
        self._report(progress_cb, 92, "Exporting textured GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = "%d_%s_textured.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = self.outputs_dir / name
        textured_mesh.export(str(out_path))
        print("[Hunyuan3D2mvPaintGenerator] Exported textured GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess_bytes(self, image_bytes, remove_bg=True):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if remove_bg:
            try:
                img = self._rembg(img)
            except Exception as e:
                print("[Hunyuan3D2mvPaintGenerator] rembg failed (%s), using raw image." % e)
        return img

    def _download_paint_weights(self):
        from huggingface_hub import snapshot_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D2mvPaintGenerator] Downloading paint weights from %s ..." % _HF_PAINT_REPO_ID)
        snapshot_download(
            repo_id=_HF_PAINT_REPO_ID,
            local_dir=str(self.model_dir),
            allow_patterns=[
                "%s/**" % self._PAINT_SUBFOLDER,
                "*.json",
            ],
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("[Hunyuan3D2mvPaintGenerator] Paint weights downloaded.")

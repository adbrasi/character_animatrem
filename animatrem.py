#!/usr/bin/env python3
"""
animatrem — Anima character+outfit LoRA trainer (dataset → caption → train → HF)
================================================================================
One command, few content questions, no technical questions. You give a project
name, a character trigger, and a single image source (MEGA / zip / URL / HF /
local). The tool:

  1. downloads the images (preserving per-outfit subfolders),
  2. captions every image — PixAI booru tags → Gemini Flash via OpenRouter,
  3. trains ONE character LoRA on diffusion-pipe (multi-outfit = multi trigger,
     one [[directory]] per group, balanced repeats),
  4. uploads the .safetensors to HuggingFace WITH a rich model card (triggers,
     per-outfit example captions, training config, ComfyUI usage).

Environment: Linux / WSL2 · /workspace (or current dir) · NVIDIA GPU (12 GB+).
Trainer: tdrussell/diffusion-pipe (the model creator's own training code).
Captioner: adbrasi/data_araknideo (PixAI tagger + OpenRouter caption LLM).

Locked training defaults (tdrussell's own recipe):
- transformer_path = anima-base-v1.0.safetensors
- vae_path         = qwen_image_vae.safetensors
- llm_path         = qwen_3_06b_base.safetensors
- llm_adapter_lr   = 0  (mandatory for stable LoRA training)
- sigmoid_scale    = 1.3 (tdrussell production value)
- timestep_sample_method = 'logit_normal' (Anima's flow-matching default)
- optimizer = adamw_optimi @ 2e-5 with betas=[0.9, 0.99], wd=0.01
- recipe = character (rank 32, MSE, mixed [768,1024], ~720 exposures/img)

Required env: OPENROUTER_API_KEY (caption LLM) + HF_TOKEN (gated PixAI model,
Anima download, LoRA upload). Read from .env or asked once.

Sources of truth for the numerical defaults:
- https://gist.github.com/tdrussell/3f79596efb8e27672da0881afd9c1d51
- https://huggingface.co/circlestone-labs/Anima
- https://github.com/tdrussell/diffusion-pipe (docs/supported_models.md)
- https://lilting.ch/en/articles/wai-anima-lora-12000-step-training
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import threading
import resource
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ── Self-bootstrap: install rich if missing ──────────────────────────
def _ensure_pkg(pip_name: str, import_name: str | None = None) -> None:
    try:
        __import__(import_name or pip_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

_ensure_pkg("rich")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.markup import escape
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)

console = Console()

# Force UTF-8 stdio even on containers with a C/POSIX locale (RunPod slim),
# otherwise typed accented text is decoded as ASCII+surrogateescape and later
# crashes json/utf-8 writes with "surrogates not allowed".
for _std in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _std.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass


def _clean_surrogates(obj):
    """Recursively repair strings mangled by ASCII+surrogateescape stdin:
    re-encode via surrogateescape to recover the original bytes, then decode as
    UTF-8. Idempotent for already-valid strings; recursive over list/dict."""
    if isinstance(obj, str):
        try:
            return obj.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return obj.encode("utf-8", "replace").decode("utf-8", "replace")
    if isinstance(obj, list):
        return [_clean_surrogates(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_surrogates(v) for k, v in obj.items()}
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
if not WORKSPACE.exists():
    WORKSPACE = Path.cwd()

DIFFUSION_PIPE_REPO = "https://github.com/tdrussell/diffusion-pipe"
DIFFUSION_PIPE_DIR = WORKSPACE / "diffusion-pipe"

MODELS_DIR = WORKSPACE / "models" / "anima"
MODEL_REPO = "circlestone-labs/Anima"

# Files inside the HF repo (relative paths). We use anima-base-v1.0 — the
# current released base checkpoint (supersedes the preview line). Older
# previews are NOT exposed because their weights diverge enough that LoRAs
# don't transfer.
ANIMA_TRANSFORMER_REL = "split_files/diffusion_models/anima-base-v1.0.safetensors"
ANIMA_VAE_REL         = "split_files/vae/qwen_image_vae.safetensors"
ANIMA_LLM_REL         = "split_files/text_encoders/qwen_3_06b_base.safetensors"

PROJECTS_DIR = WORKSPACE / "projects"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ─── Captioner engine (data_araknideo) ───────────────────────────────
# Cloned at bootstrap; drives the locked caption pipeline (PixAI booru tags
# fed as context to the OpenRouter caption LLM). The grok/vision stage is
# model-agnostic on OpenRouter, so we point --grok_model at Gemini Flash.
CAPTIONER_REPO   = "https://github.com/adbrasi/data_araknideo"
CAPTIONER_DIR    = WORKSPACE / "data_araknideo"
CAPTIONER_SCRIPT = CAPTIONER_DIR / "tag_images_by_wd14_tagger.py"

# Locked caption LLM: Gemini Flash via OpenRouter (structured JSON supported).
CAPTION_MODEL = os.environ.get("ANIMATREM_CAPTION_MODEL", "google/gemini-2.5-flash")

# Prompt profiles used for captioning (resolved inside the captioner's
# prompts/image/<profile>/ tree; the custom one is copied in at bootstrap).
PROFILE_CHARACTER          = "anima-character"          # base group (no outfit)
PROFILE_CHARACTER_OUTFIT   = "anima-character-outfit"   # outfit, LOCKED (trigger-only)
PROFILE_CHARACTER_OUTFIT_DESCRIBED = "anima-character-outfit-described"  # outfit, DESCRIBED

# This repo's own directory (script + custom prompt profile live here).
REPO_DIR           = Path(__file__).resolve().parent
CUSTOM_PROMPTS_DIR = REPO_DIR / "prompts"


# ─── Resolution presets ──────────────────────────────────────────────
# Anima is a DiT — every resolution must be "exercised" during training,
# otherwise the LoRA cracks at that resolution at inference time. tdrussell's
# rationale (verbatim from the Greg-Rutkowski LoRA card):
#   - 512  ensina identidade/composição (rápido, muitas imagens por hora)
#   - 1024 ensina detalhes médios (resolução de inferência padrão)
#   - 1536 ensina detalhes finos de estilo, opcional
# tdrussell oficial = [512, 1024, 1536]. O usuário pediu remover 512; isso
# significa LoRA mais lenta para aprender estrutura e potencial fragilidade
# em inferência <768. Documentado abaixo.
RESOLUTION_PRESETS: dict[str, tuple[str, list[int], str]] = {
    "1": ("1024",            [1024],             "1024 — padrão recomendado para inferência"),
    "2": ("768",              [768],             "768  — mais rápido, menor VRAM"),
    "3": ("1536",            [1536],             "1536 — alta qualidade (mais lento, mais VRAM)"),
    "4": ("mixed-768-1024",  [768, 1024],        "Mixed 768+1024 — robusto, recomendado em geral"),
    "5": ("mixed-full",      [768, 1024, 1536],  "Mixed 768+1024+1536 — receita estilo grande (≈ tdrussell sem 512)"),
}


# ─── GPU presets ─────────────────────────────────────────────────────
# Anima is a 2B DiT — bf16 weights are ~4 GB. Even a 32 GB card has
# massive headroom, so gradient_accumulation_steps is NOT a VRAM workaround
# for any GPU listed below. tdrussell's published Greg-Rutkowski recipe uses
# `micro_batch=1, grad_accum=4 → eff batch 4` (the canonical Anima target).
# Two valid ways to hit batch 4 on a single GPU:
#   (a) micro_batch=1, grad_accum=4   ← tdrussell-faithful (slower)
#   (b) micro_batch=4, grad_accum=1   ← VRAM-efficient (3-4× faster on 5090)
# We default to (b) because it's strictly faster and produces equivalent
# gradients (no extra grad noise vs accumulation). User can flip on demand.
#
# `max_micro_batch` is the highest per-res micro batch we trust at bf16
# with activation_checkpointing=true. Numbers calibrated against:
#   - HF #92: CAME at 1024² batch 4 ≈ 14.5 GB (Anima 2B)
#   - sorryhyun (5060 Ti, 16 GB): batch 1 at 1024 ≈ 14 GB
#   - extrapolation by VRAM headroom for higher-VRAM cards
GPU_PROFILES: dict[str, dict] = {
    "B200": {
        "label": "B200 / GB200 (180 GB+)",
        "vram_gb_min": 160,
        "max_micro_batch": {768: 32, 1024: 16, 1536: 8},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 0,
        "fp8_default": False,
    },
    "H100_80": {
        "label": "H100 80GB / A100 80GB",
        "vram_gb_min": 75,
        "max_micro_batch": {768: 16, 1024: 8, 1536: 4},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 0,
        "fp8_default": False,
    },
    "RTX_PRO_6000": {
        "label": "RTX Pro 6000 Ada / L40S (48 GB)",
        "vram_gb_min": 44,
        "max_micro_batch": {768: 12, 1024: 8, 1536: 4},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 0,
        "fp8_default": False,
    },
    "RTX_5090": {
        "label": "RTX 5090 (32 GB Blackwell)",
        "vram_gb_min": 30,
        # User-validated on real run (29 Apr 2026):
        #   micro=6 at [768,1024] mixed → 18 GB used (12 GB headroom).
        #   micro=8 at [768,1024,1536] mixed → OOM (1536 breaks first).
        # Linear extrapolation from the 6→18GB point:
        #   batch 8 at 1024 ≈ 24 GB (safe), batch 10 ≈ 30 GB (tight),
        #   batch 12 ≈ 36 GB (OOM).
        #   1536 has 2.25× more activations vs 1024, so batch 2 ≈ 16 GB,
        #   batch 4 likely OOM in mixed-res context.
        "max_micro_batch": {512: 24, 768: 16, 1024: 8, 1536: 2},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 0,
        "fp8_default": False,
    },
    "RTX_4090": {
        "label": "RTX 4090 / 5080 (24 GB)",
        "vram_gb_min": 22,
        "max_micro_batch": {768: 8, 1024: 4, 1536: 2},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 0,
        "fp8_default": False,
    },
    "RTX_3090": {
        "label": "RTX 3090 / 4080 (16-24 GB)",
        "vram_gb_min": 16,
        "max_micro_batch": {768: 4, 1024: 2, 1536: 1},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 8,
        "fp8_default": False,
    },
    "RTX_4070_16": {
        "label": "RTX 4070 Ti / 5070 (12-16 GB)",
        "vram_gb_min": 12,
        "max_micro_batch": {768: 2, 1024: 1},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 16,
        "fp8_default": True,
    },
    "GENERIC": {
        "label": "Genérica / pouco VRAM",
        "vram_gb_min": 0,
        "max_micro_batch": {768: 1},
        "activation_checkpointing": True,
        "pipeline_stages": 1,
        "blocks_to_swap": 20,
        "fp8_default": True,
    },
}


# ─── Training recipe presets ─────────────────────────────────────────
# Numbers extracted VERBATIM from tdrussell's published configs (anima_lora.toml
# gist + Greg-Rutkowski LoRA card on CivitAI #2536147), validated against
# lilting.ch character LoRA (12k-step study, v4 verified) and tdrussell HF
# Discussion #112 ("rank 32 lora, AdamW 2e-5 LR, no LLM adapter training").
#
# `exposures_per_image` already includes a +10% safety margin so the training
# overshoots the canonical sweet-spot rather than undershooting. Users always
# get the BEST checkpoint via diffusion-pipe's epoch-saved files; running a
# bit longer than necessary is safer than stopping a bit short.
#
# `target_global_batch` is tdrussell's effective per-step batch. We hit it
# preferring real micro_batch (no grad_accum gambiarra) when VRAM permits,
# else fall back to grad_accum (computed by _compute_batching).
RECIPES: dict[str, dict] = {
    "character": {
        "label":        "PERSONAGEM (50–300 imagens, lilting v4 + tdrussell #112)",
        "rank":         32,
        "lr":           2.0e-5,
        "betas":        [0.9, 0.99],
        "weight_decay": 0.01,
        "eps":          1e-8,
        "loss":         "mse",
        "pseudo_huber_c": None,
        # lilting v4 (53-img kanachan, validated): 600–720 expos/img = sweet
        # spot; ep150(600)/ep180(720) hit 100% direction accuracy, ep200(800)
        # collapsed to 0% (ghosting). Target the MIDDLE of the window (660) for
        # margin below the ~800 cliff; save ~11 checkpoints to cherry-pick.
        "exposures_per_image": 660,
        "warmup_steps": 100,
        "lr_scheduler": "constant",
        "min_repeats":  1,
        "max_repeats":  6,
        "self_attn_lr": None,
        "cross_attn_lr": None,
        "mlp_lr":        None,
        "mod_lr":        None,
        "caption_prefix": "",
        "default_resolutions": [768, 1024],   # mixed sem 1536 (datasets pequenos)
        "target_global_batch": 4,             # tdrussell canônico
        "description":   "Rank 32 · LR 2e-5 · MSE · mixed [768,1024] · ~720 expos/img.",
    },
    "style": {
        "label":        "ESTILO (500+ imagens) — receita Greg Rutkowski (tdrussell verbatim)",
        "rank":         32,
        "lr":           2.0e-5,
        "betas":        [0.9, 0.99],
        "weight_decay": 0.01,
        "eps":          1e-8,
        "loss":         "mse",
        "pseudo_huber_c": None,
        # tdrussell oficial: 40 epochs × 3 res = 120 passes/img.
        # +10% folga = 132. Em mixed-full (3 res) → ~44 epochs.
        "exposures_per_image": 132,
        "warmup_steps": 100,
        "lr_scheduler": "constant",
        "min_repeats":  1,
        "max_repeats":  1,
        "self_attn_lr": None,
        "cross_attn_lr": None,
        "mlp_lr":        None,
        "mod_lr":        None,
        "caption_prefix": "",                 # captioner injeta trigger no .txt
        "default_resolutions": [768, 1024, 1536],   # 768 é o mínimo moderno; 1536 usa micro=2 no 5090
        "target_global_batch": 4,
        "description":   "Rank 32 · LR 2e-5 · MSE · mixed [768,1024,1536] · 132 expos/img.",
    },
    "concept": {
        "label":        "CONCEITO (200–1000 imagens) — roupa, tatuagem, pose, prop",
        "rank":         32,
        "lr":           2.0e-5,
        "betas":        [0.9, 0.99],
        "weight_decay": 0.01,
        "eps":          1e-8,
        "loss":         "mse",
        "pseudo_huber_c": None,
        # Conceitos focados precisam de mais reforço que estilo (densidade de
        # exemplos do conceito específico no dataset costuma ser baixa) mas
        # menos que personagem (não é identidade completa). Midpoint + folga.
        "exposures_per_image": 240,
        "warmup_steps": 100,
        "lr_scheduler": "constant",
        "min_repeats":  1,
        "max_repeats":  3,
        "self_attn_lr": None,
        "cross_attn_lr": None,
        "mlp_lr":        None,
        "mod_lr":        None,
        "caption_prefix": "",
        "default_resolutions": [768, 1024],
        "target_global_batch": 4,
        "description":   "Rank 32 · LR 2e-5 · MSE · mixed [768,1024] · ~240 expos/img.",
    },
    "full_finetune": {
        "label":        "FULL FINETUNE (5k+ imagens, sem [adapter]) — tdrussell #112 / Bluvoll",
        "rank":         0,                    # 0 → no adapter table written
        "lr":           8.0e-6,               # Bluvoll-validated FFT LR
        "betas":        [0.9, 0.99],
        "weight_decay": 0.01,
        "eps":          1e-8,
        "loss":         "mse",
        "pseudo_huber_c": None,
        "exposures_per_image": 36,            # ~30 base + 20% folga
        "warmup_steps": 200,
        "lr_scheduler": "constant",
        "min_repeats":  1,
        "max_repeats":  1,
        "self_attn_lr": None,
        "cross_attn_lr": None,
        "mlp_lr":        None,
        "mod_lr":        None,
        "caption_prefix": "",
        "default_resolutions": [1024],        # FFT é caro; ficar em 1024 puro
        "target_global_batch": 16,            # tdrussell #112 base-train
        "description":   "Sem [adapter] · LR 8e-6 · MSE · 1024² · ~36 expos/img · global batch 16.",
    },
    "custom": {
        "label":        "CUSTOMIZADO — pergunta tudo (rank, LR, batch, scheduler, resoluções, etc.)",
        # Defaults (override one-by-one in phase4 custom branch):
        "rank":         32,
        "lr":           2.0e-5,
        "betas":        [0.9, 0.99],
        "weight_decay": 0.01,
        "eps":          1e-8,
        "loss":         "mse",
        "pseudo_huber_c": None,
        "exposures_per_image": 200,
        "warmup_steps": 100,
        "lr_scheduler": "constant",
        "min_repeats":  1,
        "max_repeats":  4,
        "self_attn_lr": None,
        "cross_attn_lr": None,
        "mlp_lr":        None,
        "mod_lr":        None,
        "caption_prefix": "",
        "default_resolutions": [1024],
        "target_global_batch": 4,
        "description":   "Você responde rank, LR, scheduler, resoluções, batch, etc.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Config dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    project_name:       str = "my-anima-lora"
    base_lora_path:     str = ""
    dataset_dirs:       list[str] = field(default_factory=list)

    # Character + outfit grouping (animatrem)
    trigger_character:  str = ""
    # groups: one dict per dataset [[directory]] block:
    #   {path, is_outfit, trigger_outfit, custom_instruction, image_count,
    #    num_repeats, caption_examples}
    groups:             list = field(default_factory=list)
    hf_private:         bool = True
    # Outfit caption mode: False (default, the correct Anima path) = describe the
    # clothing normally AND keep the outfit trigger; True = describe the outfit
    # ONLY as its trigger (locked — rigid, generally worse).
    outfit_lock:        bool = False

    # Recipe (one of RECIPES keys)
    recipe:             str = "character"

    # Model files (resolved at runtime, can be overridden)
    transformer_path:   str = ""
    vae_path:           str = ""
    llm_path:           str = ""

    # Resolution
    resolutions:        list[int] = field(default_factory=lambda: [1024])
    min_ar:             float = 0.5
    max_ar:             float = 2.0
    num_ar_buckets:     int   = 7

    # LoRA / training
    rank:               int = 32
    learning_rate:      float = 2.0e-5
    optimizer_type:     str = "adamw_optimi"
    betas:              list[float] = field(default_factory=lambda: [0.9, 0.99])
    weight_decay:       float = 0.01
    eps:                float = 1e-8
    lr_scheduler:       str = "constant"
    warmup_steps:       int = 50
    gradient_clipping:  float = 1.0
    loss:               str = "mse"
    pseudo_huber_c:     Optional[float] = None

    # Per-component LR (anima-specific, optional)
    self_attn_lr:       Optional[float] = None
    cross_attn_lr:      Optional[float] = None
    mlp_lr:             Optional[float] = None
    mod_lr:             Optional[float] = None
    llm_adapter_lr:     float = 0.0  # MANDATORY 0

    # Anima-specific
    sigmoid_scale:      float = 1.3
    timestep_sample_method: str = "logit_normal"

    # Compute schedule (auto-derived from dataset size)
    epochs:             int = 100
    num_repeats:        int = 1
    exposures_per_image: int = 660

    # Batching
    micro_batch_size:   list = field(default_factory=lambda: [[1024, 1]])  # list-of-pairs or int
    gradient_accumulation_steps: int = 4

    # GPU
    activation_checkpointing: bool = True
    pipeline_stages:    int = 1
    blocks_to_swap:     int = 0
    use_fp8:            bool = False
    transformer_dtype:  str = "bfloat16"  # "float8" if use_fp8

    # Captioning hooks (user supplies captions externally)
    caption_prefix:     str = ""
    cache_shuffle_num:  int = 0  # 0 = no shuffle (recommended for Anima NL captions)
    shuffle_tags:       bool = False

    # Save / eval — native diffusion-pipe cadence. The character path saves by
    # STEPS (save_every_n_steps = total_steps // ~10); epoch-based is kept for
    # the --advanced/FFT path. total_steps is informational.
    save_every_n_epochs:    int = 0
    save_every_n_steps:     int = 0
    eval_every_n_epochs:    int = 0
    eval_every_n_steps:     int = 0
    total_steps:            int = 0
    eval_before_first_step: bool = True
    checkpoint_every_n_minutes: int = 60

    # Output paths (computed)
    output_dir:         str = ""
    dataset_toml_path:  str = ""
    config_toml_path:   str = ""

    # Counts (refreshed on save)
    image_count:        int = 0
    caption_count:      int = 0

    @property
    def project_dir(self) -> Path:
        return PROJECTS_DIR / self.project_name

    def _saved_config_path(self) -> Path:
        return self.project_dir / ".config.json"

    def save(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        data = _clean_surrogates({k: v for k, v in self.__dict__.items()})
        self._saved_config_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, project_name: str) -> Optional["TrainingConfig"]:
        path = PROJECTS_DIR / project_name / ".config.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cfg = cls()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            return cfg
        except Exception:
            return None

    @classmethod
    def list_projects(cls) -> list[str]:
        if not PROJECTS_DIR.exists():
            return []
        return sorted(
            d.name for d in PROJECTS_DIR.iterdir()
            if d.is_dir() and (d / ".config.json").exists()
        )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        kwargs.setdefault("shell", True)
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)


def run_capture(cmd: list[str] | str, **kwargs) -> str:
    if isinstance(cmd, str):
        kwargs.setdefault("shell", True)
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return result.stdout.strip()


def check_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def mark_git_safe(path: str | Path) -> None:
    run(["git", "config", "--global", "--add", "safe.directory", str(Path(path).resolve())], check=False)


def header(title: str) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold dark_blue]{title}[/bold dark_blue]",
            border_style="dark_magenta",
            padding=(0, 2),
            expand=True,
        )
    )
    console.print()


def success(msg: str) -> None: console.print(f"  [bold dark_green]✔[/bold dark_green] {msg}")
def warn(msg: str)    -> None: console.print(f"  [bold dark_orange3]⚠[/bold dark_orange3] {msg}")
def error(msg: str)   -> None: console.print(f"  [bold red1]✘[/bold red1] {msg}")
def info(msg: str)    -> None: console.print(f"  [bold dark_cyan]ℹ[/bold dark_cyan] {msg}")


def reset_terminal_input() -> None:
    try:
        console.show_cursor(True)
    except Exception:
        pass
    if sys.stdin.isatty():
        subprocess.run(["stty", "sane"], check=False, stdin=sys.stdin,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pause() -> None:
    reset_terminal_input()
    console.print()
    console.print("  [dark_cyan]Pressione ENTER para continuar[/dark_cyan]", end="")
    input()
    console.print()


def ask(prompt: str, default: str = "", allow_empty: bool = True) -> str:
    reset_terminal_input()
    if default == "" and allow_empty:
        console.print(f"  [bold]{prompt}[/bold] [dim](vazio para pular)[/dim]: ", end="")
        return input().strip()
    console.print(f"  [bold]{prompt}[/bold] [dim]\\[{escape(default)}][/dim]: ", end="")
    result = input().strip()
    return result if result else default


def ask_yn(prompt: str, default: bool = True) -> bool:
    reset_terminal_input()
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        console.print(f"  [bold]{prompt}[/bold] {escape(hint)}: ", end="")
        result = input().strip().lower()
        if result == "":
            return default
        if result in ("y", "yes", "s", "sim"):
            return True
        if result in ("n", "no", "não", "nao"):
            return False
        warn(f"Resposta inválida: '{result}'. Use y/n")


def ask_choice(prompt: str, choices: dict[str, str], default: str = "1",
               allow_raw: bool = False) -> str:
    reset_terminal_input()
    for key, desc in choices.items():
        console.print(f"    [bold]{key})[/bold] {desc}")
    console.print()
    while True:
        console.print(f"  [bold]{prompt}[/bold] [dim]\\[{escape(default)}][/dim]: ", end="")
        result = input().strip() or default
        if result in choices:
            return result
        if allow_raw:
            return result
        warn(f"Opção inválida: {result}. Escolha entre: {', '.join(choices.keys())}")
        console.print()


def count_files(directory: str | Path, exts: set[str]) -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def count_images(directory: str | Path) -> int:
    return count_files(directory, IMAGE_EXTS)


def count_captions(directory: str | Path) -> int:
    return count_files(directory, {".txt"})


def get_dataset_entries(cfg: TrainingConfig) -> list[Path]:
    entries: list[Path] = []
    for raw in cfg.dataset_dirs:
        root = Path(raw)
        if not root.exists():
            continue
        # If root has images directly → use root.
        # Otherwise walk one level to find subdirs with images.
        direct = sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS) if root.is_dir() else 0
        if direct > 0:
            entries.append(root)
            continue
        if root.is_dir():
            child_dirs = sorted(d for d in root.iterdir()
                                if d.is_dir() and count_images(d) > 0)
            if child_dirs:
                entries.extend(child_dirs)
            elif count_images(root) > 0:
                entries.append(root)
    seen, unique = set(), []
    for e in entries:
        k = str(e.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return unique


def total_image_count(cfg: TrainingConfig) -> int:
    entries = get_dataset_entries(cfg)
    return sum(count_images(e) for e in entries) if entries else 0


def total_caption_count(cfg: TrainingConfig) -> int:
    entries = get_dataset_entries(cfg)
    return sum(count_captions(e) for e in entries) if entries else 0


def find_images_missing_captions(cfg: TrainingConfig) -> list[Path]:
    missing: list[Path] = []
    for d in get_dataset_entries(cfg):
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                if not p.with_suffix(".txt").exists():
                    missing.append(p)
    return missing


def prepare_large_safetensors_mmap() -> None:
    """Allow safetensors to mmap the Anima preview checkpoint (~8 GB)."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            new_soft = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
    except (OSError, ValueError):
        pass


# ─── Batching helper ────────────────────────────────────────────────

def _compute_batching(profile: dict, resolutions: list[int],
                      target_global_batch: int) -> tuple:
    """Pick (micro_batch_per_res, grad_accum_steps) honestly.

    Strategy:
      1. Try to hit `target_global_batch` with grad_accum=1 (pure micro batch
         scaling). This is preferred — no accumulation noise, faster.
      2. If the smallest per-res VRAM ceiling can't reach the target, fall
         back to grad_accum > 1 — but only as much as needed.

    Returns:
        (micro_batch_per_res, grad_accum, info_str)
        - micro_batch_per_res: list of [res, batch] pairs, OR a single int if
          all resolutions share the same batch (cleaner TOML).
        - info_str: human-readable summary of what was chosen and why.
    """
    max_mb = profile["max_micro_batch"]

    # Per-resolution effective ceiling (clip the requested batch).
    per_res_cap = []
    for r in resolutions:
        if r in max_mb:
            per_res_cap.append((r, max_mb[r]))
        else:
            # Pick the closest tabulated res ≤ r (conservative).
            tab = sorted(max_mb.keys())
            chosen = tab[0]
            for k in tab:
                if k <= r:
                    chosen = k
            per_res_cap.append((r, max_mb[chosen]))

    smallest_cap = min(c for _, c in per_res_cap)

    if smallest_cap >= target_global_batch:
        # Path (1): pure micro batch, no grad_accum.
        # Use target_global_batch on every resolution — VRAM allows it.
        micro = [[r, target_global_batch] for r, _ in per_res_cap]
        if len(set(b for _, b in micro)) == 1:
            return target_global_batch, 1, (
                f"micro_batch={target_global_batch} (sem grad_accum) — caber em VRAM "
                f"em todas as resoluções"
            )
        return micro, 1, "micro_batch por resolução, sem grad_accum"

    # Path (2): grad_accum needed at least somewhere. Use the smallest_cap as
    # micro on every res, scale grad_accum up to hit the target.
    grad_accum = max(1, (target_global_batch + smallest_cap - 1) // smallest_cap)
    actual = smallest_cap * grad_accum
    if actual != target_global_batch:
        info_msg = (
            f"micro_batch={smallest_cap}, grad_accum={grad_accum} → "
            f"eff batch {actual} (≥ alvo {target_global_batch})"
        )
    else:
        info_msg = (
            f"micro_batch={smallest_cap}, grad_accum={grad_accum} → "
            f"eff batch {actual}"
        )
    if len(set(c for _, c in per_res_cap)) == 1:
        return smallest_cap, grad_accum, info_msg
    # Mixed-res: keep smallest_cap on the constrained res, allow larger on others
    # (but grad_accum is global, so capping at smallest is the only honest play).
    micro_per_res = [[r, smallest_cap] for r, _ in per_res_cap]
    return micro_per_res, grad_accum, info_msg


# ─── GPU detection ──────────────────────────────────────────────────

def detect_gpu_profile(gpu_name: str, vram_gb: int) -> str:
    """Match a detected GPU + VRAM to one of the GPU_PROFILES keys."""
    n = (gpu_name or "").lower()
    if "b200" in n or "b100" in n or "gb200" in n or vram_gb >= 160:
        return "B200"
    if "h100" in n or "h200" in n:
        return "H100_80"
    if "a100" in n and vram_gb >= 75:
        return "H100_80"
    if "rtx pro 6000" in n or "l40s" in n or vram_gb in range(44, 50):
        return "RTX_PRO_6000"
    if "5090" in n:
        return "RTX_5090"
    if "4090" in n or "5080" in n or vram_gb in range(22, 26):
        return "RTX_4090"
    if "3090" in n or "4080" in n or vram_gb in range(16, 22):
        return "RTX_3090"
    if vram_gb >= 12:
        return "RTX_4070_16"
    return "GENERIC"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0 — Preflight
# ═══════════════════════════════════════════════════════════════════════════

def phase0_preflight() -> None:
    header("FASE 0 — Verificação de dependências básicas")

    required = ["git", "python3"]
    optional = [("wget", "curl"), ("nvidia-smi",)]

    all_ok = True
    for c in required:
        if check_cmd(c):
            success(f"{c} encontrado")
        else:
            error(f"{c} NÃO encontrado")
            all_ok = False

    for group in optional:
        found = [c for c in group if check_cmd(c)]
        if found:
            success(f"{'/'.join(group)}: {found[0]} disponível")
        else:
            warn(f"Nenhum encontrado: {'/'.join(group)}")

    if not all_ok:
        if not ask_yn("Continuar mesmo assim?", default=False):
            sys.exit(1)

    prepare_large_safetensors_mmap()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Environment / GPU detection
# ═══════════════════════════════════════════════════════════════════════════

def phase1_check_environment() -> dict:
    header("FASE 1 — Detecção do ambiente")

    env: dict = {}

    if not check_cmd("nvidia-smi"):
        error("nvidia-smi não encontrado. CUDA é necessário.")
        sys.exit(1)

    gpu_name = run_capture("nvidia-smi --query-gpu=name --format=csv,noheader | head -1")
    vram_mb_str = run_capture("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1")
    try:
        vram_mb = int(vram_mb_str)
    except ValueError:
        vram_mb = 0
    vram_gb = vram_mb // 1024

    env["gpu_name"] = gpu_name
    env["vram_mb"]  = vram_mb
    env["vram_gb"]  = vram_gb

    # Disk
    disk_free = run_capture(f"df -h {WORKSPACE} | awk 'NR==2{{print $4}}'")
    # RAM
    try:
        ram_mb = int(run_capture("awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo"))
    except Exception:
        ram_mb = 0
    env["ram_mb"] = ram_mb

    profile = detect_gpu_profile(gpu_name, vram_gb)
    env["gpu_profile"] = profile

    is_blackwell = any(x in gpu_name.lower() for x in ["5090", "5080", "5070", "5060", "5050", "blackwell", "b200", "b100", "gb200"])
    env["is_blackwell"] = is_blackwell

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="bold"); t.add_column()
    t.add_row("GPU",          gpu_name or "?")
    t.add_row("VRAM",         f"{vram_gb} GB ({vram_mb} MB)")
    t.add_row("RAM sistema",  f"{ram_mb // 1024} GB" if ram_mb else "?")
    t.add_row("Disco livre",  disk_free)
    t.add_row("Perfil GPU",   GPU_PROFILES[profile]["label"])
    t.add_row("Blackwell?",   "sim (CUDA 12.8+ recomendado)" if is_blackwell else "não")
    console.print(t)
    console.print()

    # HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token() or ""
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
        except Exception:
            pass
    if hf_token:
        success(f"HF_TOKEN: definido ({hf_token[:8]}...)")
    else:
        warn("HF_TOKEN não encontrado — necessário para baixar circlestone-labs/Anima")
        console.print("        [dim]export HF_TOKEN=hf_xxxxxxxxxx[/dim]")

    env["hf_token"] = hf_token
    return env


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Install diffusion-pipe
# ═══════════════════════════════════════════════════════════════════════════

def phase2_install_diffusion_pipe(env: dict) -> None:
    header("FASE 2 — Instalação do diffusion-pipe (tdrussell)")

    # huggingface_hub for fast downloads
    try:
        import huggingface_hub  # noqa: F401
        try:
            import hf_xet  # noqa: F401
        except ImportError:
            info("Instalando hf_xet (downloads rápidos)...")
            run([sys.executable, "-m", "pip", "install", "-q",
                 "huggingface_hub[hf-xet]>=0.31.4"])
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q",
             "huggingface_hub[hf-xet]>=0.31.4"])

    # Clone diffusion-pipe with submodules
    if not DIFFUSION_PIPE_DIR.exists():
        info(f"Clonando diffusion-pipe em {DIFFUSION_PIPE_DIR}...")
        DIFFUSION_PIPE_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--recurse-submodules", DIFFUSION_PIPE_REPO,
             str(DIFFUSION_PIPE_DIR)])
        mark_git_safe(DIFFUSION_PIPE_DIR)
        success(f"Repositório clonado em {DIFFUSION_PIPE_DIR}")
    else:
        mark_git_safe(DIFFUSION_PIPE_DIR)
        info("Atualizando diffusion-pipe...")
        run(["git", "-C", str(DIFFUSION_PIPE_DIR), "pull"], check=False)
        run(["git", "-C", str(DIFFUSION_PIPE_DIR), "submodule", "update", "--init", "--recursive"], check=False)
        success("diffusion-pipe atualizado")

    # PyTorch (Blackwell needs CUDA 12.8+)
    try:
        import torch
        torch_cuda = torch.version.cuda or "?"
        success(f"PyTorch {torch.__version__} (CUDA {torch_cuda})")
        if env.get("is_blackwell") and not (torch_cuda or "").startswith(("12.8", "12.9", "13.")):
            warn("Blackwell detectada mas PyTorch sem CUDA ≥12.8.")
            if ask_yn("Reinstalar PyTorch com CUDA 12.8?", default=True):
                run([sys.executable, "-m", "pip", "install", "-q",
                     "torch", "torchvision",
                     "--index-url", "https://download.pytorch.org/whl/cu128"])
    except ImportError:
        info("Instalando PyTorch...")
        cuda_idx = "cu128" if env.get("is_blackwell") else "cu124"
        run([sys.executable, "-m", "pip", "install", "-q",
             "torch", "torchvision",
             "--index-url", f"https://download.pytorch.org/whl/{cuda_idx}"])

    # Install diffusion-pipe requirements
    req_file = DIFFUSION_PIPE_DIR / "requirements.txt"
    if req_file.exists():
        info("Instalando requirements do diffusion-pipe (deepspeed, transformers, etc.)...")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])
        success("requirements instalados")
    else:
        warn(f"{req_file} não encontrado — pulei pip install -r")

    # Optional flash-attn (some optimizers/components benefit)
    if env.get("vram_gb", 0) >= 16 and not env.get("is_blackwell"):
        try:
            import flash_attn  # noqa: F401
            success("flash-attn já instalado")
        except ImportError:
            if ask_yn("Instalar flash-attn (acelera atenção)?", default=False):
                info("Instalando flash-attn (pode demorar)...")
                run([sys.executable, "-m", "pip", "install", "-q", "flash-attn",
                     "--no-build-isolation"], check=False)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2b — Install captioner engine (data_araknideo)
# ═══════════════════════════════════════════════════════════════════════════

def phase2b_install_captioner(env: dict) -> None:
    header("FASE 2b — Instalação do captioner (data_araknideo)")

    if not CAPTIONER_DIR.exists():
        info(f"Clonando data_araknideo em {CAPTIONER_DIR}...")
        CAPTIONER_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", CAPTIONER_REPO, str(CAPTIONER_DIR)])
        mark_git_safe(CAPTIONER_DIR)
        success("Captioner clonado")
    else:
        mark_git_safe(CAPTIONER_DIR)
        info("Atualizando captioner...")
        run(["git", "-C", str(CAPTIONER_DIR), "pull"], check=False)
        success("Captioner atualizado")

    # Captioner deps: PixAI tagger (torch+timm), OpenRouter LLM (requests),
    # huggingface_hub. torch already installed by phase2; the rest are light.
    req = CAPTIONER_DIR / "requirements.txt"
    if req.exists():
        info("Instalando requirements do captioner (timm, pillow, requests, ...)...")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        success("requirements do captioner instalados")

    # Copy our custom prompt profile(s) into the captioner's prompts tree.
    _install_custom_profiles()

    if not CAPTIONER_SCRIPT.exists():
        error(f"Script do captioner não encontrado: {CAPTIONER_SCRIPT}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Download Anima models
# ═══════════════════════════════════════════════════════════════════════════

def _hf_download_file(repo: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN") or None
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    return Path(hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(local_dir),
        token=token,
    ))


def phase3_download_models(env: dict) -> tuple[Path, Path, Path]:
    header("FASE 3 — Download dos modelos Anima (circlestone-labs/Anima)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Single fixed transformer: anima-base-v1.0 (current released base).
    info(f"Transformer fixo: anima-base-v1.0.safetensors (base atual)")
    console.print()

    files = [
        ("transformer", ANIMA_TRANSFORMER_REL, "DiT Anima base-v1.0 (~4.2 GB)"),
        ("vae",         ANIMA_VAE_REL,         "Qwen-Image VAE 16-ch (~250 MB)"),
        ("llm",         ANIMA_LLM_REL,         "Qwen3-0.6B base (~1.2 GB)"),
    ]

    # Show what we're downloading
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Componente", style="bold")
    t.add_column("Arquivo")
    t.add_column("Descrição", style="dim")
    for kind, rel, desc in files:
        t.add_row(kind, Path(rel).name, desc)
    console.print(t)
    console.print()

    paths: dict[str, Path] = {}
    for kind, rel, desc in files:
        local_path = MODELS_DIR / Path(rel).name
        if local_path.exists() and local_path.stat().st_size > 1024:
            size = run_capture(f"du -h {local_path} | cut -f1")
            success(f"{kind}: já presente ({size})")
            paths[kind] = local_path
            continue

        info(f"Baixando {Path(rel).name}...")
        downloaded = _hf_download_file(MODEL_REPO, rel, MODELS_DIR)

        # hf_hub_download keeps the relative subdir structure → flatten.
        if downloaded != local_path:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(downloaded), str(local_path))
            except (shutil.Error, OSError):
                shutil.copy2(str(downloaded), str(local_path))
        success(f"{kind}: baixado para {local_path.name}")
        paths[kind] = local_path

    return paths["transformer"], paths["vae"], paths["llm"]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset source preparation (local / HF / zip URL)
# ═══════════════════════════════════════════════════════════════════════════

def _is_hf_repo(value: str) -> bool:
    if re.match(r"^https?://", value):
        return False
    if Path(value).exists():
        return False
    if "/" in value and not value.startswith(("/", ".")):
        parts = value.split("/")
        if len(parts) == 2 and all(re.match(r"^[\w\-\.]+$", p) for p in parts):
            return True
    return False


def _is_mega_url(value: str) -> bool:
    return value.startswith(("https://mega.nz/", "https://mega.io/", "mega://"))


_ARCHIVE_EXTS = (".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.xz",
                 ".tar.bz2", ".tbz2")


def _detect_archive_ext(name: str) -> str | None:
    name_l = name.lower()
    for ext in _ARCHIVE_EXTS:
        if name_l.endswith(ext):
            return ext
    return None


def _ensure_archive_tool(ext: str) -> None:
    """Install the archive tool transparently if missing. apt-get with sudo
    fallback; if the user has no sudo available, we surface the error."""
    needed = {
        ".zip": ("unzip", ["unzip"]),
        ".rar": ("unrar", ["unrar"]),
        ".7z":  ("7z",    ["p7zip-full"]),
    }
    if ext in (".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tbz2"):
        return  # tarfile is in stdlib
    if ext not in needed:
        return
    cmd_name, apt_pkgs = needed[ext]
    if check_cmd(cmd_name):
        return
    info(f"Instalando {cmd_name} (extração de {ext})...")
    apt = "sudo apt-get" if check_cmd("sudo") and os.geteuid() != 0 else "apt-get"
    run(f"{apt} update -qq && {apt} install -y -qq {' '.join(apt_pkgs)}", check=False)
    if not check_cmd(cmd_name):
        error(f"Falha ao instalar {cmd_name} para extrair {ext}")
        sys.exit(1)


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract any supported archive into `dest`."""
    ext = _detect_archive_ext(archive.name)
    if ext is None:
        error(f"Extensão não reconhecida: {archive.name}")
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)

    if ext == ".zip":
        _ensure_archive_tool(".zip")
        run(["unzip", "-q", "-o", str(archive), "-d", str(dest)])
        return
    if ext == ".rar":
        _ensure_archive_tool(".rar")
        run(["unrar", "x", "-o+", "-idq", str(archive), str(dest) + os.sep])
        return
    if ext == ".7z":
        _ensure_archive_tool(".7z")
        run(["7z", "x", f"-o{dest}", "-y", str(archive)],
            stdout=subprocess.DEVNULL)
        return
    if ext in (".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tbz2"):
        import tarfile
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest)
        return
    error(f"Extensão não suportada: {ext}")
    sys.exit(1)


def _ensure_megatools() -> str:
    """Return the binary name to download Mega URLs. Prefers `mega-get`
    (megacmd), falls back to `megadl` (megatools). Auto-installs megatools
    if neither is present."""
    if check_cmd("mega-get"):
        return "mega-get"
    if check_cmd("megadl"):
        return "megadl"
    info("Instalando megatools (download de Mega.nz)...")
    apt = "sudo apt-get" if check_cmd("sudo") and os.geteuid() != 0 else "apt-get"
    run(f"{apt} update -qq && {apt} install -y -qq megatools", check=False)
    if check_cmd("megadl"):
        return "megadl"
    error("Não consegui instalar megatools. Instale 'megatools' ou 'megacmd' manualmente.")
    sys.exit(1)


def _download_url(url: str, dest_dir: Path) -> Path:
    """Download an HTTP(S) URL or a Mega.nz link into dest_dir. Returns the
    local file path of the downloaded archive (or a directory if Mega
    delivers a folder)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _is_mega_url(url):
        tool = _ensure_megatools()
        info(f"Baixando do Mega via {tool}...")
        before = set(p.name for p in dest_dir.iterdir())
        if tool == "mega-get":
            run([tool, url, str(dest_dir)])
        else:
            run([tool, "--path", str(dest_dir), url])
        after = set(p.name for p in dest_dir.iterdir())
        new = after - before
        if not new:
            error("Mega download: nada apareceu em " + str(dest_dir))
            sys.exit(1)
        # Pick the largest new file/dir (Mega often creates the named entry).
        candidates = [dest_dir / n for n in new]
        candidates.sort(key=lambda p: p.stat().st_size if p.is_file() else 0, reverse=True)
        return candidates[0]

    # HTTP(S) URL — pick filename from URL or fall back to generic name.
    fname_match = re.search(r"/([^/?#]+)(?:[?#]|$)", url)
    fname = fname_match.group(1) if fname_match else "_dataset.bin"
    if not _detect_archive_ext(fname):
        fname += ".zip"  # default extension if URL doesn't disclose
    dest = dest_dir / fname
    info(f"Baixando {fname}...")
    if check_cmd("wget"):
        run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    elif check_cmd("curl"):
        run(["curl", "-L", "--progress-bar", "-o", str(dest), url])
    else:
        error("wget/curl não encontrado para baixar URL")
        sys.exit(1)
    return dest


def _flatten_to_images(source: Path, target: Path) -> Path:
    """Move all images (and their .txt) from nested dirs into a single flat target dir.
    If the source already has every image at the top level, just return it."""
    all_images = [f for f in source.rglob("*")
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    if not all_images:
        warn(f"Nenhuma imagem encontrada em {source}")
        return source
    top = sum(1 for f in source.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if top == len(all_images):
        success(f"{len(all_images)} imagens em {source.name}/")
        return source
    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for img in all_images:
        if img.parent == target:
            continue
        dest = target / img.name
        if dest.exists():
            dest = target / f"{img.parent.name}_{img.name}"
        if dest.exists():
            continue
        shutil.move(str(img), str(dest))
        moved += 1
        # Also move the matching .txt
        txt = img.with_suffix(".txt")
        if txt.exists():
            txt_dest = dest.with_suffix(".txt")
            if not txt_dest.exists():
                shutil.move(str(txt), str(txt_dest))
    if moved:
        success(f"Movidas {moved} imagens para {target.name}/")
    return target


def _resolve_archives_in_dir(root: Path, extracted_marker: Path) -> None:
    """Find any archive inside `root` that hasn't been extracted yet, extract
    it in-place (sibling dir), and mark it as done. Useful for HF snapshots
    that ship a .zip inside the repo."""
    for arc in list(root.rglob("*")):
        if not arc.is_file():
            continue
        ext = _detect_archive_ext(arc.name)
        if not ext:
            continue
        marker = arc.with_suffix(arc.suffix + ".extracted")
        if marker.exists():
            continue
        info(f"Extraindo arquivo encontrado: {arc.name}...")
        _extract_archive(arc, arc.parent)
        marker.touch()


def _download_dataset(source: str, project_data_dir: Path) -> Path:
    """Resolve a user-provided dataset source into a local directory of
    images+captions. Supports:
      - local folder path (returned as-is)
      - local archive file (.zip/.rar/.7z/.tar*) → extracted
      - HuggingFace repo (`user/repo`) → snapshot_download + extract any archive
      - HTTP(S) URL pointing to an archive → download + extract
      - Mega.nz URL → mega-get/megadl + extract if archive
    """
    # ── HuggingFace dataset repo
    if _is_hf_repo(source):
        info(f"Dataset HuggingFace detectado: [bold]{source}[/bold]")
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or None
        target = project_data_dir / "_hf_download"
        snapshot_download(source, local_dir=str(target), token=token)
        # HF datasets sometimes ship a single .zip — extract any archive in-place
        _resolve_archives_in_dir(target, target / ".extracted")
        return _flatten_to_images(target, project_data_dir / "images")

    # ── HTTP(S) or Mega URL
    if re.match(r"^https?://", source) or _is_mega_url(source):
        downloads_dir = project_data_dir / "_downloads"
        downloaded = _download_url(source, downloads_dir)
        ds = project_data_dir / "images"
        if downloaded.is_dir():
            # Mega delivered a folder
            return _flatten_to_images(downloaded, ds)
        ext = _detect_archive_ext(downloaded.name)
        if ext is None:
            error(f"Arquivo baixado não é um archive reconhecido: {downloaded.name}")
            sys.exit(1)
        info(f"Extraindo {downloaded.name} ({ext})...")
        _extract_archive(downloaded, ds)
        try:
            downloaded.unlink()
        except OSError:
            pass
        return _flatten_to_images(ds, ds)

    # ── Local path (file or directory)
    p = Path(source).expanduser()
    if not p.exists():
        error(f"Caminho não existe: {p}")
        sys.exit(1)
    if p.is_file():
        ext = _detect_archive_ext(p.name)
        if ext is None:
            error(f"Arquivo local não é um archive reconhecido: {p.name}")
            sys.exit(1)
        info(f"Extraindo archive local {p.name} ({ext})...")
        ds = project_data_dir / "images"
        _extract_archive(p, ds)
        return _flatten_to_images(ds, ds)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# animatrem — grouped ingest (preserve per-outfit subfolders) + captioning
# ═══════════════════════════════════════════════════════════════════════════

def _slugify(name: str) -> str:
    """Trigger-safe slug: lowercase, ascii, underscores. Preserve existing
    underscores; collapse other separators."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip()).strip("_")
    s = re.sub(r"_+", "_", s).lower()
    return s or "character"


# huggingface.co/<repo>/resolve/<rev>/<path>  (optionally datasets/ or spaces/)
_HF_FILE_RE = re.compile(
    r"^https?://huggingface\.co/(?:(?P<dtype>datasets|spaces)/)?"
    r"(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>[^?#]+)"
)


def _parse_hf_file_url(url: str):
    """Parse an HF `resolve` file URL → (repo_id, repo_type, revision, path).
    Returns None if it's not an HF file URL."""
    m = _HF_FILE_RE.match(url)
    if not m:
        return None
    dtype = m.group("dtype")
    repo_type = {"datasets": "dataset", "spaces": "space"}.get(dtype, "model")
    from urllib.parse import unquote
    return m.group("repo"), repo_type, m.group("rev"), unquote(m.group("path"))


def _download_raw(source: str, dest_dir: Path) -> Path:
    """Resolve a single input source into a local root dir WITHOUT flattening
    across subfolders (so per-outfit folders survive). Local dirs are copied
    into staging so the user's original data is never mutated."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if _is_hf_repo(source):
        info(f"Dataset HuggingFace detectado: [bold]{escape(source)}[/bold]")
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or None
        snapshot_download(source, local_dir=str(dest_dir), token=token)
        _resolve_archives_in_dir(dest_dir, dest_dir / ".extracted")
        return dest_dir

    if re.match(r"^https?://", source) or _is_mega_url(source):
        # HF resolve URL → download with the token (works for private/gated).
        hf = _parse_hf_file_url(source)
        if hf is not None:
            repo_id, repo_type, rev, path_in_repo = hf
            from huggingface_hub import hf_hub_download
            token = os.environ.get("HF_TOKEN") or None
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
            info(f"Baixando de HF ({repo_type}): [bold]{escape(repo_id)}[/bold] : {escape(path_in_repo)}")
            local = Path(hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                        revision=rev, repo_type=repo_type,
                                        token=token))
            ext = _detect_archive_ext(local.name)
            if ext:
                info(f"Extraindo {local.name} ({ext})...")
                _extract_archive(local, dest_dir)
                return dest_dir
            # Not an archive: use the containing dir as the source root.
            return local.parent

        downloaded = _download_url(source, dest_dir / "_dl")
        if downloaded.is_dir():
            return downloaded
        ext = _detect_archive_ext(downloaded.name)
        if ext is None:
            error(f"Arquivo baixado não é um archive reconhecido: {downloaded.name}")
            sys.exit(1)
        info(f"Extraindo {downloaded.name} ({ext})...")
        _extract_archive(downloaded, dest_dir)
        try:
            downloaded.unlink()
        except OSError:
            pass
        return dest_dir

    p = Path(source).expanduser()
    if not p.exists():
        error(f"Caminho não existe: {p}")
        sys.exit(1)
    if p.is_file():
        ext = _detect_archive_ext(p.name)
        if ext is None:
            error(f"Arquivo local não é um archive reconhecido: {p.name}")
            sys.exit(1)
        info(f"Extraindo archive local {p.name} ({ext})...")
        _extract_archive(p, dest_dir)
        return dest_dir
    # Local directory → copy into staging (never mutate the user's folder).
    local_copy = dest_dir / "local"
    if not local_copy.exists():
        info(f"Copiando pasta local para staging: {escape(str(p))}")
        shutil.copytree(p, local_copy)
    return local_copy


def _descend_single_wrapper(root: Path) -> Path:
    """Descend through single-wrapper dirs (archives often wrap everything in
    one top folder). Stops when a dir has top-level images or ≠1 real subdir."""
    cur = root
    for _ in range(8):
        if not cur.is_dir():
            return cur
        top_imgs = any(p.is_file() and p.suffix.lower() in IMAGE_EXTS
                       for p in cur.iterdir())
        subdirs = [d for d in cur.iterdir()
                   if d.is_dir() and not d.name.startswith((".", "_"))]
        if top_imgs or len(subdirs) != 1:
            return cur
        cur = subdirs[0]
    return cur


def _flatten_within(group_dir: Path) -> None:
    """Move nested images (+ matching .txt) up INTO group_dir so the group is a
    flat folder. Stays within one group — never merges across groups."""
    imgs = [f for f in group_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    for img in imgs:
        if img.parent == group_dir:
            continue
        dest = group_dir / img.name
        if dest.exists():
            dest = group_dir / f"{img.parent.name}_{img.name}"
        if dest.exists():
            continue
        shutil.move(str(img), str(dest))
        txt = img.with_suffix(".txt")
        if txt.exists() and not dest.with_suffix(".txt").exists():
            shutil.move(str(txt), str(dest.with_suffix(".txt")))


def detect_groups(root: Path) -> list[Path]:
    """Return the list of GROUP dirs from a downloaded root, preserving the
    top-level subfolder split (one group per outfit folder). Each returned dir
    is flattened within itself. Operates only inside our staging."""
    root = _descend_single_wrapper(root)
    if not root.is_dir():
        return []
    direct = any(p.is_file() and p.suffix.lower() in IMAGE_EXTS
                 for p in root.iterdir())
    if direct:
        _flatten_within(root)
        return [root]
    subdirs = sorted(d for d in root.iterdir()
                     if d.is_dir() and not d.name.startswith((".", "_"))
                     and count_images(d) > 0)
    if subdirs:
        for d in subdirs:
            _flatten_within(d)
        return subdirs
    if count_images(root) > 0:  # images nested but no clean subdir split
        _flatten_within(root)
        return [root]
    return []


def _install_custom_profiles() -> None:
    """Copy animatrem's custom prompt profiles into the captioner's prompts
    tree so `--prompt_profile anima-character-outfit` resolves."""
    src_root = CUSTOM_PROMPTS_DIR
    dst_root = CAPTIONER_DIR / "prompts"
    if not src_root.exists():
        warn(f"Prompts custom não encontrados em {src_root}")
        return
    for prof_dir in (src_root / "image").glob("*"):
        if not prof_dir.is_dir():
            continue
        dst = dst_root / "image" / prof_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in prof_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    success(f"Profiles custom instalados em {dst_root / 'image'}")


def _run_captioner_on_group(cfg: "TrainingConfig", group: dict) -> None:
    """Run PixAI + Gemini-Flash captioning on one group folder. Writes .txt
    next to each image (the caption LLM output)."""
    group_dir = Path(group["path"])
    is_outfit = bool(group.get("is_outfit"))
    if is_outfit:
        profile = (PROFILE_CHARACTER_OUTFIT if getattr(cfg, "outfit_lock", True)
                   else PROFILE_CHARACTER_OUTFIT_DESCRIBED)
    else:
        profile = PROFILE_CHARACTER
    cmd = [
        sys.executable, str(CAPTIONER_SCRIPT), str(group_dir),
        "--taggers", "pixai,grok",
        "--grok_provider", "openrouter",
        "--grok_model", CAPTION_MODEL,
        "--prompt_profile", profile,
        "--prompt_var", f"trigger_character={cfg.trigger_character}",
        "--recursive", "--force", "--remove_underscore",
        "--thresh", "0.30",
    ]
    if is_outfit:
        cmd += ["--prompt_var", f"trigger_outfit={group['trigger_outfit']}"]
        # Always pass custom_instruction (empty ok) so the profile placeholder
        # never resolves to "missing variable". Subprocess list-form → free
        # text (quotes/braces/newlines) is passed verbatim, no shell escaping.
        cmd += ["--prompt_var",
                f"custom_instruction={group.get('custom_instruction', '')}"]
    # check=False: a partial failure on one group shouldn't abort the run; the
    # caption sanity check (phase6) + per-group count below surface problems.
    run(cmd, cwd=str(CAPTIONER_DIR), check=False)


def _sample_captions(group_dir: Path, n: int = 3) -> list[str]:
    """Read up to n non-empty captions from a group, for the HF model card."""
    out: list[str] = []
    for txt in sorted(group_dir.glob("*.txt")):
        try:
            content = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if content:
            out.append(content)
        if len(out) >= n:
            break
    return out


def _count_nonempty_captions(group_dir: Path) -> int:
    """Count .txt files in a group that exist and are non-empty."""
    total = 0
    for txt in group_dir.glob("*.txt"):
        try:
            if txt.read_text(encoding="utf-8", errors="replace").strip():
                total += 1
        except OSError:
            continue
    return total


def _summarize_groups(cfg: "TrainingConfig") -> None:
    t = Table(title="Grupos detectados", show_header=True, header_style="bold",
              box=None, padding=(0, 2))
    t.add_column("#", style="bold"); t.add_column("Pasta")
    t.add_column("Imgs"); t.add_column("Tipo"); t.add_column("Trigger")
    for i, g in enumerate(cfg.groups, 1):
        tipo = "outfit" if g["is_outfit"] else "personagem (base)"
        trig = g["trigger_outfit"] if g["is_outfit"] else cfg.trigger_character
        t.add_row(str(i), Path(g["path"]).name, str(g["image_count"]), tipo, trig)
    console.print(t)
    console.print()


SCHED_NUM_REPEATS = 10       # repeat each image 10× per epoch (same all groups)
SCHED_SAVE_DIVISOR = 10      # base: ~1 save per 1/10 of the run, before batch


def _micro_batch_size(cfg: "TrainingConfig") -> int:
    """Effective micro_batch_size_per_gpu (min across resolutions if mixed)."""
    mb = cfg.micro_batch_size
    if isinstance(mb, int):
        return max(1, mb)
    if mb:
        return max(1, min(b for _, b in mb))
    return 1


def _save_every_n_steps(total_steps: int, micro_batch: int) -> int:
    """save_every_n_steps that ACCOUNTS FOR micro_batch: a bigger batch moves
    more per step, so the sweet spot passes in fewer steps → save more often.
    Base interval = total_steps // SCHED_SAVE_DIVISOR, then divided by the
    micro_batch (the "4"). So batch 4 saves 4× more often than batch 1."""
    base = max(1, total_steps // SCHED_SAVE_DIVISOR)
    return max(1, base // max(1, micro_batch))


def _compute_group_schedule(cfg: "TrainingConfig", recipe: dict,
                            eff_batch: int) -> None:
    """Simple, native diffusion-pipe schedule (no reinvented cadence):

      num_repeats = SCHED_NUM_REPEATS for every group;
      epochs derived so exposures/image = epochs × num_repeats × n_res hits the
      recipe target (~660, the validated sweet spot);
      total_steps = epochs × (Σ images × num_repeats × n_res) / eff_batch;
      save_every_n_steps = (total_steps // 10) // micro_batch  — the engine's
      native `save_every_n_steps` (dirs named step<N>/), made denser for larger
      batches so a fast-moving run still gets fine-grained checkpoints early.
    """
    n_res = max(1, len(cfg.resolutions))

    cfg.num_repeats = SCHED_NUM_REPEATS
    for g in cfg.groups:
        g["num_repeats"] = SCHED_NUM_REPEATS

    target_exposures = recipe["exposures_per_image"]
    cfg.exposures_per_image = target_exposures
    cfg.epochs = max(1, round(target_exposures / (SCHED_NUM_REPEATS * n_res)))

    total_images = sum(max(0, int(g.get("image_count", 0))) for g in cfg.groups)
    samples_per_epoch = max(1, total_images * SCHED_NUM_REPEATS * n_res)
    steps_per_epoch = max(1, samples_per_epoch // max(1, eff_batch))
    cfg.total_steps = steps_per_epoch * cfg.epochs

    # Native step-based saving, accounting for micro_batch (the "4").
    cfg.save_every_n_steps = _save_every_n_steps(cfg.total_steps,
                                                 _micro_batch_size(cfg))
    cfg.eval_every_n_steps = cfg.save_every_n_steps
    cfg.save_every_n_epochs = 0   # step cadence drives this path
    cfg.eval_every_n_epochs = 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Configure dataset + recipe
# ═══════════════════════════════════════════════════════════════════════════

def _apply_recipe(cfg: TrainingConfig, recipe_key: str) -> dict:
    """Copy recipe values into cfg. Returns the recipe dict for follow-up use."""
    recipe = RECIPES[recipe_key]
    cfg.recipe              = recipe_key
    cfg.rank                = recipe["rank"]
    cfg.learning_rate       = recipe["lr"]
    cfg.betas               = list(recipe["betas"])
    cfg.weight_decay        = recipe["weight_decay"]
    cfg.eps                 = recipe["eps"]
    cfg.loss                = recipe["loss"]
    cfg.pseudo_huber_c      = recipe["pseudo_huber_c"]
    cfg.exposures_per_image = recipe["exposures_per_image"]
    cfg.warmup_steps        = recipe["warmup_steps"]
    cfg.lr_scheduler        = recipe["lr_scheduler"]
    cfg.self_attn_lr        = recipe["self_attn_lr"]
    cfg.cross_attn_lr       = recipe["cross_attn_lr"]
    cfg.mlp_lr              = recipe["mlp_lr"]
    cfg.mod_lr              = recipe["mod_lr"]
    cfg.caption_prefix      = recipe["caption_prefix"]
    cfg.resolutions         = list(recipe["default_resolutions"])
    return recipe


def _collect_dataset_sources(cfg: TrainingConfig, prompt_text: str) -> None:
    """Collect dataset sources up-front (comma- or newline-separated)."""
    console.print("  [bold]Fontes de dataset[/bold]")
    console.print("  [dim]Aceita: caminho local (pasta ou archive), URL .zip/.rar/.tar/.7z,[/dim]")
    console.print("  [dim]link Mega.nz, repo HuggingFace (`user/repo`).[/dim]")
    console.print("  [dim]Múltiplas fontes: separe com vírgula.[/dim]")
    console.print()

    project_data_dir = cfg.project_dir / "data"
    project_data_dir.mkdir(parents=True, exist_ok=True)

    sources_str = ask(prompt_text, "/workspace/dataset", allow_empty=False)
    raw_sources = [s.strip() for s in re.split(r"[,\n]", sources_str) if s.strip()]
    if not raw_sources:
        error("Nenhuma fonte fornecida.")
        sys.exit(1)

    cfg.dataset_dirs = []
    for idx, src in enumerate(raw_sources, 1):
        target = project_data_dir if idx == 1 else project_data_dir / f"source_{idx:02d}"
        target.mkdir(parents=True, exist_ok=True)
        ds = _download_dataset(src, target)
        cfg.dataset_dirs.append(str(ds))
        n_imgs = count_images(ds)
        n_txts = count_captions(ds)
        success(f"Fonte #{idx}: {n_imgs} imagens / {n_txts} captions em {escape(str(ds))}")

    cfg.image_count   = total_image_count(cfg)
    cfg.caption_count = total_caption_count(cfg)
    success(f"Total: {cfg.image_count} imagens / {cfg.caption_count} captions em "
            f"{len(get_dataset_entries(cfg))} bloco(s)")
    if cfg.image_count == 0:
        error("Nenhuma imagem encontrada nas fontes.")
        sys.exit(1)


def _compute_schedule(cfg: TrainingConfig, recipe: dict, eff_batch: int) -> None:
    """Auto-compute epochs/num_repeats from the dataset size + recipe target.

    Each diffusion-pipe "epoch" iterates the dataset once per resolution. So
    mixed [768,1024,1536] → 1 epoch = 3 passes/img.

    For huge datasets (>5k imgs) we cap total passes to 500k to avoid burning
    weeks of GPU time on style/concept LoRAs.
    """
    n_res = max(1, len(cfg.resolutions))
    target_exposures = recipe["exposures_per_image"]

    if cfg.image_count > 5000:
        capped = max(10, 500_000 // cfg.image_count)
        if capped < target_exposures:
            info(f"Dataset gigante ({cfg.image_count} imgs): reduzindo "
                 f"exposures/img de {target_exposures} → {capped} "
                 f"(orçamento ~500k passes total).")
            target_exposures = capped
    cfg.exposures_per_image = target_exposures

    # eff_batch is in optimizer steps per pass-of-the-dataset. Each epoch =
    # n_res passes. We want at least ~100 optimizer steps per epoch for a
    # reasonable eval cadence. If the dataset is small, raise num_repeats.
    steps_per_epoch_at_r1 = max(1, (cfg.image_count * n_res) // max(1, eff_batch))
    if steps_per_epoch_at_r1 >= 100:
        cfg.num_repeats = 1
    else:
        cfg.num_repeats = min(recipe["max_repeats"],
                              max(recipe["min_repeats"],
                                  200 // max(1, steps_per_epoch_at_r1)))

    passes_per_epoch_per_img = cfg.num_repeats * n_res
    cfg.epochs = max(1, target_exposures // max(1, passes_per_epoch_per_img))

    # Save cadence — a HANDFUL of checkpoints to cherry-pick from, NOT one per
    # epoch. Research (lilting Receita C) saves every ~10 epochs on a 150-epoch
    # run (~15 saves); scaled to our shorter character runs that's ~1 save per
    # 5 epochs → ~TARGET_SAVES checkpoints. The best epoch is mid-to-late in the
    # validated window, so you compare the last few.
    #   - LoRA (rank > 0): ~TARGET_SAVES checkpoints total (~140 MB each).
    #   - FFT (rank == 0): whole 2B model each time (~4 GB) → cap at ~5.
    TARGET_SAVES = 11
    if cfg.rank > 0:
        cfg.save_every_n_epochs = max(1, round(cfg.epochs / TARGET_SAVES))
    else:
        cfg.save_every_n_epochs = max(1, cfg.epochs // 5)
    # Sample/eval in lockstep with saves (research: sample_every == save_every)
    # so each checkpoint has a matching preview, without extra eval overhead.
    cfg.eval_every_n_epochs = cfg.save_every_n_epochs


def _apply_gpu_profile(cfg: TrainingConfig, profile: dict, recipe: dict,
                       fp8: bool) -> None:
    """Materialize batching + memory knobs from profile + recipe."""
    cfg.activation_checkpointing = profile["activation_checkpointing"]
    cfg.pipeline_stages          = profile["pipeline_stages"]
    cfg.blocks_to_swap           = profile["blocks_to_swap"]
    cfg.use_fp8                  = fp8
    cfg.transformer_dtype        = "float8" if fp8 else "bfloat16"

    micro, gas, info_str = _compute_batching(
        profile, cfg.resolutions, recipe["target_global_batch"]
    )
    cfg.micro_batch_size            = micro
    cfg.gradient_accumulation_steps = gas
    info(f"Batch: {info_str}")


def phase4_configure(cfg: TrainingConfig, env: dict,
                     transformer_path: Path, vae_path: Path, llm_path: Path) -> TrainingConfig:
    header("FASE 4 — Configuração do treinamento")

    cfg.transformer_path = str(transformer_path)
    cfg.vae_path         = str(vae_path)
    cfg.llm_path         = str(llm_path)

    detected_profile_key = env.get("gpu_profile", "GENERIC")
    detected_profile = GPU_PROFILES[detected_profile_key]

    # ── Mode selection (top-level choice) ───────────────────────────
    console.print("  [bold]Modo de treinamento[/bold]")
    console.print("  [dim]Modos 1-4 = preset pronto (poucas perguntas). Modo 5 = customizado.[/dim]")
    console.print()
    recipe_keys = ["character", "style", "concept", "full_finetune", "custom"]
    mode_choices = {
        str(i + 1): f"{RECIPES[k]['label']}\n      [dim]{RECIPES[k]['description']}[/dim]"
        for i, k in enumerate(recipe_keys)
    }
    mode_default = "1"
    mchoice = ask_choice("Escolha o modo", mode_choices, mode_default)
    recipe_key = recipe_keys[int(mchoice) - 1]
    is_custom = (recipe_key == "custom")
    recipe = _apply_recipe(cfg, recipe_key)
    success(f"Modo: {recipe['label']}")

    # ── Project name ────────────────────────────────────────────────
    console.print(Rule(style="dim"))
    default_name_by_mode = {
        "character":     "anima-character-lora",
        "style":         "anima-style-lora",
        "concept":       "anima-concept-lora",
        "full_finetune": "anima-fft",
        "custom":        "anima-custom-lora",
    }
    cfg.project_name = ask("Nome do projeto",
                           default_name_by_mode[recipe_key], allow_empty=False)
    cfg.project_dir.mkdir(parents=True, exist_ok=True)

    # ── Continue-from-existing (only in custom mode; preset = from scratch) ─
    if is_custom:
        console.print(Rule(style="dim"))
        console.print("  [bold]Continuar de um LoRA existente?[/bold]")
        console.print("  [dim]Carrega pesos de uma LoRA anterior como ponto de partida.[/dim]")
        base_lora = ask("Caminho .safetensors da LoRA base", "")
        if base_lora and Path(base_lora).exists():
            cfg.base_lora_path = base_lora
            success(f"LoRA base: {escape(base_lora)}")
        else:
            cfg.base_lora_path = ""
            if base_lora:
                warn(f"Não encontrado: {escape(base_lora)} — treinando do zero")
    else:
        cfg.base_lora_path = ""

    # ── Dataset sources ─────────────────────────────────────────────
    console.print(Rule(style="dim"))
    _collect_dataset_sources(cfg, "Fonte(s) de dataset")

    # ── Caption prefix: NÃO usado ───────────────────────────────────
    # O captioner do data_araknideo já injeta o trigger DIRETAMENTE no texto
    # da caption (anima-style começa com `@{trigger_style}.`; anima-character
    # tece `{trigger_character}` como sujeito; anima-concept não usa trigger).
    # Se o trainer também injetasse via `caption_prefix` no dataset.toml, o
    # trigger entraria duplicado nas captions de treino — bug silencioso de
    # qualidade. Por isso `caption_prefix` permanece vazio em todos os modos.
    cfg.caption_prefix = ""

    # ── Custom mode: ask the rest ───────────────────────────────────
    if is_custom:
        console.print(Rule(style="dim"))
        console.print("  [bold]Resolução(ões) de treino[/bold]")
        console.print("  [dim]Mixed = cada resolução vira um pass do dataset por epoch.[/dim]")
        res_choices = {k: f"{label}  [dim]— {desc}[/dim]"
                       for k, (label, _list, desc) in RESOLUTION_PRESETS.items()}
        res_choice = ask_choice("Escolha", res_choices, "1")
        _label, res_list, _desc = RESOLUTION_PRESETS[res_choice]
        cfg.resolutions = res_list
        success(f"Resoluções: {res_list}")

        console.print(Rule(style="dim"))
        console.print("  [bold]Hiperparâmetros[/bold]")
        cfg.rank          = int(ask("rank (0 = full finetune sem [adapter])",
                                    str(cfg.rank), allow_empty=False))
        cfg.learning_rate = float(ask("learning rate",
                                      f"{cfg.learning_rate}", allow_empty=False))
        sched_choices = {"1": "constant (tdrussell oficial)",
                         "2": "cosine", "3": "linear"}
        sched_map = {"1": "constant", "2": "cosine", "3": "linear"}
        cfg.lr_scheduler = sched_map[ask_choice("lr_scheduler", sched_choices, "1")]
        cfg.warmup_steps = int(ask("warmup_steps",
                                   str(cfg.warmup_steps), allow_empty=False))
        loss_choices = {"1": "MSE (rectified flow padrão)",
                        "2": "huber_delta",
                        "3": "smooth_l1_beta"}
        lc = ask_choice("Loss", loss_choices, "1")
        if lc == "1":
            cfg.loss = "mse"; cfg.pseudo_huber_c = None
        elif lc == "2":
            cfg.loss = "huber_delta"
            cfg.pseudo_huber_c = float(ask("huber_delta value", "1.0",
                                           allow_empty=False))
        else:
            cfg.loss = "smooth_l1_beta"
            cfg.pseudo_huber_c = float(ask("smooth_l1_beta value", "0.5",
                                           allow_empty=False))

        if ask_yn("Configurar LRs por componente (self_attn/cross_attn/mlp/mod)?",
                 default=False):
            cfg.self_attn_lr  = float(ask("self_attn_lr",  f"{cfg.learning_rate}"))
            cfg.cross_attn_lr = float(ask("cross_attn_lr", f"{cfg.learning_rate}"))
            cfg.mlp_lr        = float(ask("mlp_lr",        f"{cfg.learning_rate}"))
            cfg.mod_lr        = float(ask("mod_lr",        f"{cfg.learning_rate}"))

        target_exp = int(ask("Exposições por imagem (target)",
                             str(recipe["exposures_per_image"]), allow_empty=False))
        recipe = dict(recipe, exposures_per_image=target_exp)
        cfg.exposures_per_image = target_exp

    # ── GPU profile / precision (auto, override only in custom) ─────
    console.print(Rule(style="dim"))
    info(f"GPU detectada → perfil: [bold]{detected_profile['label']}[/bold]")
    profile = detected_profile
    profile_key = detected_profile_key
    if is_custom:
        profile_keys = list(GPU_PROFILES.keys())
        profile_choices = {}
        for i, k in enumerate(profile_keys, 1):
            marker = " [bold green](sugerido)[/bold green]" if k == detected_profile_key else ""
            profile_choices[str(i)] = f"{GPU_PROFILES[k]['label']}{marker}"
        p_default = str(profile_keys.index(detected_profile_key) + 1)
        pchoice = ask_choice("Trocar de perfil de GPU?", profile_choices, p_default)
        profile_key = profile_keys[int(pchoice) - 1]
        profile = GPU_PROFILES[profile_key]

    # Precision: bf16 default unless profile or user prefers fp8.
    fp8 = profile["fp8_default"]
    if is_custom:
        fp8_choices = {
            "1": "bf16  [dim]— Qualidade máxima (default tdrussell)[/dim]",
            "2": "fp8   [dim]— ~½ VRAM, perda pequena de qualidade[/dim]",
        }
        fp8 = ask_choice("Precisão do transformer", fp8_choices,
                         "2" if profile["fp8_default"] else "1") == "2"
    success(f"Precisão: {'fp8' if fp8 else 'bf16'} (qualidade {'reduzida' if fp8 else 'máxima'})")

    _apply_gpu_profile(cfg, profile, recipe, fp8)

    # ── Custom: optional batch override ─────────────────────────────
    if is_custom:
        console.print(Rule(style="dim"))
        console.print("  [bold]Batch override (opcional)[/bold]")
        info(f"Auto: micro={cfg.micro_batch_size}, "
             f"grad_accum={cfg.gradient_accumulation_steps}")
        if ask_yn("Personalizar batch?", default=False):
            if len(cfg.resolutions) > 1:
                console.print("  [dim]Mixed-res: digite 1 valor (igual em todas) "
                              "ou N valores separados por vírgula.[/dim]")
                console.print(f"  [dim]Resoluções: {cfg.resolutions}[/dim]")
                mb_input = ask(
                    "micro_batch_size_per_gpu",
                    "4" if isinstance(cfg.micro_batch_size, int)
                    else ",".join(str(b) for _, b in cfg.micro_batch_size),
                )
                parts = [p.strip() for p in mb_input.split(",")]
                if len(parts) == len(cfg.resolutions):
                    cfg.micro_batch_size = [[r, int(b)]
                                            for r, b in zip(cfg.resolutions, parts)]
                elif len(parts) == 1:
                    cfg.micro_batch_size = [[r, int(parts[0])]
                                            for r in cfg.resolutions]
            else:
                cfg.micro_batch_size = int(ask(
                    "micro_batch_size_per_gpu",
                    str(cfg.micro_batch_size if isinstance(cfg.micro_batch_size, int)
                        else cfg.micro_batch_size[0][1]),
                ))
            cfg.gradient_accumulation_steps = int(ask(
                "gradient_accumulation_steps",
                str(cfg.gradient_accumulation_steps),
            ))

    # Recompute effective batch for schedule
    if isinstance(cfg.micro_batch_size, int):
        eff_micro = cfg.micro_batch_size
    else:
        # Mixed-res: use min (the constraining factor for grad_accum).
        eff_micro = min(b for _, b in cfg.micro_batch_size)
    eff_batch = eff_micro * cfg.gradient_accumulation_steps
    success(f"Effective batch: ~{eff_batch}")

    # ── Auto-derive schedule ────────────────────────────────────────
    _compute_schedule(cfg, recipe, eff_batch)

    n_res = max(1, len(cfg.resolutions))
    total_passes = cfg.epochs * cfg.num_repeats * n_res * cfg.image_count
    total_steps  = total_passes // max(1, eff_batch)
    info(f"Cronograma: epochs={cfg.epochs} × repeats={cfg.num_repeats} × {n_res} res = "
         f"{cfg.epochs * cfg.num_repeats * n_res} passes/img → "
         f"~{total_passes:,} exposições totais ≈ {total_steps:,} steps")

    if is_custom and ask_yn("Ajustar epochs/repeats/save_cadence manualmente?",
                            default=False):
        cfg.epochs              = int(ask("Epochs",
                                          str(cfg.epochs), allow_empty=False))
        cfg.num_repeats         = int(ask("num_repeats",
                                          str(cfg.num_repeats), allow_empty=False))
        cfg.save_every_n_epochs = int(ask("save_every_n_epochs",
                                          str(cfg.save_every_n_epochs),
                                          allow_empty=False))
        cfg.eval_every_n_epochs = int(ask("eval_every_n_epochs",
                                          str(cfg.eval_every_n_epochs),
                                          allow_empty=False))

    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# animatrem — minimal wizard (project · trigger · input · per-outfit)
# ═══════════════════════════════════════════════════════════════════════════

def phase_wizard(cfg: TrainingConfig, env: dict, transformer_path: Path,
                 vae_path: Path, llm_path: Path) -> TrainingConfig:
    header("CONFIGURAÇÃO — personagem, imagens e outfits")

    cfg.transformer_path = str(transformer_path)
    cfg.vae_path         = str(vae_path)
    cfg.llm_path         = str(llm_path)

    # Locked recipe: character (rank 32, MSE, mixed [768,1024], ~720 expos/img).
    recipe = _apply_recipe(cfg, "character")

    # 1) Project / character name → HF repo, folder, safetensors name.
    cfg.project_name = ask("Nome do projeto / personagem (ex.: aria)", "",
                           allow_empty=False).strip()
    cfg.project_dir.mkdir(parents=True, exist_ok=True)

    # 2) Character trigger — shared across all groups, placed at caption start.
    default_trigger = _slugify(cfg.project_name)
    cfg.trigger_character = _slugify(
        ask("Trigger word do personagem", default_trigger, allow_empty=False))
    success(f"Trigger do personagem: [bold]{cfg.trigger_character}[/bold]")
    cfg.save()  # persist early so the project is resumable from here on

    # 3) Single input source.
    console.print(Rule(style="dim"))
    console.print("  [dim]Aceita: pasta local, archive .zip/.rar/.7z/.tar, URL,[/dim]")
    console.print("  [dim]link Mega.nz, repo HuggingFace (`user/repo`).[/dim]")
    console.print("  [dim]Uma pasta = personagem base. Várias subpastas = 1 por outfit.[/dim]")
    console.print()
    source = ask("Link/caminho das imagens", allow_empty=False).strip()

    raw_root = _download_raw(source, cfg.project_dir / "data" / "_raw")
    group_dirs = detect_groups(raw_root)
    if not group_dirs:
        error("Nenhuma imagem encontrada na fonte.")
        sys.exit(1)

    # 4) Groups from folder structure — NO typing. Multiple subfolders → each is
    #    an outfit whose TRIGGER is the folder name (described mode, Anima path).
    #    Single folder → base character. The LLM instruction is auto-built.
    cfg.groups = []
    cfg.dataset_dirs = []
    cfg.outfit_lock = False  # described mode = the correct Anima path
    multi = len(group_dirs) > 1
    if multi:
        console.print(Rule(style="dim"))
        info(f"Detectei [bold]{len(group_dirs)}[/bold] subpastas → cada uma é um "
             f"outfit; o nome da pasta vira o trigger.")
    else:
        info("Uma pasta única → personagem base (sem outfit).")

    for gd in group_dirs:
        n = count_images(gd)
        if multi:
            trig = _slugify(gd.name)
            instruction = (
                f"The outfit shown in this set is named \"{trig}\". Include the "
                f"token \"{trig}\" right before you describe the clothing "
                f"(e.g. \"{trig}, <clothing details>\"), and keep describing the "
                f"outfit normally.")
            g: dict = {"path": str(gd), "image_count": n, "is_outfit": True,
                       "trigger_outfit": trig, "custom_instruction": instruction,
                       "num_repeats": 1, "caption_examples": []}
        else:
            g = {"path": str(gd), "image_count": n, "is_outfit": False,
                 "trigger_outfit": "", "custom_instruction": "",
                 "num_repeats": 1, "caption_examples": []}
        cfg.groups.append(g)
        cfg.dataset_dirs.append(str(gd))

    cfg.image_count    = sum(g["image_count"] for g in cfg.groups)
    cfg.caption_count  = 0
    cfg.caption_prefix = ""  # trigger baked into each caption by the captioner
    console.print()
    _summarize_groups(cfg)

    # 5) GPU profile (auto) + batching + balanced schedule — no questions.
    detected_profile_key = env.get("gpu_profile", "GENERIC")
    profile = GPU_PROFILES[detected_profile_key]
    info(f"GPU → perfil: [bold]{profile['label']}[/bold]")
    fp8 = profile["fp8_default"]
    _apply_gpu_profile(cfg, profile, recipe, fp8)
    success(f"Precisão: {'fp8' if fp8 else 'bf16'}")

    if isinstance(cfg.micro_batch_size, int):
        eff_micro = cfg.micro_batch_size
    else:
        eff_micro = min(b for _, b in cfg.micro_batch_size)
    eff_batch = eff_micro * cfg.gradient_accumulation_steps

    _compute_group_schedule(cfg, recipe, eff_batch)
    info(f"Cronograma: num_repeats={cfg.num_repeats} · epochs={cfg.epochs} · "
         f"~{cfg.total_steps} steps · salva a cada {cfg.save_every_n_steps} steps "
         f"(~{max(1, cfg.total_steps // cfg.save_every_n_steps)} checkpoints) · "
         f"res={cfg.resolutions}")

    cfg.save()  # persist full wizard result → resume skips all of the above
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Summary
# ═══════════════════════════════════════════════════════════════════════════

def phase5_summary(cfg: TrainingConfig, env: dict, skip_confirm: bool = False) -> None:
    header("FASE 5 — Resumo da configuração")

    pdir = cfg.project_dir
    if not cfg.output_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        cfg.output_dir = str(pdir / "outputs" / ts)
    if not cfg.dataset_toml_path:
        cfg.dataset_toml_path = str(pdir / "dataset.toml")
    if not cfg.config_toml_path:
        cfg.config_toml_path = str(pdir / "config.toml")

    cfg.image_count   = total_image_count(cfg)
    cfg.caption_count = total_caption_count(cfg)

    eff_micro = cfg.micro_batch_size if isinstance(cfg.micro_batch_size, int) \
        else max(b for _, b in cfg.micro_batch_size)
    eff_batch = eff_micro * cfg.gradient_accumulation_steps
    total_steps_est = cfg.total_steps if cfg.total_steps > 0 else (
        (cfg.image_count * cfg.num_repeats * max(1, len(cfg.resolutions))
         * cfg.epochs) // max(1, eff_batch))

    t1 = Table(title="Dataset", show_header=False, box=None, padding=(0, 2))
    t1.add_column(style="bold"); t1.add_column()
    t1.add_row("Fontes",   str(len(cfg.dataset_dirs)))
    t1.add_row("Imagens",  str(cfg.image_count))
    t1.add_row("Captions", str(cfg.caption_count))
    console.print(t1)
    console.print()

    t2 = Table(title="Treinamento (diffusion-pipe / Anima)", show_header=False, box=None, padding=(0, 2))
    t2.add_column(style="bold"); t2.add_column()
    t2.add_row("Receita",            f"{cfg.recipe} — {RECIPES[cfg.recipe]['label']}")
    t2.add_row("Resoluções",         str(cfg.resolutions))
    t2.add_row("Rank",               str(cfg.rank) if cfg.rank > 0 else "0 (FULL FINETUNE)")
    t2.add_row("Learning rate",      f"{cfg.learning_rate}")
    t2.add_row("Optimizer",          f"{cfg.optimizer_type} betas={cfg.betas} wd={cfg.weight_decay}")
    t2.add_row("Loss",               f"{cfg.loss}" + (f" (c={cfg.pseudo_huber_c})" if cfg.pseudo_huber_c else ""))
    t2.add_row("Scheduler",          f"{cfg.lr_scheduler} (warmup={cfg.warmup_steps})")
    t2.add_row("sigmoid_scale",      str(cfg.sigmoid_scale))
    t2.add_row("timestep_sample",    cfg.timestep_sample_method)
    t2.add_row("llm_adapter_lr",     f"{cfg.llm_adapter_lr} (mandatory 0)")
    if any(x is not None for x in [cfg.self_attn_lr, cfg.cross_attn_lr, cfg.mlp_lr, cfg.mod_lr]):
        t2.add_row("Per-component LR",
                   f"sa={cfg.self_attn_lr} ca={cfg.cross_attn_lr} mlp={cfg.mlp_lr} mod={cfg.mod_lr}")
    t2.add_row("micro_batch",        str(cfg.micro_batch_size))
    t2.add_row("grad_accum",         f"{cfg.gradient_accumulation_steps} (efetivo ~{eff_batch})")
    t2.add_row("Epochs × repeats",   f"{cfg.epochs} × {cfg.num_repeats}  (~{total_steps_est} steps)")
    if cfg.save_every_n_steps and cfg.save_every_n_steps > 0:
        n_ckpts = max(1, total_steps_est // cfg.save_every_n_steps)
        t2.add_row("Save (native)", f"a cada {cfg.save_every_n_steps} steps "
                                    f"(~{n_ckpts} checkpoints)")
    else:
        t2.add_row("Save / eval", f"a cada {cfg.save_every_n_epochs} / "
                                  f"{cfg.eval_every_n_epochs} epochs")
    t2.add_row("Activation chkpt",   "ON" if cfg.activation_checkpointing else "OFF")
    t2.add_row("blocks_to_swap",     str(cfg.blocks_to_swap))
    t2.add_row("FP8 transformer",    "yes" if cfg.use_fp8 else "no (bf16)")
    if cfg.caption_prefix:
        t2.add_row("caption_prefix", repr(cfg.caption_prefix))
    console.print(t2)
    console.print()

    cfg.save()
    success(f"Config salvo em {cfg.project_dir}")

    # No mid-flow confirm — the user already chose the mode + dataset upfront.
    # Pass `skip_confirm` is kept for API compat but isn't needed anymore.
    _ = skip_confirm


# ═══════════════════════════════════════════════════════════════════════════
# Caption phase — PixAI + Gemini Flash (OpenRouter), per group
# ═══════════════════════════════════════════════════════════════════════════

def phase_caption(cfg: TrainingConfig, env: dict) -> None:
    header("CAPTIONING — PixAI booru tags + Gemini Flash (OpenRouter)")

    recaption = "--recaption" in sys.argv

    # Which groups still need captioning? (resume-friendly: skip done groups,
    # finish partial ones). Only require the API key if there's real work.
    def _needs_caption(g: dict) -> bool:
        gd = Path(g["path"])
        return (recaption or g.get("image_count", 0) <= 0
                or _count_nonempty_captions(gd) < g["image_count"])

    pending = [g for g in cfg.groups if _needs_caption(g)]
    if not pending:
        success("Todos os grupos já legendados — nada a refazer "
                "(use --recaption para forçar).")
        for g in cfg.groups:
            g["caption_examples"] = _sample_captions(Path(g["path"]), 3)
        cfg.caption_count = total_caption_count(cfg)
        cfg.save()
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        warn("OPENROUTER_API_KEY não definido — necessário para a caption LLM.")
        console.print("        [dim]Pegue em https://openrouter.ai/keys[/dim]")
        key = ask("Cole sua OpenRouter API key", "", allow_empty=False).strip()
        if not key:
            error("Sem OPENROUTER_API_KEY não dá para gerar captions.")
            sys.exit(1)
        os.environ["OPENROUTER_API_KEY"] = key

    info(f"Modelo de caption: [bold]{CAPTION_MODEL}[/bold]  ·  tagger: PixAI")
    console.print()

    for i, g in enumerate(cfg.groups, 1):
        gd = Path(g["path"])
        if g["is_outfit"]:
            label = f"outfit '[bold]{g['trigger_outfit']}[/bold]'"
            if g.get("custom_instruction"):
                label += "  [dim](instrução custom)[/dim]"
        else:
            label = "personagem base"
        info(f"[{i}/{len(cfg.groups)}] {g['image_count']} imgs · {label} · {escape(gd.name)}")

        # Skip groups already fully captioned (resume-friendly after a crash);
        # override with --recaption.
        nonempty = _count_nonempty_captions(gd)
        if not recaption and g["image_count"] > 0 and nonempty >= g["image_count"]:
            success(f"  já legendado ({nonempty} captions) — pulando "
                    f"(use --recaption p/ refazer)")
            g["caption_examples"] = _sample_captions(gd, 3)
            continue

        _run_captioner_on_group(cfg, g)
        n_caps = _count_nonempty_captions(gd)
        if n_caps == 0:
            warn(f"  Nenhuma caption gerada em '{escape(gd.name)}' — "
                 f"cheque OPENROUTER_API_KEY / HF_TOKEN / cota.")
        else:
            success(f"  {n_caps} captions em '{escape(gd.name)}'")
        g["caption_examples"] = _sample_captions(gd, 3)

    cfg.caption_count = total_caption_count(cfg)
    cfg.save()
    success(f"Captioning concluído — {cfg.caption_count} captions.")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Caption sanity check
# ═══════════════════════════════════════════════════════════════════════════

def phase6_check_captions(cfg: TrainingConfig) -> None:
    header("FASE 6 — Sanity check de captions")

    cfg.image_count   = total_image_count(cfg)
    cfg.caption_count = total_caption_count(cfg)
    missing = find_images_missing_captions(cfg)

    if not missing and cfg.image_count > 0:
        success(f"Todas as {cfg.image_count} imagens têm .txt — ok")
        return

    if cfg.caption_count > 0 and missing:
        warn(f"{cfg.caption_count}/{cfg.image_count} imagens têm .txt; "
             f"{len(missing)} sem caption — criando .txt vazios automaticamente.")
        for p in missing[:10]:
            console.print(f"    [dim]- {escape(str(p))}[/dim]")
        if len(missing) > 10:
            console.print(f"    [dim]... e mais {len(missing) - 10}[/dim]")
        for p in missing:
            p.with_suffix(".txt").write_text("", encoding="utf-8")
        success(f"Criados {len(missing)} .txt vazios (treino prosseguirá com caption vazio).")
        return

    error("Nenhum .txt encontrado — o captioning não produziu legendas.")
    error("Verifique OPENROUTER_API_KEY / HF_TOKEN e rode novamente.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — Generate dataset.toml + config.toml
# ═══════════════════════════════════════════════════════════════════════════

def _toml_quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _toml_array(values: list) -> str:
    return "[" + ", ".join(repr(v) if not isinstance(v, list) else
                           "[" + ", ".join(str(x) for x in v) + "]"
                           for v in values) + "]"


def phase7_write_tomls(cfg: TrainingConfig) -> None:
    header("FASE 7 — Gerando dataset.toml e config.toml")

    # Backfill native step-based saving for resumed animatrem projects (saved
    # before step cadence existed) so they ALSO save ~10× total, not every
    # epoch. Only for grouped (animatrem) projects — the --advanced path keeps
    # whatever epoch cadence it set.
    if cfg.rank > 0 and cfg.groups and cfg.save_every_n_steps <= 0 and cfg.epochs > 0:
        n_res = max(1, len(cfg.resolutions))
        if isinstance(cfg.micro_batch_size, int):
            eff_micro = cfg.micro_batch_size
        elif cfg.micro_batch_size:
            eff_micro = min(b for _, b in cfg.micro_batch_size)
        else:
            eff_micro = 1
        eff_batch = max(1, eff_micro * max(1, cfg.gradient_accumulation_steps))
        if cfg.groups:
            samples_per_epoch = sum(
                int(g.get("image_count", 0)) * int(g.get("num_repeats", 1) or 1)
                for g in cfg.groups) * n_res
        else:
            samples_per_epoch = total_image_count(cfg) * max(1, cfg.num_repeats) * n_res
        steps_per_epoch = max(1, samples_per_epoch // eff_batch)
        cfg.total_steps = steps_per_epoch * max(1, cfg.epochs)
        cfg.save_every_n_steps = _save_every_n_steps(cfg.total_steps, eff_micro)
        cfg.eval_every_n_steps = cfg.save_every_n_steps

    dataset_entries = get_dataset_entries(cfg)

    # ── dataset.toml ───────────────────────────────────────────────
    res_str = "[" + ", ".join(str(r) for r in cfg.resolutions) + "]"

    lines = [
        f"# dataset.toml — Anima training (diffusion-pipe)",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"resolutions      = {res_str}",
        f"enable_ar_bucket = true",
        f"min_ar           = {cfg.min_ar}",
        f"max_ar           = {cfg.max_ar}",
        f"num_ar_buckets   = {cfg.num_ar_buckets}",
        # frame_buckets defaults to [1] for image-only training in diffusion-pipe
        # (utils/dataset.py:506) — no need to emit explicitly.
        "",
    ]
    # Per-group balanced repeats (animatrem). Fall back to the global
    # num_repeats for entries with no matching group (e.g. --advanced path).
    repeats_by_path: dict[str, int] = {}
    for g in cfg.groups:
        try:
            repeats_by_path[str(Path(g["path"]).resolve())] = int(g.get("num_repeats", 1))
        except (KeyError, TypeError, ValueError):
            continue

    for d in dataset_entries:
        nr = repeats_by_path.get(str(Path(d).resolve()), cfg.num_repeats)
        lines += [
            "[[directory]]",
            f"path        = {_toml_quote(str(d))}",
            f"num_repeats = {nr}",
        ]
        if cfg.caption_prefix:
            lines.append(f"caption_prefix = {_toml_quote(cfg.caption_prefix)}")
        lines.append("")

    Path(cfg.dataset_toml_path).write_text("\n".join(lines), encoding="utf-8")
    success(f"dataset.toml → {cfg.dataset_toml_path}")

    # ── config.toml ────────────────────────────────────────────────
    mb = cfg.micro_batch_size
    if isinstance(mb, list):
        mb_str = "[" + ", ".join(f"[{r}, {b}]" for r, b in mb) + "]"
    else:
        mb_str = str(mb)

    cfg_lines: list[str] = [
        f"# config.toml — Anima training (diffusion-pipe)",
        f"# Project: {cfg.project_name}",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"output_dir = {_toml_quote(cfg.output_dir)}",
        f"dataset    = {_toml_quote(cfg.dataset_toml_path)}",
        "",
        f"epochs                       = {cfg.epochs}",
        f"micro_batch_size_per_gpu     = {mb_str}",
        f"pipeline_stages              = {cfg.pipeline_stages}",
        f"gradient_accumulation_steps  = {cfg.gradient_accumulation_steps}",
        f"gradient_clipping            = {cfg.gradient_clipping}",
        f"warmup_steps                 = {cfg.warmup_steps}",
    ]
    if cfg.lr_scheduler and cfg.lr_scheduler != "constant":
        # constant is the diffusion-pipe default; only emit non-default.
        # Supported: 'constant', 'linear', 'cosine' (train.py:814-822).
        cfg_lines.append(f"lr_scheduler                 = {_toml_quote(cfg.lr_scheduler)}")

    if cfg.blocks_to_swap > 0:
        cfg_lines.append(f"blocks_to_swap               = {cfg.blocks_to_swap}")

    # Loss override: diffusion-pipe (cosmos_predict2.py:497-502) accepts
    # `huber_delta` or `smooth_l1_beta` at top level. Default is MSE — emit
    # only when user picked a custom loss.
    if cfg.loss == "huber_delta" and cfg.pseudo_huber_c is not None:
        cfg_lines.append(f"huber_delta                  = {cfg.pseudo_huber_c}")
    elif cfg.loss == "smooth_l1_beta" and cfg.pseudo_huber_c is not None:
        cfg_lines.append(f"smooth_l1_beta               = {cfg.pseudo_huber_c}")

    # Eval + model-save cadence. Prefer native STEP-based saving (train.py
    # names these step<N>/); fall back to epoch cadence only if steps unset
    # (--advanced / FFT). diffusion-pipe requires at least one save_every_*.
    cfg_lines.append("")
    if cfg.eval_every_n_steps and cfg.eval_every_n_steps > 0:
        cfg_lines.append(f"eval_every_n_steps               = {cfg.eval_every_n_steps}")
    else:
        cfg_lines.append(f"eval_every_n_epochs              = {max(1, cfg.eval_every_n_epochs)}")
    cfg_lines += [
        f"eval_before_first_step           = {str(cfg.eval_before_first_step).lower()}",
        f"eval_micro_batch_size_per_gpu    = 1",
        f"eval_gradient_accumulation_steps = 1",
        "",
    ]
    if cfg.save_every_n_steps and cfg.save_every_n_steps > 0:
        cfg_lines.append(f"save_every_n_steps               = {cfg.save_every_n_steps}")
    else:
        cfg_lines.append(f"save_every_n_epochs              = {max(1, cfg.save_every_n_epochs)}")
    cfg_lines += [
        f"checkpoint_every_n_minutes       = {cfg.checkpoint_every_n_minutes}",
        f"activation_checkpointing         = {str(cfg.activation_checkpointing).lower()}",
        f"partition_method                 = 'parameters'",
        f"save_dtype                       = 'bfloat16'",
        f"caching_batch_size               = 1",
        f"steps_per_print                  = 1",
        # video_clip_mode is irrelevant for image-only datasets (diffusion-pipe
        # ignores it when frame_buckets == [1]). Don't emit.
        "",
        "[model]",
        f"type                = 'anima'",
        f"transformer_path    = {_toml_quote(cfg.transformer_path)}",
        f"vae_path            = {_toml_quote(cfg.vae_path)}",
        f"llm_path            = {_toml_quote(cfg.llm_path)}",
        f"dtype               = 'bfloat16'",
    ]
    if cfg.use_fp8:
        cfg_lines.append("transformer_dtype   = 'float8'")
    cfg_lines += [
        f"sigmoid_scale       = {cfg.sigmoid_scale}",
        f"llm_adapter_lr      = {cfg.llm_adapter_lr}",
    ]
    # timestep_sample_method default in diffusion-pipe is 'logit_normal'
    # (cosmos_predict2.py:376). Don't emit when at default — matches tdrussell's
    # canonical TOML which only comments the alternative `'uniform'`.
    if cfg.timestep_sample_method and cfg.timestep_sample_method != "logit_normal":
        cfg_lines.append(f"timestep_sample_method = {_toml_quote(cfg.timestep_sample_method)}")
    # Per-component LRs
    for name, val in [("self_attn_lr",  cfg.self_attn_lr),
                      ("cross_attn_lr", cfg.cross_attn_lr),
                      ("mlp_lr",        cfg.mlp_lr),
                      ("mod_lr",        cfg.mod_lr)]:
        if val is not None:
            cfg_lines.append(f"{name:<19} = {val}")

    # Adapter — omit for full finetune (rank == 0)
    if cfg.rank > 0:
        cfg_lines += [
            "",
            "[adapter]",
            f"type   = 'lora'",
            f"rank   = {cfg.rank}",
            f"dtype  = 'bfloat16'",
        ]
        if cfg.base_lora_path:
            cfg_lines.append(f"init_from_existing = {_toml_quote(cfg.base_lora_path)}")
    else:
        cfg_lines.append("\n# FULL FINETUNE — no [adapter] table")

    cfg_lines += [
        "",
        "[optimizer]",
        f"type         = {_toml_quote(cfg.optimizer_type)}",
        f"lr           = {cfg.learning_rate}",
        f"betas        = [{cfg.betas[0]}, {cfg.betas[1]}]",
        f"weight_decay = {cfg.weight_decay}",
        f"eps          = {cfg.eps}",
        "",
        "[monitoring]",
        "enable_wandb = false",
    ]

    Path(cfg.config_toml_path).write_text("\n".join(cfg_lines), encoding="utf-8")
    success(f"config.toml  → {cfg.config_toml_path}")

    console.print()
    info("Conteúdo do config.toml:")
    for line in cfg_lines[:35]:
        console.print(f"    [dim]{escape(line)}[/dim]")
    if len(cfg_lines) > 35:
        console.print(f"    [dim]... ({len(cfg_lines) - 35} linhas)[/dim]")
    console.print()

    # No confirm here — TOMLs are deterministic from cfg. If you want to edit
    # them, the paths are printed above and you can ctrl-c, edit, and rerun.


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8 — Train
# ═══════════════════════════════════════════════════════════════════════════

_HF_UPLOAD_DISABLED = False  # set True after a storage-limit/permission failure


def _hf_upload_file(local_path: str, repo_path: str, repo_id: str) -> bool:
    global _HF_UPLOAD_DISABLED
    if _HF_UPLOAD_DISABLED:
        return False
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=local_path, path_in_repo=repo_path,
                            repo_id=repo_id, repo_type="model")
        success(f"Upload: {repo_path} → {repo_id}")
        return True
    except Exception as e:
        msg = str(e).lower()
        if ("storage limit" in msg or "403" in msg or "quota" in msg
                or "forbidden" in msg):
            _HF_UPLOAD_DISABLED = True   # stop retrying every checkpoint
            warn("Upload HF desativado: limite de storage/permissão do HuggingFace.")
            console.print("        [dim]Torne o repo público (storage público é "
                          "generoso) ou libere espaço. Os .safetensors continuam "
                          "salvos localmente em outputs/.[/dim]")
        else:
            warn(f"Upload falhou: {e}")
        return False


def _wait_for_stable(p: Path, checks: int = 4, interval: int = 5) -> bool:
    prev = -1
    for _ in range(checks):
        try:
            curr = p.stat().st_size
        except OSError:
            return False
        if curr == prev and curr > 0:
            return True
        prev = curr
        time.sleep(interval)
    return False


_CKPT_DIR_RE = re.compile(r"^(epoch|step)(\d+)$")


def _ckpt_dir_info(d: Path) -> Optional[tuple[str, int]]:
    """Return (tag, ordinal) for a diffusion-pipe model-save dir. train.py names
    them `epoch{N}` (epoch cadence) or `step{N}` (step cadence) — train.py:946/952.
    Returns e.g. ('step600', 600) or ('epoch5', 5); None if not a save dir."""
    m = _CKPT_DIR_RE.match(d.name)
    return (f"{m.group(1)}{m.group(2)}", int(m.group(2))) if m else None


def _epoch_num_from_dir(d: Path) -> Optional[int]:
    info = _ckpt_dir_info(d)
    return info[1] if info else None


def _list_ckpt_dirs(output_dir: Path) -> list[Path]:
    """All model-save dirs (epoch*/ or step*/) under output_dir, sorted by
    ordinal. A single run uses one cadence, so ordinal sort = training order."""
    if not output_dir.exists():
        return []
    cands = list(output_dir.rglob("epoch*")) + list(output_dir.rglob("step*"))
    dirs = {d for d in cands if d.is_dir() and _ckpt_dir_info(d)}
    return sorted(dirs, key=lambda d: _ckpt_dir_info(d)[1])  # type: ignore[index]


def _make_friendly_safetensors_copy(epoch_dir: Path, project_name: str,
                                    wait: bool = True) -> Optional[Path]:
    """Diffusion-pipe hardcodes `adapter_model.safetensors` (LoRA) and
    `model.safetensors` (FFT) at cosmos_predict2.py:320-324. We can't change
    the canonical name (it's referenced by `latest` tags etc.), but we CAN
    drop a friendly-named hardlink/copy alongside, and keep that one in sync
    with HF uploads. Returns the friendly path if created/refreshed.

    wait=True polls for size-stability (used by the live watcher while files are
    still being written); wait=False skips it (post-training, files are done —
    avoids blocking sleeps that made the final sweep hang)."""
    info = _ckpt_dir_info(epoch_dir)
    if info is None:
        return None
    tag, _ = info
    canonical: Optional[Path] = None
    for cand_name in ("adapter_model.safetensors", "model.safetensors"):
        cand = epoch_dir / cand_name
        if cand.exists():
            canonical = cand
            break
    if canonical is None:
        return None
    if wait and not _wait_for_stable(canonical):
        return None
    friendly = epoch_dir / f"{project_name}_{tag}.safetensors"
    if friendly.exists() and friendly.stat().st_size == canonical.stat().st_size:
        return friendly
    try:
        if friendly.exists():
            friendly.unlink()
        os.link(canonical, friendly)
    except OSError:
        try:
            shutil.copy2(canonical, friendly)
        except OSError:
            return None
    return friendly


def _rename_watcher(train_proc: subprocess.Popen, output_dir: Path,
                    project_name: str,
                    on_friendly: Optional[Callable[[Path], None]] = None) -> None:
    """Watch `output_dir` for new `epoch*/` directories. As soon as the
    canonical safetensors inside one is stable, drop a hardlink (or copy)
    named `<project>_epochN.safetensors` next to it. If `on_friendly` is
    given, call it with the friendly path each time we materialize a new
    one — used by the HF monitor to upload nicely-named artifacts."""
    seen: set[str] = set()
    while train_proc.poll() is None:
        time.sleep(30)
        for epoch_dir in _list_ckpt_dirs(output_dir):
            key = str(epoch_dir.resolve())
            friendly = _make_friendly_safetensors_copy(epoch_dir, project_name)
            if friendly and key not in seen:
                seen.add(key)
                success(f"Checkpoint renomeado: {friendly.name}")
                if on_friendly is not None:
                    try:
                        on_friendly(friendly)
                    except Exception:  # never let the watcher crash
                        pass
    # Final pass after training ends — files are done, so don't sleep-wait.
    for epoch_dir in _list_ckpt_dirs(output_dir):
        friendly = _make_friendly_safetensors_copy(epoch_dir, project_name, wait=False)
        if friendly and on_friendly is not None:
            try:
                on_friendly(friendly)
            except Exception:
                pass


def _hf_monitor(train_proc: subprocess.Popen, output_dir: Path,
                uploaded_log: Path, project_name: str, repo_id: str) -> None:
    """Periodically scan output_dir for friendly-named safetensors and upload
    each to HF. The rename watcher creates the friendly names; this monitor
    just uploads them. Entries already uploaded are tracked in `uploaded_log`."""
    try:
        uploaded_log.parent.mkdir(parents=True, exist_ok=True)
        uploaded_log.touch(exist_ok=True)
    except OSError:
        return

    def is_uploaded(name: str) -> bool:
        try:
            return name in uploaded_log.read_text().splitlines()
        except Exception:
            return False

    def mark(name: str) -> None:
        with uploaded_log.open("a") as f:
            f.write(name + "\n")

    pattern = re.compile(rf"^{re.escape(project_name)}_epoch\d+\.safetensors$")

    while train_proc.poll() is None:
        time.sleep(60)
        if not output_dir.exists():
            continue
        for ckpt in output_dir.rglob("*.safetensors"):
            if not pattern.match(ckpt.name):
                continue
            if not _wait_for_stable(ckpt):
                continue
            if is_uploaded(ckpt.name):
                continue
            if _hf_upload_file(str(ckpt), ckpt.name, repo_id):
                mark(ckpt.name)


def _find_resume_target(output_dir: Path) -> Optional[tuple[str, str]]:
    """Find the run dir to resume from, by VALIDATION (not heuristic).

    A run dir at `output_dir/<name>/` is considered resumable iff:
      1. `output_dir/<name>/latest` exists and is readable.
      2. Its content (a tag like `global_step1354`) names a directory that
         actually exists at `output_dir/<name>/<tag>/`.

    Among all resumable run dirs, picks the one whose tag has the highest
    integer step number. Returns (run_dir_name, step_tag) or None.

    This bypasses two known weaknesses of `--resume_from_checkpoint` (no arg):
      - It uses alphabetical sort to pick "most recent", which can land on
        an empty dir from a failed launch.
      - It doesn't validate that `latest` resolves to a real checkpoint.
    """
    if not output_dir.exists():
        return None

    candidates: list[tuple[Path, str, int]] = []
    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue
        latest_file = run_dir / "latest"
        if not latest_file.is_file():
            continue
        try:
            step_tag = latest_file.read_text().strip()
        except OSError:
            continue
        if not step_tag:
            continue
        step_path = run_dir / step_tag
        if not step_path.is_dir():
            continue  # latest points at a nonexistent dir → corrupted
        m = re.search(r"(\d+)", step_tag)
        step_num = int(m.group(1)) if m else 0
        candidates.append((run_dir, step_tag, step_num))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[2], reverse=True)
    chosen_run_dir, chosen_tag, _ = candidates[0]
    return chosen_run_dir.name, chosen_tag


def _ensure_hf_repo(project_name: str) -> tuple[bool, str]:
    """Try to set up an HF model repo for auto-upload. Returns (enabled, repo_id).
    Asked exactly once, in phase4. Silently disabled if no HF_TOKEN."""
    if not os.environ.get("HF_TOKEN"):
        info("HF_TOKEN ausente — upload automático desativado.")
        return False, ""
    if not ask_yn("Ativar upload automático dos checkpoints para HuggingFace?",
                 default=True):
        return False, ""
    try:
        from huggingface_hub import whoami
        username = whoami()["name"]
        default_repo = f"{username}/{project_name}"
    except Exception:
        default_repo = project_name
    repo_id = ask("Repo HuggingFace (user/repo)", default_repo, allow_empty=False)
    try:
        from huggingface_hub import HfApi
        HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)
        success(f"Repo pronto: https://huggingface.co/{repo_id}")
        return True, repo_id
    except Exception as e:
        warn(f"Não consegui criar/acessar repo: {e}")
        return False, ""


def _setup_hf(cfg: TrainingConfig) -> tuple[bool, str]:
    """animatrem default: auto-create the HF model repo (no question).
    PUBLIC by default (public storage is generous and the LoRA is meant to be
    used/shared); set ANIMATREM_HF_PRIVATE=1 for a private repo."""
    if not os.environ.get("HF_TOKEN"):
        warn("HF_TOKEN ausente — upload para HuggingFace desativado.")
        return False, ""
    ns = os.environ.get("ANIMATREM_HF_NAMESPACE", "").strip()
    if not ns:
        try:
            from huggingface_hub import whoami
            ns = whoami()["name"]
        except Exception:
            ns = ""
    repo_id = f"{ns}/{cfg.project_name}" if ns else cfg.project_name
    private = os.environ.get("ANIMATREM_HF_PRIVATE", "0").lower() in (
        "1", "true", "yes", "on")
    cfg.hf_private = private
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", private=private,
                        exist_ok=True)
        # create_repo does NOT change an already-existing repo's visibility, so
        # flip it explicitly. huggingface_hub renamed the API: newer versions
        # use update_repo_settings(private=...), older use
        # update_repo_visibility(...). Try both so a previously-private repo
        # actually becomes public (else uploads keep hitting the private quota).
        flipped = False
        for attempt in (
            lambda: api.update_repo_settings(repo_id=repo_id, private=private,
                                             repo_type="model"),
            lambda: api.update_repo_visibility(repo_id=repo_id, private=private,
                                               repo_type="model"),
        ):
            try:
                attempt()
                flipped = True
                break
            except Exception:
                continue
        vis = "privado" if private else "público"
        if flipped or not private:
            success(f"Repo HF: https://huggingface.co/{repo_id} ({vis})")
        else:
            success(f"Repo HF: https://huggingface.co/{repo_id}")
            warn(f"Não consegui forçar a visibilidade para {vis} "
                 f"(mude manualmente em Settings do repo se precisar).")
        return True, repo_id
    except Exception as e:
        warn(f"Não consegui criar repo HF ({repo_id}): {e}")
        return False, ""


def phase8_train(cfg: TrainingConfig, hf_upload: bool, repo_id: str
                 ) -> tuple[bool, str]:
    header("FASE 8 — Treinamento")

    # Build deepspeed command (matches diffusion-pipe README:117)
    cmd = [
        "deepspeed", "--num_gpus=1",
        "train.py",
        "--deepspeed",
        "--config", cfg.config_toml_path,
    ]

    # Auto-resume: validate, don't guess.
    #
    # diffusion-pipe layout per train.py:516-518:
    #   <output_dir>/<run_timestamp>/global_step<N>/    ← actual checkpoint
    #   <output_dir>/<run_timestamp>/latest             ← text file naming the
    #                                                     active global_step
    #
    # A run dir is RESUMABLE iff:
    #   (a) it contains a `latest` file
    #   (b) the file's content names a global_step<N> dir that ACTUALLY exists
    #
    # We pick the resumable run with the highest step number and pass its
    # name explicitly to `--resume_from_checkpoint <name>` (the documented
    # form per train.py:517). Empty/corrupted run dirs are ignored — no
    # cleanup heuristic, no "most recent" alphabetical guessing.
    resume_target = _find_resume_target(Path(cfg.output_dir))
    if resume_target is not None:
        run_dir_name, step_tag = resume_target
        info(f"Resume válido encontrado: {run_dir_name}/{step_tag}")
        cmd.append("--resume_from_checkpoint")
        cmd.append(run_dir_name)
        # Always reset the dataloader on resume. Diffusion-pipe's saved
        # dataloader state (`client_state['custom_loader']`) is keyed to the
        # batch size, num_repeats, and dataset content at SAVE time. If any
        # of those changed between save and resume (e.g. user edited config,
        # batch auto-recomputed after profile update, dataset added images),
        # `train_dataloader.load_state_dict(...)` produces an exhausted
        # iterator → StopIteration on first next() in train.py:166.
        # With --reset_dataloader, diffusion-pipe still preserves the epoch
        # counter (train.py:847) so progress isn't lost; only the iteration
        # position within the current epoch is reset. Pesos do modelo,
        # optimizer state, scheduler — tudo preservado. Para LoRA training
        # a posição exata dentro do epoch é irrelevante; o trade-off é zero.
        cmd.append("--reset_dataloader")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"]  = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    console.print()
    console.print(Panel(
        "[bold]Comando:[/bold]\n" +
        f"cd {DIFFUSION_PIPE_DIR}\n" +
        " \\\n  ".join(cmd),
        title="deepspeed launch", border_style="dim"))
    console.print()

    # Save command as a shell script alongside (handy for headless reruns).
    script = cfg.project_dir / "run_training.sh"
    script.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        export NCCL_P2P_DISABLE="1"
        export NCCL_IB_DISABLE="1"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        cd {DIFFUSION_PIPE_DIR}
        {' '.join(cmd)}
    """))
    script.chmod(0o755)
    info(f"Comando equivalente: {script}")

    success("Iniciando treinamento…  [dim](Ctrl+C cancela; Ctrl+C 2× força)[/dim]")
    train_proc = subprocess.Popen(
        cmd, cwd=str(DIFFUSION_PIPE_DIR), env=env, start_new_session=True,
    )

    threads: list[threading.Thread] = []
    rename_thread = threading.Thread(
        target=_rename_watcher,
        args=(train_proc, Path(cfg.output_dir), cfg.project_name, None),
        daemon=True,
    )
    rename_thread.start()
    threads.append(rename_thread)

    if hf_upload:
        uploaded_log = Path(cfg.output_dir) / ".hf_uploaded.log"
        hf_thread = threading.Thread(
            target=_hf_monitor,
            args=(train_proc, Path(cfg.output_dir), uploaded_log,
                  cfg.project_name, repo_id),
            daemon=True,
        )
        hf_thread.start()
        threads.append(hf_thread)

    # Robust cancellation. deepspeed runs in its OWN session (start_new_session),
    # so the terminal's Ctrl+C reaches only this parent — not the child. We must
    # catch SIGINT here and forward it to the child's process group ourselves:
    #   1st Ctrl+C → SIGTERM the whole group (graceful stop).
    #   2nd Ctrl+C → SIGKILL the group and hard-exit (guaranteed stop).
    # Without this, Ctrl+C looked ignored and training kept running.
    interrupts = {"n": 0}

    def _on_sigint(signum, frame):
        interrupts["n"] += 1
        try:
            pgid = os.getpgid(train_proc.pid)
        except OSError:
            pgid = None
        if interrupts["n"] == 1:
            console.print("\n[bold dark_orange3]⚠ Cancelando treino (SIGTERM)… "
                          "Ctrl+C de novo para forçar (SIGKILL).[/bold dark_orange3]")
            sig = signal.SIGTERM
        else:
            console.print("\n[bold red1]✘ Forçando encerramento (SIGKILL).[/bold red1]")
            sig = signal.SIGKILL
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except OSError:
                pass
        try:
            train_proc.send_signal(sig)
        except Exception:
            pass
        if interrupts["n"] >= 2:
            os._exit(130)

    old_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        train_proc.wait()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        for t in threads:
            t.join(timeout=3)

    cfg._interrupted = interrupts["n"] > 0  # type: ignore[attr-defined]
    if cfg._interrupted:
        warn("Treino cancelado pelo usuário. Checkpoints já salvos ficam em outputs/.")
    elif train_proc.returncode != 0:
        error(f"Treinamento terminou com código {train_proc.returncode}")

    return hf_upload, repo_id


# ═══════════════════════════════════════════════════════════════════════════
# Phase 9 — Post-training
# ═══════════════════════════════════════════════════════════════════════════

def _finalize_after_cancel(cfg: TrainingConfig) -> None:
    """Fast, local-only wrap-up after the user cancels training: name the
    already-saved checkpoints nicely and print where they are. No stability
    sleeps, no HF uploads — nothing that could hang."""
    header("Cancelado — checkpoints locais")
    out_dir = Path(cfg.output_dir)
    dirs = _list_ckpt_dirs(out_dir)
    if not dirs:
        warn(f"Nenhum checkpoint salvo ainda em {out_dir}")
        return
    friendly: list[Path] = []
    for d in dirs:
        f = _make_friendly_safetensors_copy(d, cfg.project_name, wait=False)
        if f:
            friendly.append(f)
    for f in friendly:
        info(str(f))
    if friendly:
        success(f"{len(friendly)} checkpoint(s) em {out_dir}. "
                f"Rode de novo e escolha 'Continuar' para retomar/subir.")


def phase9_post_training(cfg: TrainingConfig, hf_upload: bool, repo_id: str,
                         env: Optional[dict] = None) -> None:
    header("FASE 9 — Pós-treinamento")

    out_dir = Path(cfg.output_dir)
    if not out_dir.exists():
        warn(f"Output dir não existe: {out_dir}")
        return

    # diffusion-pipe saves models in subdirs like epoch1/, epoch2/... The
    # rename watcher already created `<project>_epochN.safetensors` inside
    # each. Sweep one more time to catch the final epoch in case the watcher
    # didn't fire (process exited too fast).
    epoch_dirs = _list_ckpt_dirs(out_dir)
    if not epoch_dirs:
        warn("Nenhuma pasta epoch*/step* encontrada.")
        return
    for d in epoch_dirs:
        _make_friendly_safetensors_copy(d, cfg.project_name, wait=False)

    latest = epoch_dirs[-1]
    success(f"Último checkpoint salvo: {latest}")

    friendly_pattern = re.compile(
        rf"^{re.escape(cfg.project_name)}_(epoch|step)\d+\.safetensors$")
    friendly_in_latest = next(
        (p for p in latest.iterdir() if p.is_file() and friendly_pattern.match(p.name)),
        None,
    )
    if friendly_in_latest is None:
        # Fallback: take adapter_model / model directly
        canonical = next((latest / n for n in
                          ("adapter_model.safetensors", "model.safetensors")
                          if (latest / n).exists()), None)
        if canonical is None:
            warn(f"Nenhum .safetensors em {latest}")
            return
        friendly_in_latest = canonical

    # Drop the absolute-final file at the project root for convenience.
    final = out_dir / f"{cfg.project_name}.safetensors"
    if final.exists():
        try:
            final.unlink()
        except OSError:
            pass
    try:
        os.link(friendly_in_latest, final)
    except OSError:
        shutil.copy2(friendly_in_latest, final)
    success(f"Cópia final: {final.name}")

    if hf_upload and repo_id:
        _hf_upload_file(str(final), final.name, repo_id)
        # Also try to upload any per-epoch friendly files we may have missed.
        for d in epoch_dirs:
            for p in d.iterdir():
                if p.is_file() and friendly_pattern.match(p.name):
                    _hf_upload_file(str(p), p.name, repo_id)

        # Rich model card + machine-readable metadata (animatrem).
        try:
            from hf_modelcard import write_model_card_files
            card_path, meta_path = write_model_card_files(
                cfg, out_dir, repo_id, env or {},
                model_repo=MODEL_REPO,
                transformer_name=Path(cfg.transformer_path).name,
                caption_model=CAPTION_MODEL,
                final_safetensors=final.name,
            )
            _hf_upload_file(str(card_path), "README.md", repo_id)
            _hf_upload_file(str(meta_path), "animatrem_metadata.json", repo_id)
            success("Model card + metadata enviados ao HuggingFace.")
        except Exception as e:
            warn(f"Falha ao gerar/enviar model card: {e}")

    # Inference hint
    triggers = [cfg.trigger_character] + [
        g["trigger_outfit"] for g in cfg.groups if g.get("is_outfit")]
    triggers = [t for t in triggers if t]
    trig_line = ", ".join(triggers) if triggers else cfg.trigger_character
    console.print()
    console.print(Panel(
        f"[bold cyan]LoRA pronto para uso no ComfyUI[/bold cyan]\n"
        f"Caminho: [bold]{final}[/bold]\n"
        f"Triggers: [bold]{escape(trig_line)}[/bold]\n"
        f"Copie para: [dim]ComfyUI/models/loras/[/dim] e use o nó 'Load LoRA' com o checkpoint Anima.\n"
        + (f"HF: [dim]https://huggingface.co/{repo_id}[/dim]" if (hf_upload and repo_id) else ""),
        border_style="cyan",
    ))
    console.print()

    success("Pronto.")


# ═══════════════════════════════════════════════════════════════════════════
# Project select / resume
# ═══════════════════════════════════════════════════════════════════════════

def _select_or_create_project() -> tuple[Optional[TrainingConfig], bool]:
    existing = TrainingConfig.list_projects()
    if not existing:
        return None, False

    header("Projetos existentes")
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("#", style="bold"); t.add_column("Projeto"); t.add_column("Status", style="dim")
    for i, name in enumerate(existing, 1):
        c = TrainingConfig.load(name)
        if c:
            status = f"recipe={c.recipe} rank={c.rank} epochs={c.epochs} res={c.resolutions}"
        else:
            status = "?"
        t.add_row(str(i), name, status)
    t.add_row(str(len(existing) + 1), "[bold green]+ Novo projeto[/bold green]", "")
    console.print(t)
    console.print()

    choice = ask("Número ou nome do projeto", str(len(existing) + 1), allow_empty=False)
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(existing):
            return _resume_project(existing[idx])
        if idx == len(existing):
            return None, False
    except ValueError:
        pass
    if choice in existing:
        return _resume_project(choice)
    return None, False


def _resume_project(name: str) -> tuple[Optional[TrainingConfig], bool]:
    saved = TrainingConfig.load(name)
    if saved is None:
        return None, False

    console.print()
    success(f"Projeto: [bold]{name}[/bold]")
    success(f"recipe={saved.recipe} rank={saved.rank} epochs={saved.epochs} res={saved.resolutions}")
    console.print()

    out_dir = Path(saved.output_dir) if saved.output_dir else Path("/nonexistent")
    has_dataset_toml = Path(saved.dataset_toml_path).exists() if saved.dataset_toml_path else False
    has_config_toml  = Path(saved.config_toml_path).exists()  if saved.config_toml_path  else False
    epoch_dirs = sorted(out_dir.rglob("epoch*")) if out_dir.exists() else []
    saved.image_count = total_image_count(saved)

    t = Table(title="Estado", show_header=False, box=None, padding=(0, 1))
    t.add_column(width=3); t.add_column()
    icon = lambda ok: "[green]✅[/green]" if ok else "[red]❌[/red]"
    t.add_row(icon(saved.image_count > 0),  f"{saved.image_count} imagens" if saved.image_count else "Sem imagens")
    t.add_row(icon(has_dataset_toml),       "dataset.toml" if has_dataset_toml else "dataset.toml faltando")
    t.add_row(icon(has_config_toml),        "config.toml"  if has_config_toml  else "config.toml faltando")
    t.add_row(icon(len(epoch_dirs) > 0),    f"{len(epoch_dirs)} epoch(s) salvos" if epoch_dirs else "Sem checkpoints")
    console.print(t)
    console.print()

    ready = saved.image_count > 0 and has_dataset_toml and has_config_toml

    choices: dict[str, str] = {}
    if ready:
        choices["1"] = "Treinar agora (TOMLs prontos)"
        choices["2"] = "Continuar de onde parou"
    else:
        choices["1"] = "Continuar de onde parou (regerar TOMLs faltantes)"
    choices["3"] = "Reconfigurar (mudar receita, rank, etc)"
    choices["4"] = "Novo projeto do zero"

    c = ask_choice("O que fazer?", choices, "1")
    if c == "1" and ready:
        return saved, True
    if c in ("1", "2"):
        return saved, False
    if c == "3":
        saved.output_dir = ""
        saved.config_toml_path = ""
        saved.dataset_toml_path = ""
        saved._reconfigure = True  # type: ignore[attr-defined]
        return saved, False
    if c == "4":
        return None, False
    return saved, False


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

BANNER = r"""
[bold dark_blue]
  ╔═══╗ ╔═╗ ╦ ╦ ╦ ╔══╗  ╔═══╗
  ╠═══╣ ║ ║ ║ ║ ║ ║  ║  ╠═══╣
  ╩   ╩ ╝ ╚ ╝ ╩ ╩ ╚══╝  ╩   ╩  Training[/bold dark_blue]
[bold dark_magenta]  ╔════════════════════╗
  ║   霊 を 見 る 機 械   ║
  ╚════════════════════╝[/bold dark_magenta]"""


def main() -> None:
    force = "--force" in sys.argv
    advanced = "--advanced" in sys.argv

    console.print()
    console.print(Panel(
        BANNER + "\n"
        "\n"
        "[bold dark_blue]animatrem[/bold dark_blue] [dim]— treinador de personagem+outfit para Anima[/dim]\n"
        "[dark_cyan]dataset → caption → treino → HuggingFace, em um comando[/dark_cyan]\n"
        f"[dim]Modelo: {MODEL_REPO} · captioner: data_araknideo · engine: diffusion-pipe[/dim]\n"
        "\n"
        "[bold dark_green]  ✔ Baixa as imagens (Mega / zip / URL / HF / pasta local)[/bold dark_green]\n"
        "[bold dark_green]  ✔ Legenda tudo: PixAI + Gemini Flash (OpenRouter)[/bold dark_green]\n"
        "[bold dark_green]  ✔ 1 LoRA multi-outfit · detecta GPU (5090/4090/4080/3090…)[/bold dark_green]\n"
        "[bold dark_green]  ✔ Sobe .safetensors + model card no HuggingFace[/bold dark_green]"
        + ("\n[dim]  (modo --advanced: receitas/hiperparâmetros manuais, sem captioning)[/dim]" if advanced else ""),
        border_style="dark_blue",
        title="[bold dark_magenta]✦ animatrem ✦[/bold dark_magenta]",
        subtitle="[dim dark_cyan]──── Anima ────[/dim dark_cyan]",
        padding=(1, 3),
    ))
    console.print()

    cfg: Optional[TrainingConfig] = None
    skip_to_train = False

    if not force:
        cfg, skip_to_train = _select_or_create_project()

    phase0_preflight()
    env = phase1_check_environment()
    phase2_install_diffusion_pipe(env)
    if not advanced:
        phase2b_install_captioner(env)
    transformer_path, vae_path, llm_path = phase3_download_models(env)

    reconfigure = getattr(cfg, "_reconfigure", False) if cfg else False
    if cfg is None or reconfigure:
        if cfg is None:
            cfg = TrainingConfig()
        cfg.transformer_path = str(transformer_path)
        cfg.vae_path         = str(vae_path)
        cfg.llm_path         = str(llm_path)
        if advanced:
            # Legacy power path: pick recipe/hyperparams; captions supplied
            # externally (no built-in captioning).
            cfg = phase4_configure(cfg, env, transformer_path, vae_path, llm_path)
        else:
            cfg = phase_wizard(cfg, env, transformer_path, vae_path, llm_path)
    else:
        # Resume: refresh model paths; the saved groups/captions drive the rest.
        cfg.transformer_path = str(transformer_path)
        cfg.vae_path         = str(vae_path)
        cfg.llm_path         = str(llm_path)

    # Caption whenever this is an animatrem project (has groups). phase_caption
    # skips already-captioned groups, so a resume is cheap and also finishes any
    # partially-captioned group instead of training on empty .txt.
    need_caption = (not advanced) and bool(getattr(cfg, "groups", []))

    # Caption first (so the summary shows real caption counts), then TOMLs.
    if need_caption:
        phase_caption(cfg, env)  # skips already-captioned groups

    phase5_summary(cfg, env, skip_confirm=skip_to_train)

    # ALWAYS (re)generate the TOMLs from the current cfg + current code, even on
    # "train now" resume — otherwise a stale config.toml from an older version
    # (e.g. epoch-based saving) would be reused and you'd get old behavior.
    phase6_check_captions(cfg)
    phase7_write_tomls(cfg)

    # HF upload: auto (private) in default mode; asked in --advanced.
    if advanced:
        hf_upload, repo_id = _ensure_hf_repo(cfg.project_name)
    else:
        hf_upload, repo_id = _setup_hf(cfg)

    hf_upload, repo_id = phase8_train(cfg, hf_upload, repo_id)
    if getattr(cfg, "_interrupted", False):
        # User cancelled training — do NOT run the (potentially slow/uploading)
        # post-processing. Just point at the checkpoints already on disk.
        _finalize_after_cancel(cfg)
    else:
        phase9_post_training(cfg, hf_upload, repo_id, env)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n  [bold dark_orange3]Interrompido pelo usuário.[/bold dark_orange3]")
        sys.exit(130)

#!/usr/bin/env python3
"""
hf_modelcard — build a rich HuggingFace model card + metadata for an animatrem
Anima character LoRA.

Standalone (no import of animatrem, to avoid cycles): it duck-types the config
object and takes the remaining facts as explicit params.

Written by phase9_post_training:
  - README.md                  (human-facing model card, HF-rendered)
  - animatrem_metadata.json    (machine-readable)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _clean(text: str) -> str:
    """Repair lone surrogates (from ASCII+surrogateescape stdin) so utf-8 writes
    never crash. Idempotent for valid strings."""
    if not isinstance(text, str):
        return text
    try:
        return text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text.encode("utf-8", "replace").decode("utf-8", "replace")


def _truncate(text: str, limit: int = 500) -> str:
    text = " ".join(_clean(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _groups(cfg: Any) -> list[dict]:
    return list(getattr(cfg, "groups", []) or [])


def _triggers(cfg: Any) -> list[dict]:
    """Ordered trigger list: character first, then each outfit."""
    out = [{"trigger": getattr(cfg, "trigger_character", ""), "kind": "character"}]
    for g in _groups(cfg):
        if g.get("is_outfit") and g.get("trigger_outfit"):
            out.append({"trigger": g["trigger_outfit"], "kind": "outfit",
                        "folder": Path(g.get("path", "")).name})
    return out


def build_metadata(cfg: Any, repo_id: str, env: dict, *, model_repo: str,
                   transformer_name: str, caption_model: str,
                   final_safetensors: str) -> dict:
    groups_meta = []
    for g in _groups(cfg):
        groups_meta.append({
            "folder": Path(g.get("path", "")).name,
            "is_outfit": bool(g.get("is_outfit")),
            "trigger_outfit": g.get("trigger_outfit", ""),
            "custom_instruction": g.get("custom_instruction", ""),
            "image_count": int(g.get("image_count", 0)),
            "num_repeats": int(g.get("num_repeats", 1)),
            "caption_examples": list(g.get("caption_examples", []))[:3],
        })
    return {
        "tool": "animatrem",
        "repo_id": repo_id,
        "project_name": getattr(cfg, "project_name", ""),
        "trigger_character": getattr(cfg, "trigger_character", ""),
        "outfit_mode": ("locked" if getattr(cfg, "outfit_lock", True)
                        else "described"),
        "triggers": _triggers(cfg),
        "safetensors": final_safetensors,
        "base_model": model_repo,
        "transformer": transformer_name,
        "caption_pipeline": {"tagger": "pixai", "llm": caption_model,
                             "provider": "openrouter"},
        "training": {
            "engine": "tdrussell/diffusion-pipe",
            "recipe": getattr(cfg, "recipe", "character"),
            "rank": getattr(cfg, "rank", 32),
            "learning_rate": getattr(cfg, "learning_rate", 2e-5),
            "optimizer": getattr(cfg, "optimizer_type", "adamw_optimi"),
            "betas": getattr(cfg, "betas", [0.9, 0.99]),
            "weight_decay": getattr(cfg, "weight_decay", 0.01),
            "resolutions": getattr(cfg, "resolutions", [768, 1024]),
            "epochs": getattr(cfg, "epochs", 0),
            "sigmoid_scale": getattr(cfg, "sigmoid_scale", 1.3),
            "llm_adapter_lr": getattr(cfg, "llm_adapter_lr", 0.0),
            "timestep_sample_method": getattr(cfg, "timestep_sample_method",
                                              "logit_normal"),
        },
        "gpu": {"name": env.get("gpu_name", ""), "vram_gb": env.get("vram_gb", 0),
                "profile": env.get("gpu_profile", "")},
        "dataset": {"total_images": getattr(cfg, "image_count", 0),
                    "groups": groups_meta},
    }


def build_model_card(cfg: Any, repo_id: str, env: dict, *, model_repo: str,
                     transformer_name: str, caption_model: str,
                     final_safetensors: str) -> str:
    project = getattr(cfg, "project_name", "anima-lora")
    char = getattr(cfg, "trigger_character", "")
    groups = _groups(cfg)
    outfit_groups = [g for g in groups if g.get("is_outfit")]

    # ── YAML frontmatter (HF-rendered) ──────────────────────────────
    fm = [
        "---",
        f"base_model: {model_repo}",
        "library_name: diffusion-pipe",
        "pipeline_tag: text-to-image",
        "tags:",
        "  - lora",
        "  - anima",
        "  - diffusion-pipe",
        "  - text-to-image",
        "  - character",
    ]
    # trigger words as HF widget-friendly instance prompts
    fm.append("instance_prompt: " + json.dumps(char))
    fm.append("---")

    # ── Trigger table ───────────────────────────────────────────────
    trig_rows = [f"| `{char}` | character | (base identity, at caption start) |"]
    for g in outfit_groups:
        instr = _truncate(g.get("custom_instruction", ""), 80) or "—"
        trig_rows.append(
            f"| `{g['trigger_outfit']}` | outfit | {instr} |")

    # ── Example prompt ──────────────────────────────────────────────
    if outfit_groups:
        ex_trigger = f"{char}, {outfit_groups[0]['trigger_outfit']}"
    else:
        ex_trigger = char

    lines: list[str] = []
    lines.append(f"# {project} — Anima character LoRA")
    lines.append("")
    lines.append(f"Character + outfit LoRA for **[Anima]({_hf_url(model_repo)})** "
                 f"(Cosmos-Predict2 + Qwen3), trained with "
                 f"[diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) "
                 f"via [animatrem](https://github.com/adbrasi/character_animatrem).")
    lines.append("")
    lines.append("## Triggers")
    lines.append("")
    lines.append("| Trigger | Kind | Notes |")
    lines.append("| --- | --- | --- |")
    lines += trig_rows
    lines.append("")
    if outfit_groups:
        lines.append("Put the **character** trigger at the start of the prompt and "
                     "add the **outfit** trigger to select that outfit. The outfit "
                     "was captioned only as its trigger (no visual description), so "
                     "it stays consistent.")
        lines.append("")

    # ── Usage ───────────────────────────────────────────────────────
    lines.append("## Usage (ComfyUI)")
    lines.append("")
    lines.append(f"1. Download `{final_safetensors}` and copy it to "
                 f"`ComfyUI/models/loras/`.")
    lines.append(f"2. Load the Anima base checkpoint (`{transformer_name}`) and add a "
                 f"**Load LoRA** node.")
    lines.append("3. Example prompt:")
    lines.append("")
    lines.append("```")
    lines.append(f"{ex_trigger}, <your scene: pose, action, environment, lighting>")
    lines.append("```")
    lines.append("")

    # ── Per-outfit sections with real caption examples ──────────────
    if groups:
        lines.append("## Datasets")
        lines.append("")
        for g in groups:
            if g.get("is_outfit"):
                title = f"Outfit `{g['trigger_outfit']}`"
            else:
                title = f"Character base (`{char}`)"
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"- Folder: `{Path(g.get('path','')).name}`  ·  "
                         f"Images: **{g.get('image_count', 0)}**  ·  "
                         f"num_repeats: {g.get('num_repeats', 1)}")
            if g.get("custom_instruction"):
                lines.append(f"- Caption instruction: "
                             f"_{_truncate(g['custom_instruction'], 300)}_")
            examples = list(g.get("caption_examples", []))[:3]
            if examples:
                lines.append("- Example captions:")
                lines.append("")
                for ex in examples:
                    lines.append(f"  > {_truncate(ex, 500)}")
                    lines.append("")
            else:
                lines.append("")

    # ── Training config ─────────────────────────────────────────────
    lines.append("## Training")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Base model | `{model_repo}` ({transformer_name}) |")
    lines.append("| Engine | tdrussell/diffusion-pipe |")
    lines.append(f"| Recipe | {getattr(cfg, 'recipe', 'character')} |")
    lines.append(f"| Rank | {getattr(cfg, 'rank', 32)} |")
    lines.append(f"| Learning rate | {getattr(cfg, 'learning_rate', 2e-5)} |")
    lines.append(f"| Optimizer | {getattr(cfg, 'optimizer_type', 'adamw_optimi')} "
                 f"betas={getattr(cfg, 'betas', [0.9, 0.99])} "
                 f"wd={getattr(cfg, 'weight_decay', 0.01)} |")
    lines.append(f"| Resolutions | {getattr(cfg, 'resolutions', [768, 1024])} |")
    lines.append(f"| Epochs | {getattr(cfg, 'epochs', 0)} |")
    lines.append(f"| sigmoid_scale | {getattr(cfg, 'sigmoid_scale', 1.3)} |")
    lines.append(f"| llm_adapter_lr | {getattr(cfg, 'llm_adapter_lr', 0.0)} |")
    gpu_name = env.get("gpu_name", "")
    if gpu_name:
        lines.append(f"| GPU | {gpu_name} ({env.get('vram_gb', 0)} GB) |")
    lines.append(f"| Caption pipeline | PixAI booru tags → {caption_model} (OpenRouter) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Generated by [animatrem](https://github.com/adbrasi/character_animatrem)._")
    lines.append("")

    return "\n".join(fm) + "\n\n" + "\n".join(lines)


def _hf_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def write_model_card_files(cfg: Any, out_dir: Path, repo_id: str, env: dict, *,
                           model_repo: str, transformer_name: str,
                           caption_model: str, final_safetensors: str
                           ) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    card = build_model_card(
        cfg, repo_id, env, model_repo=model_repo,
        transformer_name=transformer_name, caption_model=caption_model,
        final_safetensors=final_safetensors)
    meta = build_metadata(
        cfg, repo_id, env, model_repo=model_repo,
        transformer_name=transformer_name, caption_model=caption_model,
        final_safetensors=final_safetensors)

    card_path = out_dir / "README.md"
    meta_path = out_dir / "animatrem_metadata.json"
    card_path.write_text(_clean(card), encoding="utf-8")
    meta_path.write_text(_clean(json.dumps(meta, indent=2, ensure_ascii=False)),
                         encoding="utf-8")
    return card_path, meta_path

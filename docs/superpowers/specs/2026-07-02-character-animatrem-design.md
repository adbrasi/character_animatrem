# character_animatrem — Design Spec

**Date:** 2026-07-02
**Status:** IMPLEMENTED (v1). Direction approved ("pode avançar na criação").

### Implementation notes / deltas from initial design
- **Base model changed by user:** `anima-base-v1.0.safetensors` (not preview3).
- **Upstream verified:** diffusion-pipe still compatible — `model.type='anima'`
  (dispatches to CosmosPredict2), launch `deepspeed … train.py --deepspeed
  --config`, resume `--resume_from_checkpoint`+`--reset_dataloader`. New guard:
  must NOT emit `alpha` in `[adapter]` (we don't).
- **Caption LLM:** `google/gemini-2.5-flash` on OpenRouter (strict JSON schema
  supported). Captioner run as `--taggers pixai,grok --grok_provider openrouter
  --grok_model <gemini>`; PixAI runs first and feeds the LLM as context.
- **Input flow finalized:** ONE input source; the tool detects subfolders
  (descends single-wrapper dirs) → one group per subfolder; per-folder
  "is it a specific outfit?" + trigger + free-text `custom_instruction`. No
  "add another outfit" loop.
- Built by evolving `research_anima_train/anima_training_diffusion_pipe.py` into
  `animatrem.py` (new wizard/caption/HF-card phases; legacy path kept behind
  `--advanced`). Model card + metadata in `hf_modelcard.py`.
- Verified so far (no GPU): `py_compile`, and unit smoke tests for `_slugify`,
  `detect_groups` (single / multi-outfit / nested-flatten), balanced
  `num_repeats` schedule, multi-`[[directory]]` TOML, and model-card render.

## 1. Goal

A single, fast, no-technical-questions tool that fuses **dataset creation** and **Anima LoRA training** for **characters (and their outfits)** into one run. The user turns on a GPU box, runs one command, answers a handful of content questions (project name, character trigger, image link, and — only when the input has multiple folders — per-outfit name + optional caption instruction), and the tool:

1. downloads the images,
2. captions them (PixAI booru tags → Gemini Flash via OpenRouter),
3. trains one character LoRA on diffusion-pipe (multi-outfit, multi-`[[directory]]`),
4. uploads the `.safetensors` to HuggingFace **with a rich model card** (triggers, per-outfit example captions, training config, ComfyUI usage).

No questions about which tagger, which LLM, batch size, rank, lr, epochs — all locked to sensible defaults.

## 2. Non-goals

- Not a general captioner/trainer UI (style/concept/full-finetune recipes are hidden behind an advanced flag, not part of the main flow).
- Not video training (LTX/musubi is out of scope).
- No xAI/Grok path (we use OpenRouter+Gemini only) — one fewer API key.
- Not re-implementing the captioner or diffusion-pipe; we orchestrate them.

## 3. Key product decisions (confirmed with user)

- **Multi-outfit → one LoRA, multi-trigger.** One training run, one `.safetensors`. The character learns all outfits; each outfit is a distinct trigger token. Each folder becomes one `[[directory]]` block (separation is for convenience; training-equivalent to merging).
- **Always character-centric.** Every group shares the character trigger, placed at the **start** of the caption. Outfit trigger, when present, is an additional sub-trigger.
- **The outfit distinction is purely a captioning instruction.** When a folder is a "specific outfit", the LLM is told to name the outfit **only as its trigger** (no color/fabric/cut description) so the LoRA keeps the outfit **consistent** (detailed outfit prose makes the LoRA "flexible" with clothing, which the user does not want). Everything else (pose, environment, lighting, character identity) is still described.
- **Input is a single path containing the structure.** The tool inspects what was downloaded:
  - **One folder of images (or a single image)** → no outfit questions; normal character captioning; one group.
  - **Multiple subfolders** → for **each** subfolder, ask: "folder `<name>` has N images — is it a specific outfit?" If **yes**: ask outfit trigger name + an optional **free-text custom instruction** injected into that group's LLM prompt. If **no**: normal character captioning for that folder.
  - No "add another outfit" loop — the structure already came in the input.

## 4. Distribution architecture (Approach A)

New repo `adbrasi/character_animatrem`. Single Python entrypoint `animatrem.py` (Python chosen over `.sh`: whole stack is Python — Rich TUI, subprocess, HF Hub, GPU detection). A tiny `bootstrap.sh` covers the first clone+run.

One-line command:
```bash
git clone https://github.com/adbrasi/character_animatrem && cd character_animatrem && python animatrem.py
```

`animatrem.py` self-bootstraps on first run:
- creates/uses a venv,
- `git clone` **captioner engine** `https://github.com/adbrasi/data_araknideo` (maintained upstream; reused, not duplicated),
- `git clone` **training engine** `https://github.com/tdrussell/diffusion-pipe` (with submodules),
- installs deps (captioner `requirements.txt`, PyTorch cu124/cu128 per GPU, diffusion-pipe `requirements.txt`),
- downloads Anima weights from HF `circlestone-labs/Anima`,
- detects GPU (RTX 5090 / 4090 / 4080 / 3090 / etc.) → training profile.

The **trainer logic** (evolved from `research_anima_train/anima_training_diffusion_pipe.py`) lives **inside** this repo (that project is local-only and is exactly what we are improving/updating). The **captioner** is driven via its stable CLI (`tag_images_by_wd14_tagger.py`), as `fast.py` already does.

## 5. Execution flow

1. **Project name** (character name) → HF repo default, project folder name, `.safetensors` name.
2. **Character trigger** (default = slug of project name) — shared across all groups, placed at caption start.
3. **Input path** (MEGA link / zip / HTTP archive / HF dataset / local folder). Download + extract + flatten as needed (reuse `mega_download.py`, `snapshot_download`, zip extraction).
4. **Structure detection** (mirrors trainer's `get_dataset_entries`): images directly in root → single group; subfolders with images → one group per subfolder.
5. **Per-group interaction** (only when >1 folder):
   - Print "folder `<name>`: N images. Is it a specific outfit? [y/N]".
   - If **y**: prompt outfit trigger name (validated slug); prompt optional free-text custom instruction (multi-line ok).
   - If **n**: group captioned as plain character.
   (Single-folder / single-image input skips this entirely → plain character group.)
6. **Caption each group** (see §6).
7. **Assemble dataset** → `dataset.toml` with one `[[directory]]` per group, balanced `num_repeats` (see §7).
8. **Train** (see §7), producing `<project>.safetensors`.
9. **Upload to HF** with rich model card (see §8).

## 6. Captioning (locked defaults)

For each group, run the captioner CLI once:
```
python tag_images_by_wd14_tagger.py <group_dir> \
  --taggers pixai,grok \
  --grok_provider openrouter \
  --grok_model <OPENROUTER_GEMINI_FLASH_ID> \
  --prompt_profile <profile> \
  --prompt_var trigger_character=<char> \
  [--prompt_var trigger_outfit=<outfit>] \
  [--prompt_var custom_instruction=<free text>] \
  --recursive --force --remove_underscore --thresh 0.30
```
- **Tagger:** PixAI (user preference over WD14). `pixai` runs first for booru context, then `grok` (the vision-LLM stage) produces the final natural-language/hybrid caption written to `<image>.txt`.
- **Caption LLM:** Gemini Flash via OpenRouter (`--grok_provider openrouter`, model id configurable via `.env` `ANIMATREM_CAPTION_MODEL`; default a current Gemini Flash id — to be confirmed against openrouter.ai/models). Strict JSON schema vs `json_object` fallback to be verified for Gemini.
- **Profiles:**
  - **plain character group** → existing `anima-character` (hybrid tags+prose, trigger as subject, describes clothing).
  - **specific-outfit group** → new **`anima-character-outfit`** profile (see §9): character hybrid format **but** the outfit is written only as `{trigger_outfit}` (no visual outfit description) + `{custom_instruction}` slot.
- **Custom-instruction injection is safe:** the captioner's `render_prompt_template` uses regex placeholder substitution (not `str.format`), and args are passed via subprocess list form, so free text with quotes/braces/newlines cannot break templating or the shell. `custom_instruction` is always passed (empty string when unset) to avoid "missing variable" errors.
- Images that already have a `.txt` are skipped unless `--recaption` is given.

## 7. Training (diffusion-pipe)

- **One `dataset.toml`** with a `[[directory]]` per group. Global bucket keys from the character recipe (`resolutions`, `enable_ar_bucket`, `min_ar`, `max_ar`, `num_ar_buckets`). `caption_prefix` stays empty (trigger is baked into each `.txt` by the captioner).
- **Balanced exposure:** per-`[[directory]]` `num_repeats` computed so small outfit sets (e.g. 25 imgs) are not drowned by a large base set — evolution of the current single global `num_repeats`. Target ≈ equal total exposures per group, capped by the recipe's `exposures_per_image`.
- **One `config.toml`** from the locked `character` recipe: rank 32, lr 2e-5, `adamw_optimi`, bf16, `sigmoid_scale=1.3`, `llm_adapter_lr=0`, `timestep_sample_method='logit_normal'`, gradient clipping 1.0, warmup 100. Epochs/repeats auto-derived from total image count. **GPU-auto** micro-batch / grad-accum / `blocks_to_swap` / optional fp8 from the detected profile.
- **Launch:** `deepspeed --num_gpus=1 train.py --deepspeed --config config.toml` inside the diffusion-pipe clone, with resume/monitor threads (reused). **Exact command, flags, and `[model]`/`[adapter]`/`[optimizer]` keys will be reconciled against the CURRENT diffusion-pipe upstream** (the user explicitly flagged possible drift since April 2026) before shipping — see §10.
- Output: `projects/<name>/outputs/<ts>/epochN/…` → friendly `<project>_epochN.safetensors` → final `<project>.safetensors`.

## 8. Rich HuggingFace upload (new)

Besides the `.safetensors`, generate and upload:
- **`README.md` model card** with: character name; **triggers table** (character trigger + each outfit trigger); base model (Anima preview3-base), engine (diffusion-pipe), recipe + key hyperparameters, GPU used; **per-group section** with image count, trigger, custom instruction used, and **2–3 example captions** sampled from the generated `.txt`; **ComfyUI usage** (where to place the LoRA + example prompt using the triggers).
- **`metadata.json`** (machine-readable: triggers, groups, config, dataset stats).
- Repo **private by default** (configurable via `.env`/flag).
- Reuse the trainer's per-file HF upload (`create_repo` + `upload_file`); the model card + metadata are the new artifacts.

## 9. New prompt profile: `anima-character-outfit`

Lives in this repo at `prompts/image/anima-character-outfit/` and is symlinked/copied into the cloned captioner's `prompts/image/` at bootstrap so `--prompt_profile anima-character-outfit` resolves. Files:
- `profile.json`: variables `trigger_character` (required), `trigger_outfit` (required), `custom_instruction` (optional, default "").
- `system_prompt.md`: the `anima-character` hybrid rules, **plus** the outfit rule from `anima-outfit` (write the outfit only as `{trigger_outfit}`, never describe its color/fabric/cut/silhouette), **plus** a section that injects `{custom_instruction}` verbatim as high-priority per-dataset guidance.
- `user_prompt.md`: character trigger at start, keep canonical tags, prose for narrative, outfit as `{trigger_outfit}` only, keep `{tags}` placeholder for per-image substitution, and surface `{custom_instruction}`.

## 10. Risks / upstream verification (do before shipping)

- **diffusion-pipe currency:** verify current `model.type='anima'` support, exact `[model]`/`[adapter]`/`[optimizer]` keys, block-swap key name, launch flags, resume flags against `github.com/tdrussell/diffusion-pipe`. Adjust the TOML generator + launch command to match. (Research in progress.)
- **HF `circlestone-labs/Anima` layout:** confirm the three relative paths still exist.
- **OpenRouter Gemini Flash:** confirm exact model id and whether strict JSON schema works (else `json_object` fallback).
- **Smoke test without GPU:** `py_compile` + run the wizard through folder detection + caption command construction (dry-run) before a real GPU run.

## 11. Deliverables (files in `character_animatrem`)

- `animatrem.py` — orchestrator + wizard + bootstrap + trainer driver (evolved from `anima_training_diffusion_pipe.py`, refactored into functions).
- `bootstrap.sh` — first clone+run helper.
- `prompts/image/anima-character-outfit/{profile.json,system_prompt.md,user_prompt.md}`.
- `hf_modelcard.py` — model-card + metadata generator.
- `requirements.txt`, `.env.example`, `.gitignore`, `README.md`.

## 12. Build sequence

1. Reconcile diffusion-pipe upstream (§10); finalize training command + TOML keys.
2. Scaffold repo (requirements, .gitignore, .env.example, README).
3. New `anima-character-outfit` prompt profile.
4. `animatrem.py`: bootstrap (venv, clones, model download, GPU detect).
5. Wizard + input download + folder detection.
6. Per-group captioning wiring (subprocess to captioner).
7. dataset.toml/config.toml generation (balanced repeats) + deepspeed launch + monitor/resume.
8. `hf_modelcard.py` + rich HF upload.
9. Validation: `py_compile`, dry-run wizard, then a real small-dataset GPU run by the user.
10. Create GitHub repo, push, verify one-line command.

## 13. Validation

- `python -m py_compile animatrem.py hf_modelcard.py`
- Dry-run of the wizard + folder detection + caption-command construction on a tiny sample (no GPU).
- Real run on a small character+outfit dataset on the user's GPU (user will review architecture first, then execute).

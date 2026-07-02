# character_animatrem

Treinador **rápido e simples** de LoRA de personagem (e outfits) para o modelo
**Anima**, num único comando. Ele une, num só fluxo:

**baixar imagens → legendar (PixAI + Gemini Flash) → treinar (diffusion-pipe) → subir no HuggingFace com model card.**

Sem perguntas técnicas: tagger, LLM de caption, rank, LR, batch, epochs — tudo
com defaults travados. Você só responde conteúdo (nome, trigger, link).

## Comando único

```bash
git clone https://github.com/adbrasi/character_animatrem
cd character_animatrem
python animatrem.py
```

Ou, numa GPU nova, o bootstrap:

```bash
curl -fsSL https://raw.githubusercontent.com/adbrasi/character_animatrem/main/bootstrap.sh | bash
```

O `animatrem.py` faz tudo sozinho no 1º run: cria venv, clona o captioner
(`data_araknideo`) e o `diffusion-pipe`, instala deps, detecta a GPU e baixa os
modelos Anima.

## Pré-requisitos

- Linux / WSL2, GPU NVIDIA (RTX 5090 / 4090 / 4080 / 3090 Ti / etc.), 12 GB+.
- `git`, `python3`, `nvidia-smi`.
- Duas chaves (em `.env` ou perguntadas uma vez):
  - `OPENROUTER_API_KEY` — LLM de caption (Gemini Flash via OpenRouter).
  - `HF_TOKEN` — modelo gated do PixAI, download do Anima e upload do LoRA.

Copie `.env.example` para `.env` e preencha.

## Fluxo (o que ele pergunta)

1. **Nome do projeto / personagem** → nome do repo HF, da pasta e do `.safetensors`.
2. **Trigger do personagem** (default = slug do nome) → compartilhado e sempre no
   início da caption.
3. **Link/caminho das imagens** — Mega, `.zip`/`.rar`/`.7z`/`.tar`, URL, repo HF
   (`user/repo`) ou pasta local.

Ele então **inspeciona o input**:

- **Uma pasta única (ou 1 imagem)** → personagem base, sem perguntas de outfit.
- **Várias subpastas** → para **cada** subpasta pergunta:
  - *"É um outfit específico?"* → se sim: **nome/trigger do outfit** + um campo de
    **instrução livre** que entra no prompt da LLM daquele grupo.

### Como estruturar o input (multi-outfit)

Coloque cada outfit numa subpasta:

```
meu_personagem.zip
├── uniforme/      → outfit "aria_uniforme"
├── praia/         → outfit "aria_praia"
└── base/          → personagem base (responda "não é outfit")
```

Resultado: **um único LoRA** que aprende o personagem e todos os outfits. Cada
grupo vira um bloco `[[directory]]` no diffusion-pipe, com `num_repeats`
balanceado (outfits pequenos não são engolidos por um base grande).

### A ideia dos outfits (por que separar)

Descrever a roupa em detalhe deixa o LoRA "flexível" com roupas. Para outfits
**consistentes**, quando você marca uma pasta como outfit, a LLM descreve a roupa
**apenas como o trigger** (ex.: `wearing aria_praia`), sem cor/tecido/corte — e
você pode deixar uma instrução extra (ex.: *"este personagem só aparece sem
camisa ou de palhaço; nunca invente outra roupa"*).

## Uso do LoRA (ComfyUI)

Copie o `.safetensors` para `ComfyUI/models/loras/`, carregue o checkpoint base
do Anima + nó **Load LoRA**, e no prompt use o trigger do personagem no início e
o do outfit para selecionar a roupa:

```
aria, aria_praia, <cena: pose, ação, ambiente, luz>
```

## Saída no HuggingFace

Repo privado por default, com: o(s) `.safetensors`, um **model card** (tabela de
triggers, exemplos de caption por outfit, config de treino, uso no ComfyUI) e um
`animatrem_metadata.json`.

## Flags / overrides

- `python animatrem.py --advanced` — caminho antigo (receitas/hiperparâmetros
  manuais: personagem, estilo, conceito, full finetune; **sem** captioning
  embutido — você fornece os `.txt`).
- `python animatrem.py --force` — pula o menu de projetos existentes.
- `.env`:
  - `ANIMATREM_CAPTION_MODEL` (default `google/gemini-2.5-flash`)
  - `ANIMATREM_HF_PRIVATE` (`1` = privado, default)
  - `ANIMATREM_HF_NAMESPACE` (default = seu usuário HF)
  - `WORKSPACE_DIR` (default `/workspace` se existir, senão o diretório atual)

## Defaults de treino (receita do tdrussell)

Modelo `anima-base-v1.0` · recipe `character` · rank 32 · LR 2e-5 ·
`adamw_optimi` · bf16 · `sigmoid_scale=1.3` · `llm_adapter_lr=0` ·
mixed `[768,1024]` · epochs auto por exposições/imagem · GPU auto-detectada.

## Créditos

- Modelo **Anima**: [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- Engine de treino: [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)
- Captioner: [adbrasi/data_araknideo](https://github.com/adbrasi/data_araknideo)

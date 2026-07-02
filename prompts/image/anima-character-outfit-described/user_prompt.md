Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima character+outfit LoRA format (DESCRIBED mode) defined in the system prompt.

**Character trigger (subject, at the start):** `{trigger_character}`
**Outfit trigger (include as a token near the start; the clothing IS described):** `{trigger_outfit}`

Write the caption as a **single dense line in hybrid format**: start with `{trigger_character}` as the subject, add `{trigger_outfit}` as a token near the start, then mix booru tags + natural-language prose. **Describe the actual clothing** (color, fabric, cut, garments) normally — this mode keeps the outfit editable. Describe pose, expression, framing, props, environment, lighting, palette, mood.

Keep canonical concept tokens as booru tags (positions, body proportions, expressions, camera angles, counts, gaze, canonical clothing). Convert generic descriptions to prose. Do NOT include quality tags, style descriptions, or `@anything`.

Operator instruction for this dataset (obey strictly if non-empty): {custom_instruction}

**Input image:** [The attached image]
**Booru Tags (ground truth):**
{tags}

---

Front-load `{trigger_character}` and `{trigger_outfit}`, then describe the scene AND the clothing. Output only the JSON required by the system instructions.

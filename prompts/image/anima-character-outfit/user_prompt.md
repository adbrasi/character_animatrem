Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima character+outfit LoRA format defined in the system prompt.

**Character trigger (subject, at the start):** `{trigger_character}`
**Outfit trigger (write literally where clothing is mentioned, NO visual details):** `{trigger_outfit}`

Write the caption as a **single dense line in hybrid format**: start with `{trigger_character}` as the subject, then mix booru tags + natural-language prose (`{trigger_character}, tag, tag, NL clause, tag, NL clause, …`). Where the clothing would be described, write only `wearing {trigger_outfit}` / `dressed in {trigger_outfit}` — do NOT describe the outfit's color, fabric, cut, or silhouette. Describe everything else richly (pose, expression, framing, non-outfit props, environment, lighting, palette, mood).

Keep canonical concept tokens as booru tags (sexual positions, body proportions, canonical expressions, camera angles, kink scenarios, counts, gaze, monster-girl/eye features). Convert generic descriptions to prose (hair, eyes, skin, environment, lighting, mood, spatial relations). Do NOT emit clothing/outfit tags — the outfit is represented solely by `{trigger_outfit}`. Do NOT include quality tags, style descriptions, or `@anything`.

Operator instruction for this dataset (obey strictly if non-empty): {custom_instruction}

**Input image:** [The attached image]
**Booru Tags (ground truth — keep canonical ones as tags, convert generic ones to prose, ignore clothing tags):**
{tags}

---

Decide per tag: keep as booru tag (canonical concept) OR convert to prose (generic descriptive) OR drop (clothing/outfit). Use the image to refine pose, framing, expression, lighting, and color. Front-load `{trigger_character}` and place `{trigger_outfit}` once where clothing arises. Output only the JSON required by the system instructions.

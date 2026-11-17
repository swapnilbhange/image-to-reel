# The generic flow: any still image → animated Reel

Seven stages. Stages 1–4 are *thinking* (look, name, decide); 5–7 are *doing*. Skipping the thinking is what produces
the classic bad results: everything wobbles, faces melt, backgrounds show through bodies, the frame shakes.

```
0 Brief → 1 Observe → 2 Inventory → 3 Isolation plan → 4 Motion design → 5 Scene file → 6 Build + QC loop → 7 Sound + export
```

Rule of thumb that drives every decision:

> **Animate the *background and the air* a lot, the *subject* a little, the *face* never (except light).**
> Isolate an element only if something has to move **behind** it, or if **its own silhouette** has to change.

---------------------------------------------------------------------------------------------------------

## Stage 0 — Brief (30 seconds)
Decide before touching the image:
* Format: Reel/Short 1080×1920, 30 fps, 9–15 s (loops well at 10–14 s), H.264 + AAC. Keep key content out of the top ~14 % and bottom ~20 % (Instagram UI covers them).
* Fixed constraints the owner always wants (read them from the brief / past work): *frame/border stays still*, *text/logo stays still*, end card, audio on/off.
* Mood: devotional / dreamy / dramatic / cosy / eerie / epic. Mood chooses speed, colour of light and the sound bed.
* Hero: the one thing the viewer must feel is alive (a glowing eye, a flame, the sea, a rotating halo). Exactly one.

## Stage 1 — Observe  (`tools/inspect_image.py`)
```
python3 tools/inspect_image.py src.png inspect
```
Produces `overview.png` (grid every 100 px), `tile_*.png` (2× zoom, grid every 50 px), `regions.png`, `inspect.json`.
**Open them and look.** The automatic guesses are only hints; on stylised/painted art they are patchy
(in the Shiva test: it found the halo circle to within 4 px, but missed the lake, the waterfall and the face).
Read all coordinates from the grids — they are in source pixels, which is what scene.json uses.

## Stage 2 — Inventory: name every element and give it a *motion class*
Fill one row per element (template at the bottom). For each: what it is, where (bbox / polygon / point), is it in front of
something that will move, and its motion class from the table.

| Element seen | Natural motion | Effect type (`scene.json`) | Needs isolation? | Typical values (1080 px wide) |
|---|---|---|---|---|
| Sky, clouds, haze, smoke | slow drift, slight breathing of colour | `sky_drift` | No – polygon + `protect` objects | amp 4–7 px, 1 cycle/clip |
| Sun, moon, star, lamp, window, jewel, eye | pulse / flicker / flash / ramp-up | `glow` (anim pulse·flicker·flash·ramp·blink) | No | sigma 15–90 px, intensity 0.15–0.5 |
| Light shafts, aura, halo glow | rotating or pulsing rays | `rays` | Protect the subject so rays pass behind | 5–11 rays, 0.1–0.2 rev/s, intensity ≈0.15 |
| Lake / sea / pond / puddle | perspective ripple + glints | `water_ripple` | No – polygon of the surface | amp 1→4.5 px, 8–12 waves, 100 glints |
| Waterfall / river / mist / steam / fog | scrolling noise along flow | `flow` (direction, speed) | No – polygon, `mode:"bright"` keeps it to white water | 200–400 px/s |
| Wheel, mandala, halo ring, clock, gear, planet ring | rotate about the true centre (rings may counter-rotate) | `rotate_ring` + `tools/polar_fill.py` | **Yes** – whatever stands in front must be a matte; hidden ring parts are re-invented | 5–10 °/s, ease-in 1.2 s |
| Cloth, flag, scarf, ribbon, banner, curtain, sari | travelling wave from the attachment point | `sway` target `object` | **Yes** – matte of the whole figure; weight map confines the motion | amp 6–12 px, 6–7 cycles, λ 80–100 px |
| Hair, grass, leaves, foliage, fur not in the matte | same, applied to the composite | `sway` target `composite` + `protect` | No – polygon zone, matte protects skin | amp 3–6 px |
| Hovering/floating object, boat, lantern, bird | rigid bob / swing / breathe | `bob` (move, rotate, scale about a pivot) | **Yes** – its own matte | ±4–10 px, ±1–3°, 2 cycles; breathing scale 0.003 |
| Face, eyes, hands, text, logo, border | **no geometry change**; light only | `glow`, `frame_lock` | Frame/ornament mask only | – |
| Fire / candle / diya | flicker + embers rising | `glow` flicker + `particles` ember | No | flame sigma 14 px |
| Weather & atmosphere | falling / rising particles | `particles` kinds: snow rain lotus sakura rose marigold jasmine leaf autumn ember firefly sparkle dust | No (far layer behind subject, near layer in front) | 25–70 particles per layer |
| Whole picture | slow push-in / drift, bloom, vignette, colour grade | `camera`, `bloom`, `vignette`, `grade` | No | zoom 1.0→1.03 |

Pick **one hero, 2–3 supporting, then ambience**. More than ~8 simultaneous strong effects reads as noise.

## Stage 3 — Isolation plan: how to get each matte
Ask for each element that *needs* isolation (the rows marked **Yes**). Use the cheapest method that works:

| Situation | Method (`method` in scene.json) | Notes |
|---|---|---|
| Simple shape, hard edge (pole, flag, moon disc) | `polygon` (hand-drawn from the grid) | 8–20 points; soft edge via `guided` |
| Distinct colour (orange cloth on blue sky, white water) | `color` (`hsv_lo`, `hsv_hi`, optional `poly`) | Fast; clean with `close`, `specks` |
| Compact object on busy background | `grabcut` with `box` (+`fg`/`bg` points) | OK for animals, statues, objects |
| Person / complex art / many parts | `sam` with `strips` of positive/negative points | Best quality; see rules below |
| Whole portrait or product on plain background | `rembg` | One call, no prompts |
| You already have a matte | `file` (+`soft:true`) | PNG or NPY, any size |

**SAM rules learned the hard way**
* Prompt with the kwarg `sam_prompt=` (other kwargs are *silently ignored* and you get the default centre mask).
* Work in **strips** (crop ~600 px tall bands): the model resizes to 684×1024, so a tall image loses detail.
* Several small objects → several prompt sets, then union. Put positives on solid parts of one object, negatives on the thing that must *not* be included (the halo, the sky, the neighbouring limb).
* One SAM call per process (the session leaks memory → OOM); `tools/segment.py` already does this.
* Don't put positives on hair or on the pattern you want to rotate — SAM then selects the whole disc.

**Clean-up pipeline** (all keys optional, in `tools/segment.py`): `add_poly` → `cut_rects` → `close` (7) → `fill_holes` (<6000 px) → `specks` (<300 px) → `global_close` (+`gap_max`, `gap_x`) → `local_close` (boxes) → `guided` filter + `contrast`.
**Always open `work/mattes/<name>_overlay.png`.** The usual faults: missing jewellery/bracelets (background shows through a limb → use `global_close`/`local_close`), halo of old background on the edge (raise `contrast`, shrink with `cut_rects`), leaks into the frame (`cut_rects`).

**What is behind an isolated object?**
| Case | Solution |
|---|---|
| Small reveal (sway ≤ 10 px) | automatic: Telea inpaint of the lifted objects → `work/bg.png` |
| A rotating pattern is hidden behind the subject | `tools/polar_fill.py`: unroll → quilt from visible columns → roll back (`occluders`, `occluder_circles`, `bad_deg`) |
| Large reveal (subject moves a lot) | don't. Restrict the motion, or paint the plate by hand/with an inpainting model |
| Something must stay put inside a rotating ring (a medallion, an orb) | make it an object (`lift:false`, higher `z`) and list it in `occluder_circles` |

**Frame / border / logo that must stay still** → `"frame"` block: `border:[l,t,r,b]` px + optional `gold` colour classification inside `windows` for ornaments + `circles`/`rects`. It is pasted back from the original after every other effect, so petals and glows pass *under* it.

## Stage 4 — Motion design
**Timeline (14 s):** 0–1 s hook already moving · 1–3 s build to the hero event (eye opens → flash at ~2.3 s, matching a bell) · 3–11 s sustain, ambient only · last 1.5 s settle (zoom ends, audio fades). Put the best frame at ≈ 2–3 s: it becomes the cover thumbnail.
**Loops:** every periodic effect uses an **integer number of cycles per clip** (the engine rounds for you), so the clip can loop. A one-shot push-in/eased rotation is fine for a 14 s reel; set `camera.loop:true` and `rotate_ring.ease_tau:0` if you want a true loop.
**Amplitudes** are in output px at 1080 wide: wind 5–12 px, water 1–5 px, sky 4–7 px, bob 4–10 px. If you can notice the warp itself, it is too big.
**Depth:** far particles go `layer:"back"` (behind the subject, small, slow), near ones `layer:"front"` (bigger, blurred, faster). Same kind, two layers = instant depth.
**Light must respect the subject:** glow intensity ≤ 0.5, bloom threshold ≥ 190 and strength ≈ 0.16; washed-out faces are the commonest mistake.
**Sound:** bed (drone/wind/water) + one accent cue per visual event (`cues`). Loudness −16 LUFS is applied at mux.

## Stage 5 — Write `scene.json`
Start from `scene_template.json` (empty skeleton) or `examples/shiva_scene.json` (22 effects). Full parameter reference: `docs/EFFECTS.md`.
Positions are source pixels. Order of `effects` only matters within a stage (final effects are sorted automatically; the frame lock is always last).

## Stage 6 — Build and QC (the loop that actually makes it good)
```
python3 tools/segment.py     scene.json          # mattes, bg.png, frame.npy   (+ overlays to inspect)
python3 tools/polar_fill.py  scene.json          # only if there is a rotate_ring
python3 tools/audio.py       scene.json          # work/audio.wav
python3 engine/render.py     scene.json preview 0 2.4 7 13.8     # ~0.3 s per full-size still
# look at the stills at 100 % (crop faces, edges of mattes, ring seams, bracelets, frame corners), fix, repeat
python3 engine/render.py     scene.json full --workers 2          # ~75 s for 14 s @1080x1920 on 2 cores
python3 tools/qc.py          scene.json                           # sizes, frame-lock error, seam spikes, loudness, contact sheet
```
Checklist for the stills: ① background visible through limbs/jewellery? ② matte halo or cut-off hair ends? ③ ring seams or repeated art in the filled part? ④ frame corners identical to the original? ⑤ face still clean at the flash? ⑥ petals pass behind the frame? ⑦ any shimmering at polygon edges (feather more: bigger `poly` blur is built in; widen the polygon)?
Only after the stills are right do the full render — one render per request is the goal. Re-render only the segment(s) that changed.

## Stage 7 — Sound and export
`audio.mood` (`devotional`, `nature`, `cinematic`) + `layers` + `cues`, or `audio.file` for your own music. For Instagram trending audio, export silent (delete `audio`) and add the track in the app. Output: `out/<name>.mp4`, H.264 crf 17, AAC 192 k, `+faststart`.
An end card ("Film by @handle", 3 s) is appended with ffmpeg concat after the main reel (same size/fps/codec so `-c copy` works).

---------------------------------------------------------------------------------------------------------

## Failure catalogue (so you don't repeat them)
| Symptom | Cause | Fix |
|---|---|---|
| Whole chakra/background counted as foreground | wrong model (`isnet` on busy art) | SAM with point prompts |
| Every SAM run gives the same mask | wrong kwarg (`input_points`) | `sam_prompt=[{type:"point",label,data}]` |
| Process killed (exit 137) | several SAM sessions in one process | one per process |
| Background visible through leg/wrist | bracelets/hands missing from the matte | `global_close`, `local_close` boxes |
| Garbled or mirrored patch after a warp | absolute coordinates in an ROI `remap` | ROI-relative grids (`warp_roi`) |
| Wavy pole / wavy straight edges | warp weight not zero on rigid parts | `keep_still_x`, `side`, `poly`, softer `soft` |
| Orb/medallion sliced by the rotating ring | it sits inside the ring | object with `lift:false` + `occluder_circles` |
| Repeated or mirrored art in the filled ring | donor reuse | `bad_deg` for unique art; larger `seam`; the filler already penalises reuse |
| Face washed out at flash | bloom/glow too strong | glow ≤ 0.5, bloom on luminance > 190 at 0.16 |
| Segment joins flicker | different encoder settings per segment | same settings, `-c copy` concat (engine does this) |
| NaN petals | `sin()` of negative base with fractional power | clip before the power |
| Frame shimmers | effects applied after the frame | `frame_lock` is always last |

## Inventory template (copy per image)
| # | Element | Where (src px) | In front of moving thing? | Motion class → effect | Matte method | Notes/values |
|---|---|---|---|---|---|---|
| 1 | | | | | | |

## Mini worked examples
**Devotional deity with halo, frame, lake (this project).** Hero: third eye (glow ramp→flash with bell). Support: two counter-rotating ring bands (`rotate_ring`+`polar_fill`), cloth/ribbon sway (matte of the figure), hair sway (composite). Ambience: sky drift, lake ripple, waterfall `flow`, flames flicker, embers, petals back+front, rays, bloom, 3 % push-in, frame lock. Mattes: SAM strips for body/trident/cloth + bracelet closing.
**Portrait photo.** Hero: slow push-in with parallax (subject matte `bob` scale 0.003 + background `camera`), eye/jewel `glow`, hair `sway` composite (amp 3), `particles` dust/sparkle, `grade` warm. Matte: `rembg`. Never warp the face.
**Night city / street.** Hero: windows `glow` blink (points from `light_blobs`), neon `glow` flicker, `rain` front + `flow` for steam, `water_ripple` on wet road, `camera` drift. Matte: none.
**Mountain / lake landscape.** Hero: `water_ripple`; `sky_drift` big, `flow` mist in valleys, birds `bob` (objects), `rays` from the sun, `particles` leaves. Matte: polygon for birds only.
**Product shot.** Hero: moving light sweep (`rays` low intensity or animated `glow` along the edge), sparkle particles, subtle `camera`. Matte: `rembg` (to keep the product still while the background drifts).

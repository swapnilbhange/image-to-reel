---
name: image-to-reel
description: Animate any still image (devotional artwork, illustration, photo, landscape, portrait, product shot) into a 9:16 Instagram Reel / YouTube Short with natural motion - wind, water, rotating halos, glowing eyes and lamps, falling petals, snow, embers, light rays, camera push-in - plus synthesized sound. Use whenever the user uploads an image and asks to animate it, make a Reel/Short from it, or add effects like wind, water motion, flower shower, rotation or glow.
---

# Image to Reel

No video model: frames are composited from layers in Python/OpenCV and encoded with ffmpeg. One ML step (Segment Anything via `rembg`) makes cut-outs when needed.
Set `S` = this skill's folder (its base directory). Toolkit: `$S/scripts/{engine,tools}`, flow guide `$S/references/FLOW.md`, parameter reference `$S/references/EFFECTS.md`, skeleton `$S/assets/scene_template.json`, worked example `$S/assets/example/` (Shiva artwork, 22 effects, runs as is).

## Setup (once per session)
`python3 -c "import cv2, scipy; cv2.ximgproc"` and `ffmpeg -version`. If missing: `pip install --break-system-packages opencv-contrib-python-headless numpy scipy pillow rembg onnxruntime` (rembg is needed only for `sam`/`rembg` mattes). Work in a new project folder, never inside `$S`.

## Procedure (read `$S/references/FLOW.md` fully before the first image)
1. **Brief.** 1080x1920, 30 fps, 9-15 s. Note standing constraints from the user or memory (e.g. decorative frame stays still, falling flowers, end card). Choose ONE hero effect, 2-3 supporting ones, then ambience.
2. **Observe.** Copy the image to `src.png`; run `python3 $S/scripts/tools/inspect_image.py src.png inspect`. Then actually LOOK at `inspect/overview.png`, `tile_*.png`, `regions.png`; the auto-guesses are hints only (painted art: circles are found, water/faces often missed). Read coordinates from the grids (source pixels).
3. **Inventory.** List every element: where, whether it sits in front of something that will move, motion class. Mapping: sky/cloud -> sky_drift; lamp/eye/sun/jewel -> glow; light shafts -> rays; lake/sea -> water_ripple; waterfall/mist/smoke -> flow; wheel/halo/mandala -> rotate_ring (+polar_fill); cloth/flag/ribbon -> sway (object); hair/grass not in a matte -> sway (composite); hovering things -> bob; petals/snow/rain/embers/fireflies -> particles (far layer behind, near layer in front); border/logo -> frame_lock; whole picture -> camera, bloom, vignette, grade. Faces never warp.
4. **Isolation plan.** Isolate only what must be in front of something moving or whose own silhouette moves. Cheapest method that works: polygon, color, grabcut, rembg (plain background), sam (point prompts in ~600 px strips). Use `fit_circle.py` for exact ring centres. Hidden parts of rotating patterns come from `polar_fill.py`.
5. **Write `scene.json`** from `$S/assets/scene_template.json` (positions in source px). Keep amplitudes small (wind 5-12 px, water 1-5, sky 4-7 at 1080 wide); glow <= 0.5; bloom threshold >= 190.
6. **Build and QC loop** (from the project folder):
   `python3 $S/scripts/tools/segment.py scene.json` -> inspect `work/mattes/*_overlay.png` (see-through limbs? halos? leaks into the frame?)
   `python3 $S/scripts/tools/polar_fill.py scene.json` (only with a rotate_ring)
   `python3 $S/scripts/tools/audio.py scene.json`
   `python3 $S/scripts/engine/render.py scene.json preview 0 2.4 7 13` -> view stills at 100 %, fix, repeat
   `python3 $S/scripts/engine/render.py scene.json full --workers 2` (about 75 s per 14 s) -> `python3 $S/scripts/tools/qc.py scene.json`
   Aim for ONE full render per request; re-render only what changed.
7. **Deliver** the mp4 with SendUserFile and one line on what moves. Mention that sound is synthesized and can be replaced with trending audio inside Instagram; offer an end card.

## Hard-won rules
- SAM prompts go in the `sam_prompt=` kwarg (others are silently ignored); one SAM call per process (OOM otherwise); crop into strips; no positives on hair or on the pattern that must rotate.
- Bracelets/jewellery missing from a matte show the background through limbs: use `global_close` / `local_close`.
- Sway weights must be zero on rigid parts (poles, standing leg): use `side`, `poly`, `keep_still_x`.
- Something sitting inside a rotating ring (orb, medallion) = object with `lift:false`, higher `z`, plus `occluder_circles`.
- The frame mask is always pasted last, so particles and glows pass under it.
- Previously used pitfalls and fixes are tabulated at the end of FLOW.md.

## If a request needs an effect the engine lacks
Add a class in `scripts/engine/effects.py` (see its docstring for the stage methods), keep motion periodic with integer cycles, and list it in your scene file.

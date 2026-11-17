# scene.json reference

All positions / sizes marked *src px* are in SOURCE-image pixels (read them from `tools/inspect_image.py` grids).
Time values are seconds; `cycles` are integer repeats per clip.

## Top level
| key | meaning | default |
|---|---|---|
| `image` | source file (relative to scene.json) | required |
| `out_size` | `[W,H]` | `[1080,1920]` |
| `fps`, `duration` | | 30, 14 |
| `work_dir` | where mattes / plates / previews go | `work` |
| `output` | final mp4 | `out/reel.mp4` |
| `objects` | list of mattes (below) | `[]` |
| `frame` | still border / ornaments | none |
| `effects` | list of effects (below) | `[]` |
| `audio` | see `tools/audio.py` | none |

## objects[]
`name`, `method` (`sam` `grabcut` `color` `polygon` `rembg` `file`), `z` (draw order, higher = in front), `lift` (default true: removed from the
background plate; false for things that stay put on top, e.g. an orb), plus method keys and clean-up keys — see the docstring of `tools/segment.py`.

## frame
`border:[l,t,r,b]` px, `gold:{hsv_lo,hsv_hi,windows:[[x0,y0,x1,y1]..],fill_holes}`, `circles:[[x,y,r]..]`, `rects:[[x0,y0,x1,y1]..]`.

## effects[] — every entry has `type`; `off:true` disables it

| type | stage | key parameters |
|---|---|---|
| `sky_drift` | bg | `poly`, `exclude_poly`, `protect:[objects]`, `amp` (px), `cell` (noise size), `seed` |
| `water_ripple` | bg+light | `poly`, `amp_far`, `amp_near`, `waves`, `cycles`, `perspective`, `shimmer`, `glints`, `glint_color` |
| `flow` | bg | `poly`, `mode` bright\|all (`vmin`,`smax`), `direction` down\|up\|left\|right, `speed` (px/s), `strength`, `highlight` |
| `rotate_ring` | bg | `center`, `r_in`, `r_out`, `deg_per_sec` (− = clockwise), `ease_tau`, `id`, `feather`; for `polar_fill`: `occluders`, `occluder_circles`, `bad_deg`, `seam` |
| `sway` | obj / post | `object` or `target:"composite"` (+`protect`), `anchor`, `reach`, `dist` radial\|x\|y, `side` right\|left\|above\|below, `soft`, `power`, `poly`, `protect_polys`, `keep_still_x:[[x,halfwidth]]`, `amp:[ax,ay]`, `cycles`, `wavelength`, `wave_dir`, `phase`, `gain` |
| `bob` | obj | `object`, `pivot`, `move:[ax,ay]`, `rotate` (deg), `scale`, `cycles`, `phase` |
| `glow` | light | `pos` or `points`, `color` RGB, `sigma`, `rx`/`ry`, `intensity`, `anim` const\|pulse\|flicker\|flash\|ramp\|blink, `period`, `t0`, `t1`, `width`, `floor` |
| `rays` | light | `center`, `r0`, `r1`, `sets:[[n,rev_per_s,phase,weight]]`, `intensity`, `color`, `sharpness`, `protect:[objects]` |
| `particles` | back/front/light | `kinds` (lotus sakura rose hibiscus marigold jasmine leaf autumn snow rain ember firefly sparkle dust star), `layer` back\|front\|light, `count`, `size:[min,max]`, `speed:[min,max]` px/s, `wind:[min,max]`, `direction` down\|up, `color`, `alpha`, `blur_above`, `seed` |
| `bloom` | final | `threshold`, `sigma`, `strength` |
| `vignette` | final | `strength` |
| `grade` | final | `gain:[r,g,b]`, `contrast`, `saturation` |
| `camera` | final | `zoom0`, `zoom1`, `center`, `pan:[dx,dy]` src px, `loop` |
| `frame_lock` | final (last) | automatic when a `frame` block exists |

## Adding a new effect
Subclass `Effect` in `engine/effects.py`, decorate with `@effect('name')`, implement any of `bg / back / obj / post / light / front / final`
(see the docstring at the top of the file). Keep motion periodic with `cyc()` so loops stay seamless.

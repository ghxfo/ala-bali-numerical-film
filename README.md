# على بالي — Ala Bali (numerical film)

A live, generative animation built entirely from thousands of tiny digits, symbols and punctuation marks.
It interprets the lyrics *على بالي ولا انت داري* in four scenes: a woman whose thoughts circle endlessly,
the nights passing under a moon made of numbers, a calendar of years flipping by, and every word he said
closing in on her while he walks away, unaware.

Nothing is pre-rendered: running the script opens a window and the film plays in real time.

## Run it

```
pip install pygame-ce numpy pillow
python ala_bali.py
```

> If `pip install pygame` fails with *"Failed to build 'pygame'"*, use `pygame-ce` as above —
> it's a drop-in replacement that supports newer Python versions.

## Controls

| Key | Action |
| --- | --- |
| `SPACE` | pause / resume |
| `R` | restart |
| `H` | timing HUD (scene, time, fps) |
| `L` | Arabic lyric captions (needs `pip install arabic-reshaper python-bidi`) |
| `ESC` | quit |

## Tweaking

Everything is in the `CFG` class at the top of `ala_bali.py`: lyric timings (`SEGMENTS`),
colours (`PALETTE`), character density (`DENSITY`), speed (`SPEED`, `MOTION`), calendar years,
moon cycles, and an optional `AUDIO_FILE` to play the song in sync.

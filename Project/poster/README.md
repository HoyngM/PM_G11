# Poster (A2 portrait)

`poster.pdf` is the poster for the poster session; `poster.png` is a preview of it.

Every number, table and chart on the poster is read from the result files in `results/`,
so the poster always shows what the code produced. Nothing is typed in by hand.

## Files

- `poster_template.html`: all text and layout of the poster.
- `build_poster.py`: fills the template with the numbers, tables and charts (inline SVG) from `results/`.
- `results/`: output of `main.py` on `main` (6a45964). `results/README.txt` lists the exact commands.
- `jads.png`: the JADS logo (same file as `Project/jads.png`).

## Rebuild the poster

Run from this folder, with the Python environment of `Project/code` (it needs pandas and numpy):

```bash
python build_poster.py --code ../code --pdf
```

This writes `poster.html`, `poster.pdf` and `poster.png`. The PDF and PNG are printed with
Microsoft Edge in headless mode. To change the text, edit `poster_template.html` and rebuild.

If the code is run again, copy its `results/` folder over `results/` here and rebuild,
so the poster and the paper show the same numbers.

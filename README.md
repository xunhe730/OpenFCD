# OpenFCD

**Fast Checkerboard Demodulation for free-surface reconstruction — desktop app + CLI.**

Reconstructs the free-surface height field η(x, y, t) from high-speed video of
a checkerboard pattern viewed through water, using the Wildeman / Moisy
*synthetic schlieren* FCD algorithm. Designed for thin-film wave studies,
sloshing experiments, mm-scale swimming-robot wakes, and any free-surface
experiment that films a patterned backdrop through a transparent fluid.

## Status

**v0.0.1** — early release. Core pipeline and GUI are functional; expect
rough edges. Contributions welcome.

## Features

- **End-to-end desktop GUI** (PyQt6) with a STAR-CCM+-style 3-pane layout:
  simulation tree · scene tabs · properties.
- **First-class CLI** (`openfcd run path.ofcd`) — GUI and CLI share the same
  pipeline, producing byte-identical `results.h5`.
- **Per-frame debug mode** — compute a single frame, tune parameters live with
  a slider, re-compute without re-running the whole sequence.
- **Robust to imperfect reference frames**:
  - Automatic carrier-period *scale normalization* when the reference was
    captured at a slightly different zoom
  - Optional spatial high-pass to remove non-physical low-frequency drift from
    cross-session ref/def pairs
- **Project files on disk** — `*.ofcd/` directories contain a git-friendly
  `project.yaml`, snapshot runs under `runs/{id}/`, and annotations under
  `annotations/default.json`.
- **HDF5 results** with SWMR so scripts can tail the output while a run is in
  progress.

## Installation

Requires Python 3.11+.

```bash
# CLI only
pip install openfcd

# CLI + desktop GUI (PyQt6)
pip install 'openfcd[gui]'
```

### Development install

```bash
git clone https://github.com/xunhe730/OpenFCD.git
cd OpenFCD
pip install -e '.[gui,dev]'   # editable, with GUI + test/lint extras
```

## Quick start

### GUI

```bash
python -m openfcd.gui
# or
python -m openfcd
```

Workflow:

1. **File → New Project** → point at a folder of image frames, set the
   checker cell side length (mm), window thickness, fluid depth.
2. Click a frame in the tree → **Set as Reference** (pick a flat-water frame
   captured in the same session).
3. Draw an ROI + optional occlusion polygon on the preview.
4. Click **Compute This Frame** to preview one frame.
5. Tune the **Highpass σ** slider if the result shows large-scale drift.
6. Click **Run All Frames** when satisfied.

### CLI

```bash
# Create a new project
openfcd new myproject --image-folder /path/to/frames --pattern 'Img*.jpg'

# Edit myproject.ofcd/project.yaml for geometry & process parameters,
# then run
openfcd run myproject.ofcd

# Results land under myproject.ofcd/runs/{timestamp}/results.h5
```

## Algorithm notes

FCD measures η by comparing the apparent displacement of a checkerboard
pattern seen through the water surface. The algorithm steps:

1. **Flatfield** (`openfcd.core.flatfield`) — divide out smooth illumination
   variation using a Gaussian background.
2. **Carrier detection** (`openfcd.core.fcd.calculate_carriers`) — find the
   two dominant spatial-frequency peaks of the checkerboard in k-space.
3. **Scale normalization** (`openfcd.core.registration`) — if the reference
   and deformed frames were shot at slightly different zooms, rescale the
   reference to match the deformed frame's carrier period. Triggers at 0.5 %
   mismatch.
4. **Occlusion inpaint** — mask the robot/object and replace with a
   synthesized carrier field so the FCD demodulation sees continuous texture.
5. **Cosine taper** — soften image boundaries to suppress FFT edge artifacts.
6. **Demodulation + phase unwrap** (`openfcd.core.fcd.fcd`) — extract the
   pixel-displacement field (u, v) at each carrier.
7. **Poisson integration** (`fftinvgrad`) — integrate (u, v) to get the
   apparent depth, then calibrate to mm using the optical stack geometry.
8. **Post-processing** — plane detrend, filament suppression, optional
   high-pass filter, edge NaN margin.

`geometry.pattern_period_mm` is the legacy project-file key, but its physical
meaning is `checker_cell_mm`: the side length of one single checkerboard cell,
not the full black-white cycle. The carrier calibration assumes
`|k_phys| = π√2 / checker_cell_mm`.

Physical correctness depends on the reference frame. Best results come from
a flat-water reference captured in the same acquisition session as the
deformed frames. For cross-session references the high-pass filter (slider
in the GUI, `process.highpass_sigma_px` in project.yaml) removes
low-frequency drift.

## Project structure

```
openfcd/
├── core/          # FCD algorithm (fcd, flatfield, inpaint, mask, …)
├── geometry/      # Optical-stack calibration
├── io/            # project.yaml schema, HDF5 results, annotations
├── pipeline/      # Stage protocol + orchestration
├── cli/           # Typer CLI (new / run / replay / project)
└── gui/           # PyQt6 GUI (main window, panels, scenes, renderers)
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check openfcd tests
```

Golden-comparison tests (`tests/golden/`) require a local BOS reference
dataset. Set `OPENFCD_BOS_ROOT` to the dataset root to enable them; otherwise
they are skipped.

## License

[MIT](LICENSE) — see `LICENSE` for full text.

## Acknowledgements

The FCD method is due to Moisy, Rabaud & Salsac (2009), "A synthetic
Schlieren method for the measurement of the topography of a liquid
interface", and Wildeman (2018), "Real-time quantitative Schlieren imaging
by fast Fourier demodulation of a checkered backdrop". This project's
reference implementation follows Wildeman's **pyfcd** approach, extended
with a PyQt6 GUI, CLI, and engineering polish for experimental use.

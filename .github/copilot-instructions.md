# Copilot instructions for `nanopore-it`

## Build, lint, and run commands

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

Run the Streamlit app:

```bash
export APP_PASSWORD="your_secure_password"
uv run streamlit run app.py
```

Automated tests are not currently configured in this repository (no `tests/` suite and no test runner dependency in `pyproject.toml`).

## High-level architecture

- `app.py` is the orchestration layer: it authenticates users, loads `.opt` files, applies the Bessel low-pass filter, optionally masks auto-detected clear regions, and then calls the package analysis pipeline.
- `nanopore_it/analysis.py` is the core event pipeline. It converts filtered traces into `AnalysisTables` (`events`, `states`, `n_children`) by:
  - detecting event start/end candidates from threshold crossings,
  - refining boundaries against baseline statistics,
  - computing event features (dwell, delli, higher moments, FFT summary),
  - optionally running subevent segmentation with CUSUM.
- `nanopore_it/cusumv3.py` provides the state detector used by `analysis.py`; it preprocesses with central moving average/median and runs a performance-optimized single-pass CUSUM loop.
- `nanopore_it/auto_detect_clears.py` implements clear-region detection shared by the UI and core library API.
- `nanopore_it/__init__.py` defines the stable import surface (`AnalysisConfig`, `AnalysisTables`, `analyze_tables`, `detect_clear_regions`, `detect_cusumv2`).

## Key conventions in this codebase

- Keep heavy Streamlit computations cached with `@st.cache_data(max_entries=1)` (including wrapped package functions in `app.py`).
- Signal files are parsed as big-endian float64 (`np.dtype(">d")`) and analyzed as NumPy arrays; preserve this input assumption when adding loaders.
- Auto-detected clear regions are excluded by setting signal slices to `NaN`; downstream baseline/stat computations that should ignore excluded data use `nan`-aware operations.
- Units are intentional across UI and analysis:
  - sidebar samplerate/cutoff inputs are in kHz, then converted to Hz for computation,
  - dwell values are reported in microseconds,
  - event/state amplitude features are baseline-relative (`delli`, `frac`).
- `AnalysisConfig` / `AnalysisTables` are frozen keyword-only dataclasses and are treated as the contract between UI and analysis code.
- CUSUM result keys and some parameter names intentionally use legacy naming (`nStates`, `starts`, `threshhold`); keep compatibility when touching these interfaces.
- Type hints are strict and use `numpy.typing` and `TypedDict` extensively; keep new analysis code fully typed.

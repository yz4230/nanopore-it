# Repository Guidelines

## Project Structure & Module Organization

`app.py` is the Streamlit entry point and owns UI flow, authentication, upload handling, caching, and Plotly rendering. Core analysis code lives in `nanopore_it/`: `analysis.py` builds event and state tables, `auto_detect_clears.py` finds representative baseline regions, `cusumv3.py` implements CUSUM state detection, and `utils.py` contains shared loading/signal helpers. `notebook.ipynb` is exploratory. Local sample files belong in `data/`, which is ignored by git. There is no checked-in `tests/` directory yet.

## Build, Test, and Development Commands

Use `uv` for all environment and command execution.

- `uv sync`: install runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `APP_PASSWORD=your_password uv run streamlit run app.py`: run the local web app.
- `uv run ruff check .`: lint Python files.
- `uv run ruff format .`: format Python files.
- `uv run ty check`: run static type checks using the configured `.venv` environment.

Do not start the Streamlit server just to verify changes unless explicitly requested. For routine validation, `uv run ruff check .` and `uv run ty check` are sufficient.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, type hints on public helpers and numerical arrays, and dataclasses for structured configuration/results. Prefer NumPy/Pandas/SciPy vectorized operations in analysis paths; avoid slow Python loops over large signal arrays unless profiling justifies them. Keep Streamlit-expensive operations cached with `@st.cache_data` when inputs are stable. Use `snake_case` for functions and variables, `PascalCase` for dataclasses and typed dictionaries, and uppercase names for module constants such as table headers.

## Testing Guidelines

No automated test suite is currently present. Add tests under `tests/` when changing analysis behavior, file parsing, or numerical edge cases. Use names like `tests/test_analysis.py` and `test_empty_segment_returns_nan()`. Prefer small deterministic arrays over large fixture files; if fixture `.opt` files are necessary, keep them minimal and document their origin. Run linting and type checks before opening a PR.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit-style prefixes such as `feat:`, `fix:`, `refactor:`, `chore:`, and `dev:`. Keep subjects imperative and focused, for example `fix: handle empty event windows`. Pull requests should include a short problem summary, key implementation notes, validation commands run, and screenshots or screen recordings for Streamlit UI changes. Link related issues when available.

## Security & Configuration Tips

Do not commit `.env`, uploaded datasets, or generated analysis outputs. Use `.env.example` as the template and set `APP_PASSWORD` locally. Treat nanopore datasets as potentially sensitive; keep local files in ignored `data/` unless maintainers explicitly approve adding sanitized fixtures.

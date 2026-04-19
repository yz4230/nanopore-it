# nanopore-it Project Context

## Project Overview

`nanopore-it` is a high-performance Python application designed for analyzing Nanopore data. It features a web-based interactive interface built with Streamlit and Plotly for real-time signal visualization and event characterization. 

The core analysis library is located in the `nanopore_it/` directory, which performs tasks such as:
- Automated event detection and translocation event characterization.
- Clear region auto-detection to filter non-representative signal regions.
- Sub-event state detection using CUSUM algorithms.
- Frequency analysis using FFT.

**Main Technologies:**
- **Language**: Python 3.14+
- **Frontend/UI**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Processing & Analysis**: NumPy, Pandas, SciPy
- **Package & Dependency Management**: `uv`

## Architecture & Key Files

- `app.py`: The main entry point and Streamlit frontend application logic. It handles data loading, filtering (Bessel filter), authentication, and coordinates the UI components.
- `pyproject.toml`: Contains project metadata, dependencies, and environment configurations. `uv` is the standard tool used here.
- `nanopore_it/`: The core backend package containing the analytical heavy lifting.
  - `analysis.py`: Main analysis pipeline, containing data structures (`AnalysisConfig`, `AnalysisTables`) and functions to process signal data into event tables.
  - `auto_detect_clears.py`: Logic for detecting representative baseline clear signal regions.
  - `cusumv3.py`: Implements CUSUM-based state detection algorithms for complex multi-state events.

## Building and Running

The project leverages `uv` for modern Python dependency management and environments.

### Installation
```bash
# Install all dependencies and create the virtual environment
uv sync
```

### Running the Application
The application is secured with an environment-variable-based password. To run the Streamlit app:
```bash
# Set the application password
export APP_PASSWORD="your_secure_password"

# Run the Streamlit server
uv run streamlit run app.py
```

## Development Conventions

- **Typing**: The project relies heavily on type hints, specifically utilizing `numpy.typing` (e.g., `npt.NDArray`) and standard Python `typing` constructs. Ensure any new code maintains strict type annotations.
- **Linting & Formatting**: The `dev` dependency group in `pyproject.toml` includes `ruff` and `ty`. Use `ruff check` and `ruff format` to adhere to established coding standards.
- **Performance**: The core analysis logic utilizes optimized NumPy and SciPy operations to avoid Python-level loops where possible, maximizing performance for large signal datasets.
- **Caching**: The Streamlit frontend uses `@st.cache_data` extensively on expensive operations like `load_data`, `draw_signal`, and analysis functions to prevent unnecessary recomputations upon UI interactions. Maintain this pattern for heavy computations.
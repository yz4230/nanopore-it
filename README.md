# nanopore-it

A high-performance Nanopore data analysis tool with a Streamlit interface.

## Features

- **Real-time Signal Visualization**: Interactive Plotly charts for exploring raw nanopore signals.
- **Automated Event Detection**: Advanced algorithms for identifying and characterizing translocation events.
- **Clear Region Auto-detection**: Automatically identifies and filters out non-representative signal regions.
- **CUSUM State Analysis**: Sub-event state detection using CUSUM algorithms for complex multi-state events.
- **Frequency Analysis**: FFT-based spectrum comparison between baseline and event signals.
- **Authentication**: Secure access with environment-variable-based password protection.

## Installation

This project uses `uv` for dependency management.

```bash
# Clone the repository
git clone https://github.com/yz/nanopore-it.git
cd nanopore-it

# Install dependencies
uv sync
```

## Usage

1. Set the application password in your environment:
   ```bash
   export APP_PASSWORD="your_secure_password"
   ```

2. Run the Streamlit application:
   ```bash
   uv run streamlit run app.py
   ```

3. Upload your `.opt` data files and adjust analysis parameters in the sidebar.

## Project Structure

- `app.py`: Streamlit frontend and application logic.
- `nanopore_it/`: Core analysis library.
  - `analysis.py`: Main analysis pipeline and event characterization.
  - `auto_detect_clears.py`: Logic for detecting representative signal regions.
  - `cusumv3.py`: CUSUM-based state detection algorithms.
- `pyproject.toml`: Project metadata and dependencies.

## Technologies

- **Streamlit**: Web interface.
- **Plotly**: Interactive visualizations.
- **NumPy & Pandas**: Data processing and analysis.
- **SciPy**: Signal processing and statistical functions.
- **uv**: Modern Python packaging and dependency management.

# AGI Timelines Model (METR)

This Streamlit app allows you to interactively forecast AGI timelines using the METR model, based on the paper ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499) and insights from [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell).

## Features

- **Interactive Parameter Controls:**
  - Set start task length, AGI task length, doubling time, acceleration, and shift.
  - Choose from preset models (e.g., o3, Claude, GPT-4o, DeepSeek, etc.) for start task length.
  - Advanced mode unlocks additional parameters: elicitation boost, reliability, task complexity, and reference date.
  - Select from preset doubling time models or use a custom value.

- **Results Visualization:**
  - Histogram of AGI arrival years.
  - Exponential growth plot of task length over time, showing:
    - Median, 10% earliest, and 90% latest curves (with AGI achievement markers).
  - Computed AGI dates for median, 10% earliest, and 90% latest scenarios, all in long date format.
  - Yearly probability table for AGI arrival.

- **Robust Error Handling:**
  - Handles NaN values and outliers in model results.
  - Displays warnings and fallback results if model execution fails.

## Setup Instructions

### 1. Clone the Repository
   ```bash
git clone <repo-url>
   cd agi_timelines
   ```

### 2. Install Dependencies
It is recommended to use a virtual environment:
   ```bash
python3 -m venv venv
source venv/bin/activate
   pip install -r requirements.txt
   ```

**Required packages include:**
- streamlit
- squigglepy
- numpy
- matplotlib
2. Install dependencies using Poetry:
   ```bash
   # Install Poetry if you don't have it
   pip install poetry
   
   # Install dependencies
   poetry install
   ```

3. Run Jupyter notebook:
   ```bash
   poetry run jupyter notebook
   ```

If you don't have a `requirements.txt`, create one with:
```
streamlit
squigglepy
numpy
matplotlib
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

## Usage
- Adjust parameters in the sidebar to explore different AGI timeline scenarios.
- Use "Advanced Mode" for more detailed control and to select reference models or adjust reliability, elicitation, and task complexity.
- View the results in the main panel:
  - Histogram of AGI arrival years
  - Exponential growth plot (with three percentile curves)
  - Computed AGI dates for median, 10% earliest, and 90% latest
  - Yearly probability table

## Notes
- The app uses 100,000 samples for model runs by default for robust results.
- The exponential growth plot is a simplified visualization and may not reflect all stochasticity in the full model, but is parameterized to match the sidebar settings.
- All results update live as you change parameters and rerun the model.

## References
- [Measuring AI Ability to Complete Long Tasks (METR paper)](https://arxiv.org/abs/2503.14499)
- [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell)

---

For questions or issues, please open an issue on GitHub or contact the project maintainer.

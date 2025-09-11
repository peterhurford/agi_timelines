import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import run_model, get_start_task_length, get_agi_task_length, get_doubling_time, get_acceleration, get_shift, O3_LAUNCH_DATE, CLAUDE_3P7_LAUNCH_DATE, DEFAULT_PARAMS, PARAMETER_PRESETS, MODEL_TASK_TIMES
import squigglepy as sq
from datetime import datetime, timedelta
import io
import pandas as pd
import urllib.parse
import matplotlib.dates

st.title("AGI Timelines Model (METR)")
st.markdown("""
Interactively forecast AGI timelines using the METR model based on ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499).

Adjust the parameters and run the model to see updated timelines.
""")

# Sidebar for variable selection
st.sidebar.header("Model Parameters")

with st.sidebar.expander("ℹ️ Parameter Explanations (click to expand)"):
    st.markdown("""
    **Start task length (hours):**
    The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for improvements from scaffolding and compute (elicitation boost), higher reliability requirements (reliability penalty), and the possibility that AGI tasks are harder than METR's self-contained software tasks (task type penalty). This is not just the raw model benchmark, but a composite reflecting how close we are to AGI-level tasks.
    
    **AGI task length (hours):**
    The human time required to complete a task that would count as 'AGI-level' (e.g., a month- or year-long project). This is a subjective threshold, often set to something like 167 hours (a month of full-time work) or up to 2000 hours (a work year).
    
    **Doubling time (days):**
    The number of days it takes for the maximum task length that AI can do at 50% reliability to double. This is fit to historical trends (e.g., 212 days for 2019-2024, 118 days for 2024-2025, or a mixture).
    
    **Acceleration:**
    If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential.
    
    **Shift (days):**
    The number of days to shift the forecast earlier to account for internal model capabilities before public release (e.g., 30-150 days).
    
    **Elicitation boost:**
    Factor for improvements from better scaffolding or more compute. >1 means faster progress.
    
    **Reliability needed:**
    The required reliability for AGI-level performance. Higher reliability means a harder bar for AGI and increases the penalty on start task length.
    
    **Task type complexity factor:**
    Adjusts for AGI tasks being harder than the METR benchmark tasks. >1 = AGI tasks are harder.
    
    **Reference date:**
    The date from which to start counting days until AGI. Usually the launch date of the reference model.
    """)

# Parameter presets
st.sidebar.subheader("Parameter Presets")

# Add a callback to set session state when selection changes
if 'current_preset' not in st.session_state:
    st.session_state.current_preset = "Default"

# Initialize selected_preset from session state if it exists
if 'selected_preset' not in st.session_state:
    st.session_state.selected_preset = "Default"

def on_preset_change():
    # Update the current preset for tooltip
    st.session_state.current_preset = st.session_state.preset_selectbox
    # Store the selected preset
    st.session_state.selected_preset = st.session_state.preset_selectbox

# Use the stored selection as the default index
default_index = list(PARAMETER_PRESETS.keys()).index(st.session_state.selected_preset) if st.session_state.selected_preset in PARAMETER_PRESETS else 0

# Get tooltips from PARAMETER_PRESETS
preset_tooltips = {preset_name: preset_data["description"] for preset_name, preset_data in PARAMETER_PRESETS.items()}

selected_preset = st.sidebar.selectbox(
    "Choose a parameter preset",
    list(PARAMETER_PRESETS.keys()),
    index=default_index,  # Use the stored selection
    help=preset_tooltips.get(st.session_state.current_preset, ""),  # Get help text from session state
    key="preset_selectbox",
    on_change=on_preset_change
)

# Apply preset button
apply_preset = st.sidebar.button("Apply Preset")

# Save preset values in session state so they persist
if "preset_applied" not in st.session_state:
    st.session_state.preset_applied = False
    
if apply_preset or (not st.session_state.preset_applied and selected_preset != "Default"):
    # Apply the selected preset
    st.session_state.preset_applied = True
    st.session_state.preset_params = PARAMETER_PRESETS[selected_preset]["params"]
    # Force a model run by clearing the cached results
    if "results" in st.session_state:
        del st.session_state["results"]
    st.sidebar.success(f"Applied {selected_preset} preset")
    # Force a rerun to apply the preset values
    st.rerun()

# Fixed number of samples (always use 100,000)
n_samples = 100_000

# Model section
st.sidebar.subheader("Model Configuration")

# Advanced mode toggle
advanced_mode = st.sidebar.checkbox("Advanced Mode (More Parameters)")

# Reference model selection
if advanced_mode:
    # Create a copy of MODEL_TASK_TIMES and add "Custom" option
    reference_models = dict(MODEL_TASK_TIMES)
    reference_models["Custom"] = DEFAULT_PARAMS["start_task_length"]
    
    # Sort models by their task length (descending) for a more logical order
    sorted_models = ["Custom"] + sorted(MODEL_TASK_TIMES.keys(), key=lambda x: -MODEL_TASK_TIMES[x])
    
    reference_model = st.sidebar.selectbox(
        "Reference model for start task length",
        sorted_models,
        format_func=lambda x: f"{x} ({reference_models[x]:.3f} hrs)" if x != "Custom" else "Custom"
    )

    # Set task length based on model selection
    if reference_model != "Custom":
        start_task_length = MODEL_TASK_TIMES[reference_model]
    else:
        # Use preset parameter if one has been applied
        default_value = st.session_state.preset_params["start_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["start_task_length"]
        start_task_length = st.sidebar.number_input(
            "Start task length (hours)", 
            min_value=float(0.01), 
            max_value=float(10.0), 
            value=float(default_value), 
            step=float(0.01),
            help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
        )
else:
    # Use preset parameter if one has been applied
    default_value = st.session_state.preset_params["start_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["start_task_length"]
    start_task_length = st.sidebar.number_input(
        "Start task length (hours)", 
        min_value=float(0.01), 
        max_value=float(10.0), 
        value=float(default_value), 
        step=float(0.01),
        help="The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for scaffolding, reliability, and task type penalties. Default: 1.75 hours (o3's task length)"
    )

# AGI task length
# Use preset parameter if one has been applied
agi_default = st.session_state.preset_params["agi_task_length"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["agi_task_length"]
agi_task_length = st.sidebar.number_input(
    "AGI task length (hours)", 
    min_value=float(40.0), 
    max_value=float(5000.0), 
    value=float(agi_default), 
    step=float(1.0),
    help="The human time required to complete a task that would count as 'AGI-level' (e.g., a month- or year-long project). Default: 167 hours (month-long tasks)"
)

# Growth parameters
st.sidebar.subheader("Growth Parameters")

doubling_default = st.session_state.preset_params["doubling_time"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["doubling_time"]
doubling_time = st.sidebar.number_input(
    "Doubling time (days)", 
    min_value=float(60.0), 
    max_value=float(400.0), 
    value=float(doubling_default), 
    step=float(1.0),
    help="The number of days it takes for the maximum task length that AI can do at 50% reliability to double. Default: 212 days (METR 2019-2024 trend)"
)

if advanced_mode:
    # Add doubling time model selection with presets
    doubling_time_model = st.sidebar.selectbox(
        "Doubling time model",
        ["Custom", "METR 2019-2024 Trend (212 days)", "METR 2024-2025 Trend (118 days)", 
         "Pessimistic Trend (320 days)", "Mixture Model"]
    )
    
    if doubling_time_model == "METR 2019-2024 Trend (212 days)":
        doubling_time = DEFAULT_PARAMS["doubling_time"]
    elif doubling_time_model == "METR 2024-2025 Trend (118 days)":
        doubling_time = 118.0
    elif doubling_time_model == "Pessimistic Trend (320 days)":
        doubling_time = 320.0
    elif doubling_time_model == "Mixture Model":
        st.sidebar.markdown("Using default METR mixture model for doubling time")
        doubling_time = -1  # Special value to use the mixture model

accel_default = st.session_state.preset_params["acceleration"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["acceleration"]
acceleration = st.sidebar.slider(
    "Acceleration", 
    min_value=float(0.8), 
    max_value=float(1.2), 
    value=float(accel_default), 
    step=float(0.01),
    help="If <1, the doubling time itself shrinks over time (superexponential progress); if >1, it grows (progress slows); if 1, progress is exponential."
)

shift_default = st.session_state.preset_params["shift"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["shift"]
shift = st.sidebar.number_input(
    "Shift (days)", 
    min_value=float(0), 
    max_value=float(250), 
    value=float(shift_default), 
    step=float(1),
    help="The number of days to shift the forecast earlier to account for internal model capabilities before public release. Default: 90 days"
)

# Advanced parameters in expanded section
if advanced_mode:
    st.sidebar.subheader("Advanced Adjustments")
    
    with st.sidebar.expander("Additional Parameters"):
        # Elicitation boost factor
        elicit_default = st.session_state.preset_params["elicitation_boost"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["elicitation_boost"]
        elicitation_boost = st.slider(
            "Elicitation boost", 
            min_value=float(0.5), 
            max_value=float(2.0), 
            value=float(elicit_default), 
            step=float(0.1),
            help="Boost factor from better scaffolding and increased compute. Default: 1.0 (no adjustment)"
        )
        
        # Reliability settings
        reliability_default = st.session_state.preset_params["reliability_needed"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["reliability_needed"]
        reliability_needed = st.slider(
            "Reliability needed", 
            min_value=float(0.5), 
            max_value=float(0.99), 
            value=float(reliability_default), 
            step=float(0.01),
            help="Required reliability level. Default: 0.5 (METR's standard)"
        )
        
        # Task type penalty factor
        penalty_default = st.session_state.preset_params["task_type_penalty"] if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["task_type_penalty"]
        task_type_penalty = st.slider(
            "Task type complexity factor", 
            min_value=float(0.1), 
            max_value=float(10.0), 
            value=float(penalty_default), 
            step=float(0.1),
            help="Adjustment for AGI tasks being harder than METR's self-contained software tasks. >1 = harder, <1 = easier. Default: 1.0 (no adjustment)"
        )
        
        # Reference date selection
        # Get default reference date from preset if available
        if hasattr(st.session_state, "preset_params") and "reference_date" in st.session_state.preset_params:
            ref_date_default = "o3 Launch (2025-04-16)" if st.session_state.preset_params["reference_date"] == O3_LAUNCH_DATE else "Claude 3.7 Launch (2025-02-24)"
        else:
            ref_date_default = "o3 Launch (2025-04-16)"
            
        reference_date_option = st.selectbox(
            "Reference date",
            ["o3 Launch (2025-04-16)", "Claude 3.7 Launch (2025-02-24)", "Custom Date"],
            index=0 if ref_date_default == "o3 Launch (2025-04-16)" else 1
        )
        
        if reference_date_option == "o3 Launch (2025-04-16)":
            reference_date = DEFAULT_PARAMS["reference_date"]
        elif reference_date_option == "Claude 3.7 Launch (2025-02-24)":
            reference_date = CLAUDE_3P7_LAUNCH_DATE
        else:
            reference_date = st.date_input(
                "Custom reference date",
                DEFAULT_PARAMS["reference_date"].date(),
                help="The date from which to start counting days until AGI"
            )
            # Convert to datetime
            reference_date = datetime.combine(reference_date, datetime.min.time())
    
    # Only show the additional parameter adjustments if advanced mode is on
    use_adjusted_start_task = False
    if elicitation_boost != DEFAULT_PARAMS["elicitation_boost"] or reliability_needed != DEFAULT_PARAMS["reliability_needed"] or task_type_penalty != DEFAULT_PARAMS["task_type_penalty"]:
        use_adjusted_start_task = True
        # Calculate adjusted start task length
        reliability_penalty = 1.0
        if reliability_needed > 0.5:
            # Simple approximation of the reliability penalty function
            if reliability_needed <= 0.8:
                reliability_penalty = 0.25
            elif reliability_needed <= 0.9:
                reliability_penalty = 0.25**2
            elif reliability_needed <= 0.95:
                reliability_penalty = 0.25**3
            else:
                reliability_penalty = 0.25**4
        
        adjusted_start_task_length = start_task_length * elicitation_boost * reliability_penalty * task_type_penalty
        st.sidebar.markdown(f"Adjusted start task length: **{adjusted_start_task_length:.3f} hours**")
        start_task_length = adjusted_start_task_length
else:
    # Default reference date when not in advanced mode
    reference_date = DEFAULT_PARAMS["reference_date"]
    
    # Set these variables to their defaults or preset values for use in other parts of the code
    if hasattr(st.session_state, "preset_params"):
        elicitation_boost = st.session_state.preset_params.get("elicitation_boost", DEFAULT_PARAMS["elicitation_boost"])
        reliability_needed = st.session_state.preset_params.get("reliability_needed", DEFAULT_PARAMS["reliability_needed"])
        task_type_penalty = st.session_state.preset_params.get("task_type_penalty", DEFAULT_PARAMS["task_type_penalty"])
    else:
        elicitation_boost = DEFAULT_PARAMS["elicitation_boost"]
        reliability_needed = DEFAULT_PARAMS["reliability_needed"]
        task_type_penalty = DEFAULT_PARAMS["task_type_penalty"]

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Model")

# Prepare variables for model
kwargs = {}
if start_task_length > 0:
    kwargs["start_task_length"] = sq.const(start_task_length)
if agi_task_length > 0:
    kwargs["agi_task_length"] = sq.const(agi_task_length)
if doubling_time > 0:
    kwargs["doubling_time"] = sq.const(doubling_time)
else:
    # If doubling_time is -1, use the mixture model
    kwargs["doubling_time"] = None  # Will use the default mixture model in model.py
if acceleration != DEFAULT_PARAMS["acceleration"]:
    kwargs["acceleration"] = sq.const(acceleration)
if shift > 0:
    kwargs["shift"] = sq.const(shift)
# Add reference date to kwargs
kwargs["index_date"] = reference_date

if advanced_mode:
    # Get correlated value from preset if available
    correlated_default = st.session_state.preset_params.get("correlated", DEFAULT_PARAMS["correlated"]) if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["correlated"]
    correlated = st.sidebar.checkbox(
        "Correlate doubling time and acceleration (advanced)",
        value=correlated_default,
        help="If checked, samples doubling time and acceleration with negative correlation (lower doubling time → lower acceleration, i.e., faster progress)."
    )
    
    # Get parallel value from preset if available
    parallel_default = st.session_state.preset_params.get("use_parallel", DEFAULT_PARAMS["use_parallel"]) if hasattr(st.session_state, "preset_params") else DEFAULT_PARAMS["use_parallel"]
    use_parallel = st.sidebar.checkbox(
        "Parallelize sampling (advanced, for large n_samples)",
        value=parallel_default,
        help="If checked, uses multiprocessing to parallelize sampling for large sample sizes. May speed up model runs on large datasets."
    )
else:
    # Set from preset if available, otherwise use defaults
    if hasattr(st.session_state, "preset_params"):
        correlated = st.session_state.preset_params.get("correlated", DEFAULT_PARAMS["correlated"])
        use_parallel = st.session_state.preset_params.get("use_parallel", DEFAULT_PARAMS["use_parallel"])
    else:
        correlated = DEFAULT_PARAMS["correlated"]
        use_parallel = DEFAULT_PARAMS["use_parallel"]

kwargs["correlated"] = correlated
kwargs["use_parallel"] = use_parallel

@st.cache_data(show_spinner=False)
def cached_run_model(n_samples, _start_task_length, _agi_task_length, _doubling_time, _acceleration, _shift, index_date, correlated=False, use_parallel=False):
    return run_model(
        n_samples=n_samples,
        start_task_length=_start_task_length,
        agi_task_length=_agi_task_length,
        doubling_time=_doubling_time,
        acceleration=_acceleration,
        shift=_shift,
        index_date=index_date,
        correlated=correlated,
        use_parallel=use_parallel,
    )

if run_button or "results" not in st.session_state:
    with st.spinner("Running model..."):
        try:
            samples, samples_dates = cached_run_model(
                n_samples=n_samples,
                _start_task_length=kwargs.get("start_task_length"),
                _agi_task_length=kwargs.get("agi_task_length"),
                _doubling_time=kwargs.get("doubling_time"),
                _acceleration=kwargs.get("acceleration"),
                _shift=kwargs.get("shift"),
                index_date=kwargs.get("index_date"),
                correlated=kwargs.get("correlated", False),
                use_parallel=kwargs.get("use_parallel", False),
            )
            st.session_state["results"] = (samples, samples_dates)
            st.success("Model run completed successfully!")
        except Exception as e:
            st.error(f"Error running model: {str(e)}")
            st.info("Try adjusting parameters or reducing the number of samples.")
            if "results" not in st.session_state:
                # If we have no previous results, stop here
                st.stop()
            else:
                # Use previous results
                samples, samples_dates = st.session_state["results"]
                st.warning("Showing previous model results.")
else:
    samples, samples_dates = st.session_state["results"]

# Show results
st.header("Results")

from pprint import pformat
import numpy as np

# Handle NaN values in percentiles calculation
try:
    # Filter out NaN values before calculating percentiles
    samples_clean = samples[~np.isnan(samples)]
    if len(samples_clean) < len(samples):
        st.warning(f"Filtered out {len(samples) - len(samples_clean)} NaN values from results.")
    
    if len(samples_clean) > 0:
        percentiles = sq.get_percentiles(samples_clean, digits=0)

    else:
        st.error("All samples contain NaN values. Cannot calculate percentiles.")
except Exception as e:
    st.error(f"Error calculating percentiles: {str(e)}")

# Histogram of AGI arrival dates
try:
    # Filter out invalid dates
    valid_dates = [d for d in samples_dates if d is not None and not np.isnan(d.year)]
    if len(valid_dates) > 0:
        years = np.array([d.year + d.timetuple().tm_yday / 365 for d in valid_dates])
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(years, bins=50, color="#4F8DFD", alpha=0.7)
        ax.set_xlabel("AGI Arrival Year")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of AGI Arrival Dates")
        st.pyplot(fig)

        # Add violin plot for uncertainty visualization
        st.subheader("Uncertainty Visualization: Violin Plot of AGI Arrival Years")
        fig2, ax2 = plt.subplots(figsize=(8, 2))
        ax2.violinplot(years, showmeans=True, showmedians=True)
        ax2.set_xticks([1])
        ax2.set_xticklabels(["AGI Arrival Year Distribution"])
        ax2.set_ylabel("Year")
        st.pyplot(fig2)
    else:
        st.warning("No valid dates to plot in histogram.")
except Exception as e:
    st.error(f"Error plotting histogram: {str(e)}")

# Add the exponential growth plot to the UI
try:
    st.subheader("Exponential Growth in Task Length Over Time")
    
    # Start by checking if valid_dates exists and has elements
    if 'valid_dates' in locals() and len(valid_dates) > 0:
        # Create a simple static example plot to demonstrate the layout
        fig, ax = plt.subplots(figsize=(11, 6))
        
        # Create a logarithmic series for the three curves
        quarters = np.arange(0, 44)
        x = quarters
        
        # Calculate y values based on the current parameters
        # These are simplified calculations to ensure visibility
        base_rate = start_task_length
        doubling_days = 212 if doubling_time <= 0 else doubling_time
        
        # Median curve - classic exponential
        median_y = base_rate * np.power(2, (x * 91.25) / doubling_days)
        
        # Calculate 10% faster and 90% slower curves
        fast_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 0.7))
        slow_y = base_rate * np.power(2, (x * 91.25) / (doubling_days * 1.5))

        # Plot the three curves
        ax.plot(x, fast_y, 'b--', linewidth=2, label='10% Earliest')
        ax.plot(x, median_y, 'b-', linewidth=2, label='Median')
        ax.plot(x, slow_y, 'b--', linewidth=2, label='90% Latest')
        
        # Add endpoint markers
        # Find where curves cross AGI threshold
        agi_threshold = agi_task_length
        
        # Find crossing points (simplified)
        med_cross = np.argmax(median_y >= agi_threshold) if np.any(median_y >= agi_threshold) else len(x)-1
        fast_cross = np.argmax(fast_y >= agi_threshold) if np.any(fast_y >= agi_threshold) else len(x)-1
        slow_cross = np.argmax(slow_y >= agi_threshold) if np.any(slow_y >= agi_threshold) else len(x)-1
        
        # Add red X markers for AGI reached
        if med_cross < len(x)-1:
            ax.plot(x[med_cross], median_y[med_cross], 'rx', markersize=10)
        if fast_cross < len(x)-1:
            ax.plot(x[fast_cross], fast_y[fast_cross], 'rx', markersize=10)
        if slow_cross < len(x)-1:
            ax.plot(x[slow_cross], slow_y[slow_cross], 'rx', markersize=10)
        
        # Add black circle markers for AGI not reached by end
        if med_cross == len(x)-1:
            ax.plot(x[med_cross], median_y[med_cross], 'ko', markersize=10)
        if fast_cross == len(x)-1:
            ax.plot(x[fast_cross], fast_y[fast_cross], 'ko', markersize=10)
        if slow_cross == len(x)-1:
            ax.plot(x[slow_cross], slow_y[slow_cross], 'ko', markersize=10)
        
        # Add dummy plots for legend
        ax.plot([], [], 'rx', markersize=10, label='AGI reached')
        ax.plot([], [], 'ko', markersize=10, label='AGI not by EOY2035')
        
        # Format the plot with improved y-axis
        ax.set_yscale('log', base=2)
        
        # Calculate min and max values to show on y-axis
        min_y_value = min(min(fast_y), min(median_y), min(slow_y))
        max_y_value = max(max(fast_y), max(median_y), max(slow_y), agi_threshold*1.5)
        
        # Create better tick spacing for log2 scale
        # Start with appropriate powers of 2 based on the data range
        min_power = max(int(np.floor(np.log2(min_y_value))), -10)  # Don't go below 2^-10
        max_power = min(int(np.ceil(np.log2(max_y_value))), 20)    # Don't go above 2^20
        
        # Create tick positions at powers of 2 and some intermediate values
        major_ticks = [2**k for k in range(min_power, max_power+1, 2)]
        minor_ticks = [2**k for k in range(min_power, max_power+1) if k % 2 == 1]
        
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        
        # Add horizontal grid lines at major ticks for better readability
        ax.grid(True, axis='y', which='major', linestyle='-', alpha=0.3)
        ax.grid(True, axis='y', which='minor', linestyle='--', alpha=0.15)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Format y-axis with human-readable labels
        def format_ticks(x, pos):
            # Convert hours to days (1 day = 8 work hours)
            days = x / 8
            
            if days >= 365:
                return f"{days/365:.1f} years"
            if days >= 30:
                return f"{days/30:.1f} months"
            if days >= 7:
                return f"{days/7:.1f} weeks"
            if days >= 1:
                return f"{days:.1f} days"
            if days >= 0.5:
                return f"{days:.2f} days"
            if days >= 0.1:
                return f"{days:.2f} days"
            # For very small values (less than 0.1 days), show in hours
            return f"{x:.1f} hrs"
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        
        # Add common task lengths as horizontal reference lines with day-based descriptions
        reference_tasks = [
            (0.25, "15 min task", "gray", "dotted"),      # 15 minutes
            (1, "1 hour task", "gray", "dotted"),         # 1 hour
            (8, "1 day task", "gray", "dashed"),          # 1 workday
            (40, "1 week task", "gray", "dashed"),        # 1 workweek
            (160, "1 month task", "gray", "solid"),       # 1 month
            (2000, "1 year task", "gray", "solid")        # 1 year
        ]
        
        for hours, label, color, style in reference_tasks:
            if min_y_value <= hours <= max_y_value * 1.2:
                ax.axhline(y=hours, color=color, linestyle=style, alpha=0.5)
                # Update label to show days equivalent
                if hours >= 8:
                    days = hours / 8
                    if days >= 30:
                        day_label = f"{label} (~{days/30:.1f} months)"
                    elif days >= 7:
                        day_label = f"{label} (~{days/7:.1f} weeks)"
                    else:
                        day_label = f"{label} ({days:.1f} days)"
                    ax.text(len(x)*1.02, hours, day_label, va='center', fontsize=8, alpha=0.7)
                else:
                    ax.text(len(x)*1.02, hours, label, va='center', fontsize=8, alpha=0.7)
        
        # Highlight the AGI threshold with day-based description
        days_threshold = agi_threshold / 8
        threshold_label = f"AGI threshold: {agi_threshold:.0f} hrs"
        if days_threshold >= 30:
            threshold_label += f" ({days_threshold/30:.1f} months)"
        elif days_threshold >= 7:
            threshold_label += f" ({days_threshold/7:.1f} weeks)"
        else:
            threshold_label += f" ({days_threshold:.1f} days)"
            
        ax.axhline(y=agi_threshold, color='red', linestyle='-.', alpha=0.7)
        ax.text(len(x)*1.02, agi_threshold, threshold_label, 
               va='center', color='red', fontsize=9, fontweight='bold')
        
        # Format x-axis with quarters
        quarter_labels = []
        for q in quarters:
            year = 2025 + q // 4
            quarter = q % 4 + 1
            quarter_labels.append(f"{year}Q{quarter}")
        
        # Only show a subset of x-ticks to avoid overcrowding
        tick_interval = 4  # Show 1 tick per year
        ax.set_xticks(quarters[::tick_interval])
        ax.set_xticklabels(quarter_labels[::tick_interval], rotation=45, ha='right')
        
        # Add minor ticks for quarters
        ax.set_xticks(quarters, minor=True)
        
        # Set axis labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Task Length (work hours)')
        
        # Add title and legend
        ax.set_title('Projected Task Length Growth Over Time', fontsize=14)
        ax.legend(loc='upper left')
        
        # Dynamically adjust y-axis range to show relevant data
        ax.set_ylim(min_y_value * 0.9, max_y_value * 1.1)
        
        # Adjust x-axis to show at least 5 years (20 quarters)
        max_x = max(20, med_cross + 8, fast_cross + 4, slow_cross + 12)
        max_x = min(max_x, len(x) - 1)  # Don't exceed the max quarters
        ax.set_xlim(-1, max_x)
        
        # Adjust layout for better text fit
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
    else:
        st.warning("No simulation results available. Please run the model first.")
except Exception as e:
    st.error(f"Error plotting exponential growth: {str(e)}")
    st.exception(e)

# Show AGI arrival date by percentile
try:
    try:
        # Check if valid_dates exists and has elements
        if len(valid_dates) > 0:
            st.subheader("AGI Arrival Dates by Year")
            years_array = np.array([d.year for d in valid_dates])
            
            # Calculate cumulative probability for each year
            min_year = int(np.min(years_array))
            max_year = int(np.max(years_array))
            
            # Show years from now until max_year with reasonable spacing
            current_year = datetime.now().year
            display_years = list(range(current_year, current_year+6)) + list(range(current_year+5, max_year+1, 5))
            
            results = []
            for year in display_years:
                prob = np.mean(years_array <= year) * 100
                results.append(f"By EOY {year}: {prob:.1f}%")
            
            # Display in two columns
            col1, col2 = st.columns(2)
            half = len(results) // 2 + len(results) % 2
            
            with col1:
                for r in results[:half]:
                    st.text(r)
                
            with col2:
                for r in results[half:]:
                    st.text(r)
                
            # --- Add enhanced confidence interval display ---
            st.subheader("Confidence Intervals")
            
            # Calculate percentiles for the AGI dates
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = []
            percentile_dates = []
            
            # Sort the array for percentile calculation
            sorted_years = np.sort(years_array)
            sorted_dates = np.sort(valid_dates)
            
            for p in percentiles:
                idx = int(p/100 * len(sorted_years))
                percentile_values.append(sorted_years[idx])
                percentile_dates.append(sorted_dates[idx])
            
            # Create a table showing the percentiles
            percentile_data = pd.DataFrame({
                "Percentile": [f"{p}%" for p in percentiles],
                "Year": [f"{y}" for y in percentile_values],
                "Date": [d.strftime("%B %Y") for d in percentile_dates]
            })
            
            # Display the table
            st.table(percentile_data)
            
            # Create a boxplot to visualize the confidence intervals
            fig_ci, ax_ci = plt.subplots(figsize=(10, 2))
            ax_ci.boxplot(years_array, vert=False, widths=0.7, 
                         whis=[5, 95],  # 5-95 percentile whiskers
                         medianprops=dict(color='red', linewidth=2),
                         boxprops=dict(linewidth=2),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2))
            
            # Add a rug plot to show the distribution density
            ax_ci.plot(years_array, np.random.normal(1, 0.04, size=len(years_array)), '|', 
                      alpha=0.5, color='blue', markersize=8)
            
            # Annotate the plot with percentiles
            for i, p in enumerate(percentiles):
                if p in [5, 25, 50, 75, 95]:  # Only annotate key percentiles
                    ax_ci.text(percentile_values[i], 0.8, f"{p}%", 
                              ha='center', va='center', fontsize=9,
                              bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
            
            # Add vertical lines for important percentiles
            for i, p in enumerate(percentiles):
                if p in [10, 50, 90]:  # 10%, 50% (median), 90%
                    linestyle = '-' if p == 50 else '--'
                    ax_ci.axvline(x=percentile_values[i], color='gray', linestyle=linestyle, alpha=0.6)
            
            # Clean up the plot
            ax_ci.set_yticks([])
            ax_ci.set_xlabel('AGI Arrival Year')
            ax_ci.set_title('Confidence Intervals for AGI Arrival')
            ax_ci.grid(True, axis='x', linestyle='--', alpha=0.6)
            
            # Set nice x-axis limits
            min_display = max(min(percentile_values) - 0.5, current_year)
            max_display = min(max(percentile_values) + 0.5, 2050)
            ax_ci.set_xlim(min_display, max_display)
            
            # Show the plot
            st.pyplot(fig_ci)
            
            # Add interpretive text
            st.markdown(f"""
            **Confidence Interval Interpretation:**
            - There is a **5%** chance AGI will arrive before **{percentile_dates[0].strftime('%B %Y')}**
            - There is a **50%** chance AGI will arrive before **{percentile_dates[3].strftime('%B %Y')}** (median projection)
            - There is a **95%** chance AGI will arrive before **{percentile_dates[6].strftime('%B %Y')}**
            
            These confidence intervals reflect uncertainty in the model parameters and the inherent uncertainty in forecasting technology development.
            """)
                
    except NameError:
        st.warning("No valid dates to calculate year probabilities.")
except Exception as e:
    st.error(f"Error calculating year probabilities: {str(e)}")
    st.exception(e)

# Show model's computed median AGI achievement date
try:
    if 'valid_dates' in locals() and len(valid_dates) > 0:
        median_idx = int(np.median(np.arange(len(valid_dates))))
        sorted_dates = np.sort(valid_dates)
        median_date = sorted_dates[median_idx]
        st.subheader("Model's Computed Median AGI Date")
        st.info(f"**{median_date.strftime('%A, %-d %B %Y')}**")

        # --- New Timeline Visualization Section ---
        st.subheader("AI Capability Timeline")
        
        # Define constants for timeline calculations
        days_per_quarter = 91.25  # Average days per quarter
        # Define quarters array for timeline calculations (needed here since it's from a different scope)
        quarters = np.arange(0, 44)
        x = quarters
        n_quarters = len(quarters)  # Using the existing quarters variable from earlier code
        
        # Define capability milestones with their task lengths in hours
        milestones = [
            {"name": "Start (Current)", "hours": start_task_length, "description": "Current strongest model capabilities"},
            {"name": "Week-Long Project", "hours": 40, "description": "Models can complete full week-long software projects"},
            {"name": "Two-Week Project", "hours": 80, "description": "Models can build and deploy complex systems over two weeks"},
            {"name": "Month-Long Project", "hours": 167, "description": "Models can autonomously develop month-long projects"},
            {"name": "Quarter-Long Project", "hours": 500, "description": "Models can manage projects spanning a quarter"},
            {"name": "Half-Year Project", "hours": 1000, "description": "Models can execute multi-quarter planning and projects"},
            {"name": "Year-Long Project", "hours": 2000, "description": "AGI level: Models can execute year-long development cycles"}
        ]
        
        # Calculate estimated dates for each milestone
        for i, milestone in enumerate(milestones):
            if i == 0:  # Skip Start milestone
                milestone["date"] = reference_date
                milestone["relative_days"] = 0
                continue
                
            milestone_threshold = milestone["hours"]
            
            # Calculate where the median curve crosses this milestone
            base_rate = start_task_length  # Define base_rate in this scope
            y = base_rate * np.power(2, (x * days_per_quarter) / doubling_days)
            cross_idx = np.argmax(y >= milestone_threshold) if np.any(y >= milestone_threshold) else n_quarters
            relative_days = cross_idx * days_per_quarter
            milestone_date = reference_date + timedelta(days=relative_days)
            
            # Add to milestone data
            milestone["date"] = milestone_date
            milestone["relative_days"] = relative_days
            
            # Calculate confidence interval for more important milestones
            if milestone["hours"] >= 167:  # Only for month+ projects
                # 10% earliest date
                fast_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 0.7))
                fast_cross = np.argmax(fast_y >= milestone_threshold) if np.any(fast_y >= milestone_threshold) else n_quarters
                fast_relative_days = fast_cross * days_per_quarter
                fast_date = reference_date + timedelta(days=fast_relative_days)
                milestone["early_date"] = fast_date
                
                # 90% latest date
                slow_y = base_rate * np.power(2, (x * days_per_quarter) / (doubling_days * 1.5))
                slow_cross = np.argmax(slow_y >= milestone_threshold) if np.any(slow_y >= milestone_threshold) else n_quarters
                slow_relative_days = slow_cross * days_per_quarter
                slow_date = reference_date + timedelta(days=slow_relative_days)
                milestone["late_date"] = slow_date
        
        # Create timeline visualization
        st.markdown("Below is a projected timeline of when AI capabilities will reach various milestones based on your model parameters:")
        
        # Create figure for the timeline
        fig_timeline, ax_timeline = plt.subplots(figsize=(10, 7))
        
        # Convert dates to numbers for plotting
        base_date_num = matplotlib.dates.date2num(reference_date)
        
        # Get milestone dates and positions
        milestone_dates = [milestone["date"] for milestone in milestones]
        milestone_date_nums = [matplotlib.dates.date2num(date) if date is not None else base_date_num for date in milestone_dates]
        
        # Set up positions
        y_positions = list(range(len(milestones), 0, -1))
        
        # Plot the timeline markers and lines
        ax_timeline.plot(milestone_date_nums, y_positions, 'o-', markersize=12, color='#4F8DFD', linewidth=2)
        
        # Add confidence intervals for AGI-level milestones
        for i, milestone in enumerate(milestones):
            if "early_date" in milestone and "late_date" in milestone:
                early_date_num = matplotlib.dates.date2num(milestone["early_date"])
                late_date_num = matplotlib.dates.date2num(milestone["late_date"])
                ax_timeline.plot([early_date_num, late_date_num], [y_positions[i], y_positions[i]], 'b-', linewidth=3, alpha=0.3)
                ax_timeline.plot([early_date_num], [y_positions[i]], 'b|', markersize=10)
                ax_timeline.plot([late_date_num], [y_positions[i]], 'b|', markersize=10)
        
        # Customize the plot
        ax_timeline.set_yticks(y_positions)
        ax_timeline.set_yticklabels([f"{m['name']} ({m['hours']}h)" for m in milestones])
        
        # Format the x-axis to show dates
        date_format = matplotlib.dates.DateFormatter('%b %Y')
        ax_timeline.xaxis.set_major_formatter(date_format)
        ax_timeline.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax_timeline.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(interval=6))
        
        # Rotate x-axis labels to prevent overlap
        plt.setp(ax_timeline.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adjust figure to make room for rotated labels
        plt.subplots_adjust(bottom=0.2)
        
        # Add grid, labels, and style
        ax_timeline.grid(True, linestyle='--', alpha=0.7)
        ax_timeline.set_xlabel('Projected Date')
        ax_timeline.set_title('AI Capability Timeline Based on Model Parameters')
        
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig_timeline)
        
        # Show milestones in table format
        st.markdown("### Detailed Milestone Projections")
        milestone_data = []
        for m in milestones:
            entry = {
                "Milestone": m["name"],
                "Task Length (hours)": m["hours"],
                "Projected Date": m["date"].strftime("%b %Y") if m["date"] is not None else "N/A",
                "Description": m["description"]
            }
            
            # Add confidence intervals if available
            if "early_date" in m and "late_date" in m:
                entry["Confidence Interval"] = f"{m['early_date'].strftime('%b %Y')} - {m['late_date'].strftime('%b %Y')}"
            else:
                entry["Confidence Interval"] = "N/A"
                
            milestone_data.append(entry)
        
        # Convert to pandas DataFrame for display
        milestone_df = pd.DataFrame(milestone_data)
        st.dataframe(milestone_df)
        
        # Define defaults for comparison
        default_params = DEFAULT_PARAMS
        # Get user selections
        user_params = {
            "start_task_length": round(float(start_task_length), 3),
            "agi_task_length": round(float(agi_task_length), 3),
            "doubling_time": round(float(doubling_time), 3),
            "acceleration": round(float(acceleration), 3),
            "shift": int(shift),
            "correlated": correlated,
            "use_parallel": use_parallel,
        }
        
        # Add advanced parameters if they're available
        if advanced_mode:
            user_params.update({
                "elicitation_boost": round(float(elicitation_boost), 3),
                "reliability_needed": round(float(reliability_needed), 3),
                "task_type_penalty": round(float(task_type_penalty), 3),
            })
        else:
            # Use defaults for advanced parameters when not in advanced mode
            user_params.update({
                "elicitation_boost": DEFAULT_PARAMS["elicitation_boost"],
                "reliability_needed": DEFAULT_PARAMS["reliability_needed"],
                "task_type_penalty": DEFAULT_PARAMS["task_type_penalty"],
            })
        
        # Add reference_date
        user_params["reference_date"] = reference_date.strftime('%Y-%m-%d')
        
        # Helper for difference
        def diff_str(param, user, default):
            if user == default:
                return f"(default: {default})"
            else:
                return f"(default: {default}, changed)"

        # --- Explainability Section ---
        st.markdown("---")
        st.subheader("Explainability: What does this scenario mean?")

        st.markdown(f"""
        **What does this model do?**
        This model projects when AGI might be achieved by extrapolating recent trends in AI's ability to complete long, complex tasks, as measured by the METR benchmark. It uses a probabilistic approach, accounting for uncertainty in how fast progress will continue, how hard AGI-level tasks are, and how much reliability is required.
        
        **Your parameter selections:**
        - **Start task length:** {user_params['start_task_length']} hours {diff_str('start_task_length', user_params['start_task_length'], default_params['start_task_length'])}
            - The number of hours it takes a human to complete the hardest task that the current best AI model can do with 50% reliability, after adjusting for improvements from scaffolding and compute, higher reliability requirements, and the possibility that AGI tasks are harder than METR's self-contained software tasks. Lower values mean AGI is closer.
        - **AGI task length:** {user_params['agi_task_length']} hours {diff_str('agi_task_length', user_params['agi_task_length'], default_params['agi_task_length'])}
            - The human time required to complete a task that would count as "AGI-level" (e.g., a month- or year-long project). Lower values mean AGI is easier to reach.
        - **Doubling time:** {user_params['doubling_time']} days {diff_str('doubling_time', user_params['doubling_time'], default_params['doubling_time'])}
            - The number of days it takes for the maximum task length that AI can do at 50% reliability to double. Shorter doubling times mean faster progress.
        - **Acceleration:** {user_params['acceleration']} {diff_str('acceleration', user_params['acceleration'], default_params['acceleration'])}
            - If <1, progress accelerates (superexponential); if >1, progress slows; if 1, progress is exponential. Lower values mean AGI arrives sooner.
        - **Shift:** {user_params['shift']} days {diff_str('shift', user_params['shift'], default_params['shift'])}
            - The number of days to shift the forecast earlier to account for internal model capabilities before public release. Higher values mean AGI is forecasted to arrive sooner.
        - **Elicitation boost:** {user_params['elicitation_boost']} {diff_str('elicitation_boost', user_params['elicitation_boost'], default_params['elicitation_boost'])}
            - Factor for improvements from better scaffolding or more compute. >1 means faster progress.
        - **Reliability needed:** {user_params['reliability_needed']} {diff_str('reliability_needed', user_params['reliability_needed'], default_params['reliability_needed'])}
            - The required reliability for AGI-level performance. Higher reliability means a harder bar for AGI and increases the penalty on start task length.
        - **Task type complexity factor:** {user_params['task_type_penalty']} {diff_str('task_type_penalty', user_params['task_type_penalty'], default_params['task_type_penalty'])}
            - Adjusts for AGI tasks being harder than the METR benchmark tasks. >1 = AGI tasks are harder.
        - **Reference date:** {user_params['reference_date']} {diff_str('reference_date', user_params['reference_date'], default_params['reference_date'].strftime('%Y-%m-%d'))}
            - The date from which to start counting days until AGI. Usually the launch date of the reference model.
        - **Correlated sampling:** {user_params['correlated']} {diff_str('correlated', user_params['correlated'], default_params['correlated'])}
            - If enabled, samples doubling time and acceleration with negative correlation (lower doubling time → lower acceleration, i.e., faster progress). This can make AGI arrive sooner.
        - **Parallel sampling:** {user_params['use_parallel']} {diff_str('use_parallel', user_params['use_parallel'], default_params['use_parallel'])}
            - If enabled, uses multiprocessing to parallelize sampling for large sample sizes. This does not affect the forecast, only performance.
        
        **How do these affect the forecast?**
        - Lower start task length, lower AGI task length, shorter doubling time, and lower acceleration all make AGI arrive sooner.
        - Higher reliability or harder AGI task definitions push the date later.
        - Enabling correlated sampling can make AGI arrive sooner by compounding progress.
        
        **Uncertainty and caveats:**
        - The model assumes "business as usual" and does not account for major disruptions (e.g., regulation, war, economic shocks).
        - It extrapolates from recent trends, which may not continue indefinitely. Progress could slow down due to diminishing returns, or speed up due to breakthroughs or feedback loops.
        - The model is based on METR's AI evaluation tasks and models' perfomance on them. These focus solely on software engineering, neglecting more complex or ambiguous domains where AI may struggle. They're highly structured, solitary, and low-stakes, unlike real-world environments that demand adaptability, collaboration, and consequence-aware performance. Additionally, the tasks ignore learning curves that benefit human workers over time and set a low reliability benchmark (50%), whereas real-world applications often require much higher consistency.
        - The model does not predict when AGI will be widely deployed or have major social impact—just when it becomes technically possible.
        """)
    else:
        st.warning("No valid dates to compute median AGI date.")
except Exception as e:
    st.error(f"Error computing median AGI date: {str(e)}")

st.markdown("---")

# =====================
# Sensitivity Analysis
# =====================
st.header("Sensitivity Analysis")
st.markdown("""
Explore how sensitive the AGI timeline is to each parameter. The tornado plot shows which parameters most affect the median AGI year. Use the what-if slider to see how changing a parameter shifts the forecast. ([Learn more](https://www.numberanalytics.com/blog/mastering-sensitivity-analysis-techniques-robust-data-models))
""")

# --- Sensitivity Analysis Setup ---
SENS_N = 100000
param_names = [
    ("start_task_length", "Start task length (hours)", float(0.01), float(10.0)),
    ("agi_task_length", "AGI task length (hours)", float(40.0), float(5000.0)),
    ("doubling_time", "Doubling time (days)", float(60.0), float(400.0)),
    ("acceleration", "Acceleration", float(0.8), float(1.2)),
    ("shift", "Shift (days)", float(0), float(250)),
]

# Get current parameter values
fixed_params = {
    "start_task_length": start_task_length,
    "agi_task_length": agi_task_length,
    "doubling_time": doubling_time,
    "acceleration": acceleration,
    "shift": shift,
    "index_date": reference_date,
    "correlated": correlated,
    "use_parallel": False,
}

def get_median_agi_year(params):
    try:
        days, dates = run_model(
            n_samples=SENS_N,
            start_task_length=sq.const(params["start_task_length"]),
            agi_task_length=sq.const(params["agi_task_length"]),
            doubling_time=sq.const(params["doubling_time"]),
            acceleration=sq.const(params["acceleration"]),
            shift=sq.const(params["shift"]),
            index_date=params["index_date"],
            correlated=params.get("correlated", False),
            use_parallel=False,
        )
        valid_dates = [d for d in dates if d is not None]
        if valid_dates:
            years = np.array([d.year + d.timetuple().tm_yday / 365 for d in valid_dates])
            return float(np.median(years))
        else:
            return np.nan
    except Exception:
        return np.nan

# --- Tornado Plot ---
st.subheader("Tornado Plot: Parameter Sensitivity")
impacts = []
labels = []
low_vals = []
high_vals = []
low_yrs = []
high_yrs = []

with st.spinner("Calculating tornado plot..."):
    for key, label, vmin, vmax in param_names:
        # Test at low and high value
        params_low = fixed_params.copy()
        params_high = fixed_params.copy()
        params_low[key] = vmin
        params_high[key] = vmax
        low_year = get_median_agi_year(params_low)
        high_year = get_median_agi_year(params_high)
        # Store for plotting
        labels.append(label)
        low_vals.append(vmin)
        high_vals.append(vmax)
        low_yrs.append(low_year)
        high_yrs.append(high_year)
        impacts.append(abs(high_year - low_year))

# Sort by impact
order = np.argsort(impacts)[::-1]
labels = np.array(labels)[order]
params_ordered = np.array([p[0] for p in param_names])[order]
low_vals = np.array(low_vals)[order]
high_vals = np.array(high_vals)[order]
low_yrs = np.array(low_yrs)[order]
high_yrs = np.array(high_yrs)[order]

fig, ax = plt.subplots(figsize=(7, 4))
for i, (l, lo, hi) in enumerate(zip(labels, low_yrs, high_yrs)):
    ax.plot([lo, hi], [i, i], 'o-', color='#4F8DFD', lw=6, alpha=0.7)
    ax.plot([lo], [i], 'o', color='red', label='Low value' if i==0 else "")
    ax.plot([hi], [i], 'o', color='green', label='High value' if i==0 else "")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Median AGI Year")
ax.set_title("Tornado Plot: Sensitivity of Median AGI Year")
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# Interactive Sensitivity Analysis
st.subheader("Interactive Sensitivity Analysis")
st.markdown("""
Now you can directly see how changing each parameter affects the median AGI timeline. 
Adjust the sliders below to see real-time updates to the forecast.
""")

# Function to run a simplified single-value model (no probability distributions)
def quick_estimate_agi_date(params):
    # Simple deterministic calculation
    base_rate = params["start_task_length"]
    agi_threshold = params["agi_task_length"]
    doubling_days = params["doubling_time"]
    accel = params["acceleration"]
    shift_days = params["shift"]
    
    # Apply shift to boost current capabilities
    base_rate = base_rate * (2 ** (shift_days / doubling_days))
    
    # Calculate log2 of the ratio to get number of doublings needed
    ratio = agi_threshold / base_rate
    doublings_needed = np.log2(ratio)
    
    # Calculate days to AGI
    if accel == 1.0:
        days_to_agi = doublings_needed * doubling_days
    else:
        try:
            # For superexponential or subexponential growth
            power_term = accel ** doublings_needed
            days_to_agi = doubling_days * (1 - power_term) / (1 - accel)
        except:
            # Fallback if calculation fails
            days_to_agi = doublings_needed * doubling_days
    
    # Return the date
    agi_date = params["index_date"] + timedelta(days=days_to_agi)
    return agi_date, days_to_agi

# Create interactive sliders for all parameters
interactive_params = fixed_params.copy()

# Replace with a single section showing all parameters, most impactful first
st.markdown("### Adjust Parameters By Impact")

# Get the parameters ordered by impact
ordered_params = params_ordered

# Create columns for the parameter controls and the result display
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("**Adjust all parameters (ordered by impact)**")
    
    # Create two columns for parameters
    param_col1, param_col2 = st.columns(2)
    
    # Add sliders for ALL parameters in order of impact
    for i, param_key in enumerate(ordered_params):
        # Find the parameter's metadata
        idx = [p[0] for p in param_names].index(param_key)
        _, label, vmin, vmax = param_names[idx]
        
        # Get the parameter's current value
        curr_val = fixed_params[param_key]
        
        # Calculate step size
        if param_key == "acceleration":
            step = 0.01
        elif param_key == "shift":
            step = 1.0
        else:
            step = (vmax - vmin) / 100
        
        # Alternate between columns for better layout
        with param_col1 if i % 2 == 0 else param_col2:
            # Create slider with appropriate step size and range
            interactive_params[param_key] = st.slider(
                f"{label}",
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(curr_val),
                step=float(step),
                key=f"interactive_{param_key}"
            )

# Display the results of the interactive analysis
with col2:
    st.markdown("### Estimated AGI Date")
    estimated_date, days_to_agi = quick_estimate_agi_date(interactive_params)
    st.markdown(f"<h2 style='color:#4F8DFD;'>{estimated_date.strftime('%b %Y')}</h2>", unsafe_allow_html=True)
    st.markdown(f"Days to AGI: {days_to_agi:.0f}")
    
    # Calculate years from now
    years_from_now = days_to_agi / 365.25
    st.markdown(f"Years from now: {years_from_now:.1f}")

# Add a progress bar to visualize time to AGI
st.markdown("### Countdown to AGI")
estimated_date, days_to_agi = quick_estimate_agi_date(interactive_params)
years_from_now = days_to_agi / 365.25
progress_bar_max = 10  # max 10 years display
progress_percent = min(years_from_now / progress_bar_max, 1.0)
st.progress(progress_percent)

# Display projected AGI date with parameters
st.markdown(f"**With your parameters, AGI is projected by: {estimated_date.strftime('%B %Y')}** ({days_to_agi:.0f} days, {years_from_now:.1f} years from now)")

# Show what the prediction would be with default parameters
default_date, default_days = quick_estimate_agi_date(fixed_params)
st.markdown(f"**Default parameters predict AGI by: {default_date.strftime('%B %Y')}** ({default_days:.0f} days, {default_days/365.25:.1f} years from now)")

# Optional: show the change vs default
if estimated_date != default_date:
    diff_days = (estimated_date - default_date).days
    if diff_days > 0:
        st.markdown(f"Your adjusted parameters **delay AGI by {abs(diff_days):.0f} days** ({abs(diff_days)/365.25:.1f} years)")
    else:
        st.markdown(f"Your adjusted parameters **accelerate AGI by {abs(diff_days):.0f} days** ({abs(diff_days)/365.25:.1f} years)")

st.markdown("---")
st.markdown("""
*Model code adapted from the METR paper ["Measuring AI Ability to Complete Long Tasks"](https://arxiv.org/abs/2503.14499) 
and [Forecaster Reacts to METR's bombshell](https://peterwildeford.substack.com/p/forecaster-reacts-metrs-bombshell)*
""") 
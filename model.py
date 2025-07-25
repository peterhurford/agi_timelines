import numpy as np
import squigglepy as sq
from datetime import datetime, timedelta
import multiprocessing as mp

# --- Constants ---
O3_LAUNCH_DATE = datetime(2025, 4, 16)
CLAUDE_3P7_LAUNCH_DATE = datetime(2025, 2, 24)

# --- Model Functions ---
def calculate_doubling_time(start_task_length, agi_task_length, doubling_time, acceleration=1):
    # If acceleration is a constant 1, use the simple formula
    # Otherwise, use the general formula
    # This works for both scalars and squigglepy distributions

    # Try to detect if acceleration is exactly 1 (float or sq.Constant)
    is_one = False
    try:
        # If it's a squigglepy constant
        if hasattr(acceleration, 'value'):
            is_one = float(acceleration.value) == 1.0
        else:
            is_one = float(acceleration) == 1.0
    except Exception:
        pass

    ratio = agi_task_length / start_task_length
    doublings_needed = sq.dist_log(ratio) / np.log(2)

    if is_one:
        return doublings_needed * doubling_time
    else:
        # Use ** for exponentiation with distributions
        power_term = acceleration ** doublings_needed
        return doubling_time * (1 - power_term) / (1 - acceleration)

def samples_to_date(samples, index_date=O3_LAUNCH_DATE):
    # Convert samples to numpy array if it's not already
    samples_array = np.asarray(samples)
    
    # Replace any NaN values with a reasonable default (5 years)
    samples_array = np.nan_to_num(samples_array, nan=365*5)
    
    # Clip extremely large values to prevent date overflow
    # (100 years should be enough for most forecasts)
    max_days = 365 * 100  # 100 years
    samples_array = np.clip(samples_array, 0, max_days)
    
    # Round up to nearest integer day and convert to int safely
    days = np.ceil(samples_array).astype(np.int64)
    
    # Vectorized date conversion
    date_converter = np.vectorize(lambda x: index_date + timedelta(days=int(x)))
    return date_converter(days)

# --- Default Distributions (can be overridden) ---
def get_start_task_length(n_samples=100_000):
    current_best = 1.75  # o3 task length at 50% reliability?
    elicitation_boost = sq.mixture([[0.3, 1], [0.4, 1.2], [0.3, 1.5]])
    reliability_needed = sq.mixture([[0.3, 0.5], [0.4, 0.8], [0.2, 0.9], [0.1, 0.99]])
    def reliability_count_to_penalty(r):
        r = np.asarray(r)
        reliability_levels = np.array([0.50, 0.80, 0.90, 0.95, 0.99])
        penalty = np.array([1.0, 0.25, 0.25**2, 0.25**3, 0.25**4])
        idx = np.abs(r[..., None] - reliability_levels).argmin(axis=-1)
        return penalty[idx]
    task_type_penalty = sq.mixture([[0.2, 1], [0.4, 1 / sq.lognorm(5, 20)], [0.4, 1 / sq.lognorm(10, 1000)]])
    start_task_length = current_best * elicitation_boost

    # Sample reliability and apply penalty
    reliability_samples = sq.sample(reliability_needed, n=n_samples)
    penalty_samples = reliability_count_to_penalty(reliability_samples)
    penalty_dist = sq.Empirical(penalty_samples)
    start_task_length = start_task_length * penalty_dist

    start_task_length *= task_type_penalty
    start_task_length = sq.dist_max(30/60/60, start_task_length)
    return start_task_length

def get_agi_task_length():
    return sq.lognorm(80, 2000, credibility=80, lclip=40)

def get_doubling_time():
    return sq.mixture([[0.4, 212], [0.2, 118], [0.1, 320], [0.3, sq.lognorm(lognorm_mean=126, lognorm_sd=40)]])

def get_acceleration():
    return sq.mixture([[0.6, 1], [0.4, 1 - sq.lognorm(0.005, 0.1, credibility=80)]])

def get_shift():
    return sq.norm(30, 30*5, credibility=80, lclip=0)

def adapted_metr_model(start_task_length, agi_task_length, doubling_time, acceleration, shift):
    # Calculate the scaling factor more carefully
    # Use standard ** operator for exponentiation with distributions
    start_task_length = sq.const(start_task_length) if isinstance(start_task_length, (int, float)) else start_task_length
    agi_task_length = sq.const(agi_task_length) if isinstance(agi_task_length, (int, float)) else agi_task_length
    doubling_time = sq.const(doubling_time) if isinstance(doubling_time, (int, float)) else doubling_time
    acceleration = sq.const(acceleration) if isinstance(acceleration, (int, float)) else acceleration
    shift = sq.const(shift) if isinstance(shift, (int, float)) else shift
    
    # Use ** for exponentiation
    scaling_factor = 2 ** (shift / doubling_time)
    
    start_task_length_adjusted = start_task_length * scaling_factor
    days = calculate_doubling_time(start_task_length_adjusted, agi_task_length, doubling_time, acceleration)
    measurement_error_variance = sq.invlognorm(0.8, 1.5)
    return days * measurement_error_variance

def _sample_helper(f, n):
    return sq.sample(f(n), n=n) if callable(f) else sq.sample(f, n=n)

def run_model(
    n_samples=100_000,
    start_task_length=None,
    agi_task_length=None,
    doubling_time=None,
    acceleration=None,
    shift=None,
    index_date=O3_LAUNCH_DATE,
    correlated=False,
    use_parallel=False,
):
    """
    Run the AGI timeline model.
    If correlated=True, samples doubling_time and acceleration with negative correlation (lower doubling_time -> lower acceleration).
    If use_parallel=True and n_samples > 20,000, parallelize the sampling step for speed.
    """
    n_samples = min(max(1000, n_samples), 200_000)

    try:
        if use_parallel and n_samples > 20000:
            with mp.Pool(4) as pool:
                results = pool.starmap(
                    _sample_helper,
                    [
                        (get_start_task_length if start_task_length is None else start_task_length, n_samples),
                        (get_agi_task_length if agi_task_length is None else agi_task_length, n_samples),
                        (get_doubling_time if doubling_time is None else doubling_time, n_samples),
                        (get_acceleration if acceleration is None else acceleration, n_samples),
                        (get_shift if shift is None else shift, n_samples),
                    ]
                )
            start_task_length, agi_task_length, doubling_time, acceleration, shift = results
            if correlated:
                u = np.random.uniform(0, 1, n_samples)
                doubling_time = 60 + 340 * u
                acceleration = 1 - 0.2 * (1 - u)
        else:
            start_task_length = sq.sample(get_start_task_length(n_samples) if start_task_length is None else start_task_length, n=n_samples)
            agi_task_length = sq.sample(get_agi_task_length() if agi_task_length is None else agi_task_length, n=n_samples)
            if correlated:
                u = np.random.uniform(0, 1, n_samples)
                doubling_time = 60 + 340 * u
                acceleration = 1 - 0.2 * (1 - u)
            else:
                doubling_time = sq.sample(get_doubling_time() if doubling_time is None else doubling_time, n=n_samples)
                acceleration = sq.sample(get_acceleration() if acceleration is None else acceleration, n=n_samples)
            shift = sq.sample(get_shift() if shift is None else shift, n=n_samples)

        # Convert to numpy arrays
        start_task_length = np.asarray(start_task_length)
        agi_task_length = np.asarray(agi_task_length)
        doubling_time = np.asarray(doubling_time)
        acceleration = np.asarray(acceleration)
        shift = np.asarray(shift)

        # Safety checks to avoid NaNs
        # Ensure all values are positive where needed
        start_task_length = np.maximum(start_task_length, 1e-10)  # Avoid log(0)
        agi_task_length = np.maximum(agi_task_length, 1e-9)  # Avoid log(0)
        doubling_time = np.maximum(doubling_time, 0.1)  # Avoid division by 0
        
        # Handle special cases in acceleration
        # Acceleration exactly 1.0 needs special handling
        accel_special = np.isclose(acceleration, 1.0, rtol=1e-10, atol=1e-14)
        # Clip acceleration to valid range (avoid NaNs from powers)
        acceleration = np.clip(acceleration, 0.5, 1.5)

        # Now do all calculations in numpy
        scaling_factor = 2 ** (shift / doubling_time)
        start_task_length_adjusted = start_task_length * scaling_factor

        # Calculate doublings needed - avoid log(negative)
        ratio = np.maximum(agi_task_length / start_task_length_adjusted, 1e-10)
        doublings_needed = np.log(ratio) / np.log(2)

        # Calculate days to AGI
        days = np.zeros_like(doublings_needed)
        
        # Handle acceleration = 1 case 
        days[accel_special] = doublings_needed[accel_special] * doubling_time[accel_special]
        
        # Handle acceleration != 1 case safely
        non_special = ~accel_special
        if np.any(non_special):
            # Safe calculation avoiding division by zero and invalid powers
            accel_non_special = acceleration[non_special]
            doublings_non_special = doublings_needed[non_special]
            dt_non_special = doubling_time[non_special]
            
            # Calculate power term safely (avoid negative bases with fractional exponents)
            # Use exp(log(a) * b) instead of a ** b for negative a
            power_term = np.zeros_like(doublings_non_special)
            
            # Valid powers (acceleration > 0, real doublings)
            valid_power = (accel_non_special > 0) & np.isfinite(doublings_non_special)
            power_term[valid_power] = accel_non_special[valid_power] ** doublings_non_special[valid_power]
            
            # Safe division
            denominator = 1 - accel_non_special
            # Avoid division by zero
            safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
            
            days[non_special] = dt_non_special * (1 - power_term) / safe_denom

        # Add measurement error
        measurement_error_variance = sq.sample(sq.invlognorm(0.8, 1.5), n=n_samples)
        days = days * measurement_error_variance

        # Clean up any remaining NaNs or infinities
        days = np.nan_to_num(days, nan=365*5, posinf=365*50, neginf=0)
        
        # Filter out extreme values
        days = np.clip(days, 0, 365*100)

        # Convert to dates
        samples_dates = samples_to_date(days, index_date=index_date)
        return days, samples_dates
        
    except Exception as e:
        # If sampling fails, provide a helpful error message
        error_msg = f"Error in model calculations: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e 

DEFAULT_PARAMS = {
    "start_task_length": 1.75,
    "agi_task_length": 167.0,
    "doubling_time": 212.0,
    "acceleration": 1.0,
    "shift": 90,
    "correlated": False,
    "use_parallel": False,
    "elicitation_boost": 1.0,
    "reliability_needed": 0.5,
    "task_type_penalty": 1.0,
    "reference_date": O3_LAUNCH_DATE,
} 

# Parameter presets with descriptions
PARAMETER_PRESETS = {
    "Default": {
        "params": DEFAULT_PARAMS,
        "description": "Balanced parameters based on current AI trends from METR study. Uses 1.75 hours start task length (o3's level), 167 hours AGI task length (month of work), 212 days doubling time, and standard growth assumptions."
    },
    "Conservative": {
        "params": {
            "start_task_length": 0.8,  # Lower capability assessment
            "agi_task_length": 2000,   # Full work year
            "doubling_time": 320,      # Pessimistic trend
            "acceleration": 1.1,       # Progress slows down
            "shift": 30,               # Conservative shift
            "elicitation_boost": 0.8,  # Less scaffolding improvement
            "reliability_needed": 0.8, # Higher reliability needed
            "task_type_penalty": 4.0,  # AGI tasks much harder than METR tasks
            "reference_date": O3_LAUNCH_DATE,
            "correlated": False,
            "use_parallel": True,
        },
        "description": "Pessimistic timeline with more challenging AGI requirements. Uses lower current capability assessment, full work year AGI task (2000 hours), slower progress rate (320 days doubling), and assumption that AGI tasks are much harder than benchmarks."
    },
    "Aggressive": {
        "params": {
            "start_task_length": 2.5,  # Higher capability assessment
            "agi_task_length": 80,     # Just 2 work weeks
            "doubling_time": 118,      # 2024-2025 trend (faster)
            "acceleration": 0.9,       # Superexponential progress
            "shift": 120,              # Significant shift
            "elicitation_boost": 1.5,  # Better scaffolding
            "reliability_needed": 0.5, # METR standard
            "task_type_penalty": 0.5,  # AGI tasks easier than METR tasks
            "reference_date": O3_LAUNCH_DATE, 
            "correlated": True,        # Correlated sampling
            "use_parallel": True,
        },
        "description": "Optimistic timeline with faster progress. Uses higher capability assessment, 80-hour AGI threshold (2 weeks of work), fast doubling time (118 days), and superexponential progress assumptions, along with significant lead time from private models."
    },
    "Middle Ground": {
        "params": {
            "start_task_length": 1.75, # Default o3 value
            "agi_task_length": 400,    # Few months of work
            "doubling_time": 160,      # Between default and aggressive
            "acceleration": 1.0,       # Exponential progress
            "shift": 60,               # Moderate shift
            "elicitation_boost": 1.0,  # No adjustment
            "reliability_needed": 0.7, # Moderate reliability
            "task_type_penalty": 1.5,  # AGI tasks somewhat harder
            "reference_date": O3_LAUNCH_DATE,
            "correlated": False,
            "use_parallel": True,
        },
        "description": "Moderate forecasting assumptions between Default and Aggressive. Uses 400-hour AGI task length (10 weeks), 160-day doubling time, and moderate reliability requirements."
    }
} 

# Model task time dictionary (in hours)
MODEL_TASK_TIMES = {
    "o3": DEFAULT_PARAMS["start_task_length"],  # ~1hr45min
    "o4-mini": 1.50,  # ~1hr30min
    "Claude 3.7 Sonnet": 0.983,  # 59min
    "o1": 0.65,  # 39min
    "Claude 3.5 Sonnet (new)": 0.467,  # 28min
    "GPT4.5": 0.467,  # ~28min
    "DeepSeek R1": 0.417,  # ~25min
    "o1 Preview": 0.367,  # 22min
    "Claude 3.5 Sonnet (old)": 0.3,  # 18min
    "GPT4o": 0.15,  # 9min
    "Claude 3 Opus": 0.1,  # 6min
    "GPT3.5 Turbo": 0.01,  # 36sec
} 
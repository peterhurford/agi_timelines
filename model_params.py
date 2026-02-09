import numpy as np
import squigglepy as sq

from datetime import datetime

from model_data import model_data

# -----------
# GET INITIAL
# -----------

# Default parameters
reliability_metric = 'performance_50p'
doubling_time_5percentile = 70 # days
doubling_time_95percentile = 333 # days
end_year = 2030

# START TASK LENGTH: How many max minutes of all AGI-relevant tasks can AI reliably do to a sufficient degree of reliability?

# define current best
best_model = max(
    (m for m in model_data.values() if m.get(reliability_metric) is not None), 
    key=lambda m: m[reliability_metric]
)
current_best = best_model[reliability_metric]
current_best_date = best_model['launch_date']


# ------------------
# DEFINE ADJUSTMENTS
# ------------------

# 1. Elicitiation boost - Can you get a boost to scores by iterating on scaffolding and other elicitation techniques? How much should we multiply up to adjust for this?
elicitation_boost = sq.mixture([
        [0.3, 1.0],  # 30% chance no, 40% chance you can get a 1.2x speed up, 30% chance of 1.5x.
        [0.4, 1.2],
        [0.3, 1.5],
    ])

# 2. METR didn't use human-cost inference compute. Can you get a boost to scores by increasing inference compute to human level? How much should we multiply up task length to adjust for this?
inference_compute_adj = sq.lognorm(lognorm_mean=2, lognorm_sd=1, lclip=1)

# 3. What amount of reliability will we need? Is 50% sufficient? Probability distribution over hypotheses
reliability_needed = sq.mixture(
    [[0.2, 0.5], [0.4, 0.8], [0.2, 0.9], [0.1, 0.95], [0.1, 0.99]]
)

# Turn the reliability number into an actual adjustment mulitiplier
def reliability_count_to_penalty(reliability):
    r = np.asarray(reliability, dtype=float)
    reliability = np.array([0.50, 0.80, 0.90, 0.95, 0.99])
    penalty = np.array([1.0, 0.25, 0.25**2, 0.25**3, 0.25**4])
    matches = r[..., None] == reliability
    hit_any = matches.any(axis=-1)
    idx = matches.argmax(axis=-1)
    out = np.full_like(r, np.nan, dtype=float)
    out[hit_any] = penalty[idx[hit_any]]
    return out


# 4. Adjustment for task type penalty -- How much multiplier should we adjust down to adjust for the fact that METR's suite is not all AGI relevant tasks?
task_type_penalty = sq.mixture([
        [0.1, 1],  # 10% chance that METR's software tasks are sufficient for AGI
        [0.9, 1 / sq.lognorm(5, 200)],
    ])  # 90% chance that true AGI tasks are 5-200x harder than METR's software tasks
# This is roughly based on comparing OSWorld to METR https://metr.org/blog/2025-07-14-how-does-time-horizon-vary-across-domains/

# 5. Adjustment for messy tasks -- benchmark tasks are clean, very well specified, and close-ended. Real world tasks are not. How much to adjust for that?
messy_tasks_penalty = sq.mixture([[0.1, 1], [0.9, 1 / sq.norm(1, 10)]])

# 6. Experience pentalty -- How much multiplier should we adjust down to adjust for the fact that METR's contractors are not max skill?
# METR March 2025 'horizon length' paper finds maintainers do 5-18x better than contractors. Jul 2025 paper finds negative speedup from experienced SWEs using AI.
experience_penalty = sq.mixture([
        [0.3, 1],  # 30% case for not adjusting for this, because a lot of the economy is not max skill and maybe max skill isn't really necessary
        [0.4, 1 / sq.lognorm(5, 18)],  # 40% for making METR's 5-18x adjustment
        [0.3, 1 / sq.lognorm(1, 5)],
    ])  # 30% for making a milder compromise adjustment


# ---------------------
# CREATE DISTRIBUTION #
# ---------------------

# Start with current best, add elicitation boost
start_task_length = current_best * elicitation_boost

# add inference compute adjustment
start_task_length = start_task_length * inference_compute_adj

# add reliability penalty
start_task_length = start_task_length * sq.dist_fn(
    reliability_needed, reliability_count_to_penalty
)

# Add task type penalty
start_task_length *= task_type_penalty

# Add messy tasks penalty
start_task_length *= messy_tasks_penalty

# Add experience penalty
start_task_length *= experience_penalty

# Add a minimum value of 1sec
start_task_length = sq.dist_max(1 / 60 / 60, start_task_length)


# -------------------
# ADDITIONAL PARAMS #
# -------------------

# -----------
# AGI TASK LENGTH: What length of time (in hours) is needed to be AGI?
agi_task_length = sq.lognorm(80, 2000, credibility=80, lclip=40)

# -----------
# DOUBLING TIME: How many days does it take to double the effective task length?
doubling_time = sq.lognorm(doubling_time_5percentile, doubling_time_95percentile, credibility=90)  # `Track Acceleration` Boostrap Analysis 95% CI range

# -----------
# SHIFT PARAMETER: How much earlier (in days) are capabilities developed internally versus made available to the public?
shift = sq.norm(30, 30 * 9, credibility=90, lclip=0)

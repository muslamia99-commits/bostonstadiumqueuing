"""
MBTA Foxboro Commuter Rail - 2026 FIFA World Cup Service Simulation
====================================================================
Models the Foxboro Express line as an M/D/c queue:
  - M: Poisson passenger arrivals (stochastic demand)
  - D: Deterministic train service (fixed capacity & dwell time)
  - c: Multiple trains running concurrently (configurable)

Research Questions Addressed:
  1. Mode shift required given 75% parking reduction
  2. Minimum service rate to keep P(wait) < 5%
  3. Impact on regular Foxboro/Franklin line riders

Usage:
    python foxboro_simulation.py
"""

import simpy
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict
import math

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RANDOM_SEED = 42

# Stadium / Demand Parameters
STADIUM_CAPACITY = 66_000          # Gillette Stadium seating
PARKING_REDUCTION = 0.75           # 75% fewer parking spots
PRE_GAME_WINDOW_MIN = 180          # minutes before kickoff with significant arrivals
POST_GAME_WINDOW_MIN = 90          # minutes after final whistle with significant departures

# Train / Service Parameters (Deterministic — the D in M/D/c)
TRAIN_CAPACITY = 1_000             # passengers per bi-level consist (4 cars × ~250)
DWELL_TIME_MIN = 5.0               # fixed dwell time at South Station (minutes)
TRAVEL_TIME_MIN = 55.0             # South Station → Foxboro express (minutes)
TURNAROUND_TIME_MIN = 20.0         # time at Foxboro before return trip

# Platform / Infrastructure
NUM_PLATFORMS = 2                  # usable platforms at South Station for Foxboro service

# Simulation Parameters
SIM_DURATION_MIN = PRE_GAME_WINDOW_MIN + POST_GAME_WINDOW_MIN  # total window modeled

# Game Profiles — 7 World Cup matches at Gillette
# Each dict: kickoff_local (str), expected_fill_rate (0–1), day_of_week
GAME_PROFILES = [
    {"name": "Match 1",  "kickoff": "2:00 PM",  "fill_rate": 0.90, "day": "Weekday"},
    {"name": "Match 2",  "kickoff": "5:00 PM",  "fill_rate": 0.95, "day": "Weekend"},
    {"name": "Match 3",  "kickoff": "8:00 PM",  "fill_rate": 0.98, "day": "Weekday"},
    {"name": "Match 4",  "kickoff": "2:00 PM",  "fill_rate": 0.85, "day": "Weekend"},
    {"name": "Match 5",  "kickoff": "5:00 PM",  "fill_rate": 0.92, "day": "Weekday"},
    {"name": "Match 6",  "kickoff": "8:00 PM",  "fill_rate": 1.00, "day": "Weekend"},
    {"name": "Match 7",  "kickoff": "5:00 PM",  "fill_rate": 0.97, "day": "Weekend"},  # Final
]

# ─────────────────────────────────────────────
# ANALYTICAL TOOLS (Erlang C)
# ─────────────────────────────────────────────

def erlang_c(c: int, lam: float, mu: float) -> float:
    """
    Compute the Erlang C probability — P(arriving customer must wait).

    Parameters
    ----------
    c   : number of servers (trains per hour)
    lam : arrival rate (passengers per minute)
    mu  : service rate per server (passengers per minute, = capacity / dwell)

    Returns
    -------
    P(wait) in [0, 1], or 1.0 if system is saturated (rho >= 1)
    """
    rho = lam / (c * mu)  # utilization per server
    if rho >= 1.0:
        return 1.0  # system unstable

    # Traffic intensity
    a = lam / mu

    # Numerator of Erlang C
    numerator = (a ** c / math.factorial(c)) * (1 / (1 - rho))

    # Denominator: sum_{k=0}^{c-1} a^k/k!  +  numerator
    erlang_b_sum = sum(a**k / math.factorial(k) for k in range(c))
    denominator = erlang_b_sum + numerator

    return numerator / denominator


def mode_shift_analysis(stadium_capacity: int,
                        fill_rate: float,
                        parking_reduction: float,
                        baseline_transit_share: float = 0.10) -> Dict:
    """
    Estimate the required mode shift to commuter rail after parking reduction.

    Assumptions
    -----------
    - Baseline: ~10% of attendees use transit for typical NFL games
    - Average car occupancy: 2.5 persons
    - Parking removed = demand that must shift to transit or other modes
    """
    attendance = int(stadium_capacity * fill_rate)
    original_parking = 20_000
    reduced_parking = int(original_parking * (1 - parking_reduction))

    # Persons who previously drove (rough estimate: 1 car per 2.5 attendees filling old lots)
    persons_displaced = (original_parking - reduced_parking) * 2.5

    # Required rail ridership (displaced drivers who choose rail over rideshare/other)
    # Assume 60% of displaced drivers shift to rail, 40% to rideshare/carpool
    rail_shift = persons_displaced * 0.60
    new_transit_share = (baseline_transit_share * attendance + rail_shift) / attendance

    return {
        "attendance": attendance,
        "original_parking": original_parking,
        "reduced_parking": reduced_parking,
        "persons_displaced": int(persons_displaced),
        "estimated_rail_ridership": int(rail_shift),
        "new_transit_share_pct": round(new_transit_share * 100, 1),
    }


# ─────────────────────────────────────────────
# METRICS COLLECTOR
# ─────────────────────────────────────────────

@dataclass
class Metrics:
    """Collects per-passenger and per-train statistics during simulation."""
    wait_times: List[float] = field(default_factory=list)
    queue_lengths: List[float] = field(default_factory=list)
    train_loads: List[int] = field(default_factory=list)
    passengers_boarded: int = 0
    passengers_missed: int = 0   # arrived but sim ended before boarding
    timestamps: List[float] = field(default_factory=list)

    def record_wait(self, wait: float):
        self.wait_times.append(wait)

    def record_queue(self, t: float, length: int):
        self.timestamps.append(t)
        self.queue_lengths.append(length)

    def record_train(self, load: int):
        self.train_loads.append(load)
        self.passengers_boarded += load

    def summary(self) -> Dict:
        if not self.wait_times:
            return {"error": "No data collected"}
        return {
            "avg_wait_min":        round(np.mean(self.wait_times), 2),
            "p95_wait_min":        round(np.percentile(self.wait_times, 95), 2),
            "max_wait_min":        round(np.max(self.wait_times), 2),
            "pct_wait_over_0":     round(np.mean([w > 0 for w in self.wait_times]) * 100, 1),
            "avg_queue_length":    round(np.mean(self.queue_lengths), 1) if self.queue_lengths else 0,
            "max_queue_length":    max(self.queue_lengths) if self.queue_lengths else 0,
            "avg_train_load":      round(np.mean(self.train_loads), 0) if self.train_loads else 0,
            "total_trains_run":    len(self.train_loads),
            "passengers_boarded":  self.passengers_boarded,
            "passengers_missed":   self.passengers_missed,
        }


# ─────────────────────────────────────────────
# SIMPY PROCESSES
# ─────────────────────────────────────────────

def passenger_generator(env: simpy.Environment,
                         platform: simpy.Resource,
                         metrics: Metrics,
                         lam_per_min: float,
                         phase: str = "pre_game"):
    """
    Generate passengers according to a Poisson process (M in M/D/c).

    Arrival rate `lam_per_min` varies by phase:
      - pre_game: ramps up toward kickoff
      - post_game: burst immediately after final whistle
    """
    t = 0
    while True:
        # Poisson inter-arrival: exponential with mean 1/lambda
        inter_arrival = random.expovariate(lam_per_min)
        yield env.timeout(inter_arrival)
        t += inter_arrival

        # Spawn a passenger process
        env.process(passenger(env, platform, metrics))


def passenger(env: simpy.Environment,
              platform: simpy.Resource,
              metrics: Metrics):
    """
    A single passenger arrives, waits for a train, and boards.
    Models the customer side of the M/D/c queue.
    """
    arrive_time = env.now

    # Record queue snapshot
    metrics.record_queue(arrive_time, len(platform.queue))

    # Request a spot on the next available train (via platform resource)
    with platform.request() as req:
        yield req
        wait = env.now - arrive_time
        metrics.record_wait(wait)
        # Boarding itself is deterministic (absorbed into train dwell process)
        yield env.timeout(0)  # instantaneous — dwell handled in train process


def train_dispatcher(env: simpy.Environment,
                     platform: simpy.Resource,
                     metrics: Metrics,
                     headway_min: float,
                     sim_duration: float):
    """
    Dispatch trains on a fixed headway (D in M/D/c — deterministic service).

    Each train:
      1. Occupies a platform slot
      2. Dwells for DWELL_TIME_MIN (fixed)
      3. Boards up to TRAIN_CAPACITY passengers from the queue
      4. Departs
    """
    train_id = 0
    while env.now < sim_duration:
        yield env.timeout(headway_min)  # fixed schedule interval
        train_id += 1
        env.process(train_service(env, platform, metrics, f"Train {train_id}"))


def train_service(env: simpy.Environment,
                  platform: simpy.Resource,
                  metrics: Metrics,
                  name: str):
    """
    A single train's service cycle at South Station / Foxboro.
    Deterministic dwell time — the D in M/D/c.
    """
    arrive = env.now
    print(f"  [{arrive:6.1f} min] {name} arrives at platform | Queue: {len(platform.queue)}")

    # Fixed dwell — deterministic service time
    yield env.timeout(DWELL_TIME_MIN)

    # Count how many passengers were served (limited by capacity)
    # Passengers already holding platform.request() are "boarded"
    passengers_this_train = min(platform.capacity, TRAIN_CAPACITY)
    metrics.record_train(passengers_this_train)

    depart = env.now
    print(f"  [{depart:6.1f} min] {name} departs | Boarded: {passengers_this_train} | Queue remaining: {len(platform.queue)}")


# ─────────────────────────────────────────────
# SCENARIO RUNNER
# ─────────────────────────────────────────────

def run_scenario(game: Dict,
                 headway_min: float,
                 num_platforms: int = NUM_PLATFORMS,
                 verbose: bool = True) -> Dict:
    """
    Run a full pre-game arrival simulation for one World Cup match.

    Parameters
    ----------
    game         : game profile dict (from GAME_PROFILES)
    headway_min  : train frequency in minutes (what we're optimizing)
    num_platforms: number of concurrent train slots (c in M/D/c)
    verbose      : print train-level events

    Returns
    -------
    dict with simulation metrics + analytical Erlang C comparison
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Demand estimation ──────────────────────────────────────────
    shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
    rail_demand = shift["estimated_rail_ridership"]

    # Spread pre-game arrivals over PRE_GAME_WINDOW_MIN with a ramp-up shape
    # More passengers arrive in the final 60 min before kickoff
    # Model as average over the window, with peak ~2x average in final hour
    avg_lam = rail_demand / PRE_GAME_WINDOW_MIN  # passengers per minute (mean)

    # ── Analytical: Erlang C ──────────────────────────────────────
    # Service rate: one train serves TRAIN_CAPACITY pax in DWELL_TIME_MIN
    mu = TRAIN_CAPACITY / DWELL_TIME_MIN          # passengers per minute per "server"
    c = num_platforms                              # concurrent trains
    p_wait_analytical = erlang_c(c, avg_lam, mu)

    # ── SimPy Simulation ─────────────────────────────────────────
    env = simpy.Environment()
    metrics = Metrics()

    # Platform resource: c = num_platforms (each can hold one train at a time)
    platform = simpy.Resource(env, capacity=num_platforms)

    # Launch passenger generator and train dispatcher
    env.process(passenger_generator(env, platform, metrics, avg_lam))
    env.process(train_dispatcher(env, platform, metrics, headway_min, SIM_DURATION_MIN))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {game['name']} | Kickoff: {game['kickoff']} | {game['day']}")
        print(f"  Fill Rate: {game['fill_rate']*100:.0f}% | Est. Rail Riders: {rail_demand:,}")
        print(f"  Headway: {headway_min} min | Platforms: {num_platforms}")
        print(f"  Erlang C P(wait) [analytical]: {p_wait_analytical*100:.2f}%")
        print(f"{'='*60}")

    env.run(until=SIM_DURATION_MIN)

    # Mark unboarded passengers as missed
    metrics.passengers_missed = len(platform.queue)

    sim_summary = metrics.summary()
    sim_summary["erlang_c_p_wait_pct"] = round(p_wait_analytical * 100, 2)
    sim_summary["estimated_rail_demand"] = rail_demand
    sim_summary["avg_arrival_rate_per_min"] = round(avg_lam, 2)
    sim_summary["utilization_rho"] = round(avg_lam / (c * mu), 4)
    sim_summary["headway_min"] = headway_min
    sim_summary["game"] = game["name"]

    return sim_summary


def find_optimal_headway(game: Dict,
                         headway_range: range = range(5, 35, 5),
                         target_p_wait: float = 0.05) -> Dict:
    """
    Sweep headway values and find the minimum headway to keep P(wait) < target.
    Uses the Erlang C formula for fast analytical sweep, then confirms with SimPy.

    Parameters
    ----------
    game           : game profile
    headway_range  : headway values to test (minutes)
    target_p_wait  : threshold (default 5% per research question 2)

    Returns
    -------
    dict with recommended headway and full sweep results
    """
    shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
    lam = shift["estimated_rail_ridership"] / PRE_GAME_WINDOW_MIN
    mu = TRAIN_CAPACITY / DWELL_TIME_MIN
    c = NUM_PLATFORMS

    results = []
    for hw in headway_range:
        # Trains per minute = 1/headway; but Erlang C uses server count, not headway
        # For a fixed c-platform system, headway controls how many trains we CAN dispatch
        # Use analytical formula for sweep
        p_wait = erlang_c(c, lam, mu)
        rho = lam / (c * mu)
        results.append({
            "headway_min": hw,
            "p_wait_pct": round(p_wait * 100, 2),
            "rho": round(rho, 4),
            "feasible": p_wait <= target_p_wait
        })

    # Recommended: smallest headway that satisfies constraint
    feasible = [r for r in results if r["feasible"]]
    recommended_hw = feasible[0]["headway_min"] if feasible else None

    return {
        "game": game["name"],
        "recommended_headway_min": recommended_hw,
        "sweep": results,
        "estimated_rail_demand": shift["estimated_rail_ridership"],
    }


# ─────────────────────────────────────────────
# REGULAR RIDER IMPACT ANALYSIS
# ─────────────────────────────────────────────

def regular_rider_impact(headway_express_min: float,
                         regular_trains_per_hour: int = 2,
                         regular_rider_count: int = 800) -> Dict:
    """
    Estimate the impact of express World Cup service on regular Foxboro/Franklin
    line riders (Research Question 3).

    When express trains consume platform capacity, regular trains may be delayed
    or cancelled. This function estimates the capacity conflict.

    Parameters
    ----------
    headway_express_min    : headway between World Cup express trains
    regular_trains_per_hour: baseline Franklin/Foxboro local service frequency
    regular_rider_count    : typical non-event ridership on the line per hour
    """
    express_per_hour = 60 / headway_express_min
    total_trains_per_hour = express_per_hour + regular_trains_per_hour
    platform_utilization = total_trains_per_hour / (NUM_PLATFORMS * (60 / DWELL_TIME_MIN))

    # If utilization > 1, regular trains will face delays
    delay_risk = platform_utilization > 0.85  # flag at 85% utilization

    # Estimate regular riders affected: proportion of hour dominated by express
    express_fraction = express_per_hour / total_trains_per_hour
    riders_impacted = int(regular_rider_count * express_fraction)

    return {
        "express_trains_per_hour": round(express_per_hour, 1),
        "regular_trains_per_hour": regular_trains_per_hour,
        "total_trains_per_hour": round(total_trains_per_hour, 1),
        "platform_utilization_pct": round(platform_utilization * 100, 1),
        "delay_risk_flag": delay_risk,
        "regular_riders_impacted_estimate": riders_impacted,
    }


# ─────────────────────────────────────────────
# MAIN: RUN ALL SCENARIOS
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  MBTA FOXBORO LINE — 2026 FIFA WORLD CUP SIMULATION")
    print("="*60)

    # ── 1. Mode Shift Analysis ────────────────────────────────────
    print("\n── SECTION 1: MODE SHIFT ANALYSIS ──────────────────────────")
    for game in GAME_PROFILES:
        shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
        print(f"\n{game['name']} ({game['kickoff']}, {game['day']})")
        print(f"  Attendance:               {shift['attendance']:,}")
        print(f"  Reduced parking spots:    {shift['reduced_parking']:,}  (from {shift['original_parking']:,})")
        print(f"  Persons displaced by car: {shift['persons_displaced']:,}")
        print(f"  Estimated rail demand:    {shift['estimated_rail_ridership']:,}")
        print(f"  New transit share:        {shift['new_transit_share_pct']}%")

    # ── 2. Optimal Headway Sweep (Erlang C) ──────────────────────
    print("\n\n── SECTION 2: OPTIMAL HEADWAY SWEEP (Target P(wait) < 5%) ─")
    for game in GAME_PROFILES:
        result = find_optimal_headway(game, headway_range=range(5, 35, 5))
        print(f"\n{game['name']}: Recommended headway = {result['recommended_headway_min']} min")
        print(f"  {'Headway':>8} | {'P(wait)':>8} | {'Rho':>6} | {'OK?':>5}")
        print(f"  {'-'*36}")
        for row in result["sweep"]:
            ok = "✓" if row["feasible"] else "✗"
            print(f"  {row['headway_min']:>7}m | {row['p_wait_pct']:>7}% | {row['rho']:>6.4f} | {ok:>5}")

    # ── 3. Full SimPy Simulation — Representative Games ──────────
    print("\n\n── SECTION 3: SIMPY SIMULATION — SELECTED MATCHES ─────────")
    test_games = [GAME_PROFILES[0], GAME_PROFILES[2], GAME_PROFILES[5]]  # low / mid / sellout
    for game in test_games:
        result = run_scenario(game, headway_min=15, num_platforms=NUM_PLATFORMS)
        print(f"\n  Simulation Results for {game['name']}:")
        for k, v in result.items():
            if k not in ("game",):
                print(f"    {k:<35} {v}")

    # ── 4. Regular Rider Impact ───────────────────────────────────
    print("\n\n── SECTION 4: REGULAR RIDER IMPACT ANALYSIS ────────────────")
    for hw in [10, 15, 20, 30]:
        impact = regular_rider_impact(headway_express_min=hw)
        flag = "⚠ DELAY RISK" if impact["delay_risk_flag"] else "  OK"
        print(f"\n  Express headway: {hw} min")
        print(f"    Express trains/hr:    {impact['express_trains_per_hour']}")
        print(f"    Platform utilization: {impact['platform_utilization_pct']}%  {flag}")
        print(f"    Regular riders at risk: {impact['regular_riders_impacted_estimate']:,}")

    print("\n" + "="*60)
    print("  Simulation complete.")
    print("="*60 + "\n")
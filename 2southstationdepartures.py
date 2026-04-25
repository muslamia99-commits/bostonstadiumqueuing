"""
MBTA Foxboro Commuter Rail - 2026 FIFA World Cup Service Simulation
====================================================================
Models the Foxboro Express line as an M/D/c queue:
  - M: Poisson passenger arrivals (stochastic demand)
  - D: Deterministic train service (fixed capacity & dwell time)
  - c: Multiple trains running concurrently (configurable)

Boarding Groups:
  - Passengers are assigned to groups 1–5 based on arrival time window
  - Group 1 = earliest arrivals (highest priority), Group 5 = latest (lowest)
  - SimPy PriorityResource ensures earlier groups board first

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
STADIUM_CAPACITY = 65_000
PARKING_REDUCTION = 0.75
PRE_GAME_WINDOW_MIN = 390          # minutes before kickoff with significant arrivals

# Train / Service Parameters (Deterministic — the D in M/D/c)
TRAIN_CAPACITY = 180
DWELL_TIME_MIN = 5.0               # fixed dwell time at South Station (minutes)
TRAVEL_TIME_MIN = 55.0
TURNAROUND_TIME_MIN = 20.0
MAX_RAIL_RIDERSHIP = 20_000

# Platform / Infrastructure
NUM_PLATFORMS = 2

# Boarding Groups
NUM_BOARDING_GROUPS = 5            # passengers split into 5 groups by arrival time
# Group 1 = first 1/5 of window (earliest arrivers, board first)
# Group 5 = last 1/5 of window  (latest arrivers, board last)
# SimPy priority: lower number = higher priority, so group number = priority value

# Simulation Parameters
SIM_DURATION_MIN = PRE_GAME_WINDOW_MIN

# Game Profiles — 7 World Cup matches at Gillette
GAME_PROFILES = [
    {"name": "Match 5: Haiti vs. Scotland, June 13th at 9 PM",   "Departure Time": "2:30 PM",  "fill_rate": 1.00, "day": "Weekend"},
    {"name": "Match 18: Iraq vs. Norway, June 16th at 6 PM",     "Departure Time": "11:30 AM", "fill_rate": 1.00, "day": "Weekday"},
    {"name": "Match 30: Scotland vs. Morocco, June 19th at 6 PM","Departure Time": "11:30 AM", "fill_rate": 1.00, "day": "Weekday"},
    {"name": "Match 45: England vs. Ghana, June 23rd at 4 PM",   "Departure Time": "9:30 AM",  "fill_rate": 1.00, "day": "Weekday"},
    {"name": "Match 61: Norway vs. France, June 26th at 3 PM",   "Departure Time": "8:30 AM",  "fill_rate": 1.00, "day": "Weekday"},
    {"name": "Match 74: Round of 32 Game, June 29th at 4:30 PM", "Departure Time": "10:00 AM", "fill_rate": 1.00, "day": "Weekday"},
    {"name": "Match 97: Quarter Finals, July 9th at 4 PM",       "Departure Time": "9:30 AM",  "fill_rate": 1.00, "day": "Weekday"},
]


# ─────────────────────────────────────────────
# BOARDING GROUP ASSIGNMENT
# ─────────────────────────────────────────────

def assign_boarding_group(arrival_time: float,
                           sim_duration: float,
                           num_groups: int = NUM_BOARDING_GROUPS) -> int:
    """
    Assign a passenger to a boarding group based on when they arrive
    within the simulation window.

    The window is divided into equal slices:
      - Group 1: arrives in first 1/num_groups of the window  (highest priority)
      - Group N: arrives in last  1/num_groups of the window  (lowest priority)

    Parameters
    ----------
    arrival_time : current simulation time when passenger arrives
    sim_duration : total simulation window length
    num_groups   : number of boarding groups (default 5)

    Returns
    -------
    int in [1, num_groups] — lower = earlier arrival = higher boarding priority
    """
    slice_width = sim_duration / num_groups
    group = int(arrival_time / slice_width) + 1
    return min(group, num_groups)  # clamp to num_groups for edge case at end


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
    rho = lam / (c * mu)
    if rho >= 1.0:
        return 1.0

    a = lam / mu
    numerator = (a ** c / math.factorial(c)) * (1 / (1 - rho))
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

    riders_displaced = (original_parking - reduced_parking) * 2.5
    rail_shift = min(riders_displaced * 0.60, MAX_RAIL_RIDERSHIP)
    new_transit_share = (baseline_transit_share * attendance + rail_shift) / attendance

    return {
        "attendance": attendance,
        "original_parking": original_parking,
        "reduced_parking": reduced_parking,
        "riders_displaced": int(riders_displaced),
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
    passengers_missed: int = 0
    timestamps: List[float] = field(default_factory=list)

    # Per-group wait time tracking — keyed by group number 1..NUM_BOARDING_GROUPS
    group_wait_times: Dict[int, List[float]] = field(
        default_factory=lambda: {g: [] for g in range(1, NUM_BOARDING_GROUPS + 1)}
    )

    def record_wait(self, wait: float, group: int):
        self.wait_times.append(wait)
        self.group_wait_times[group].append(wait)

    def record_queue(self, t: float, length: int):
        self.timestamps.append(t)
        self.queue_lengths.append(length)

    def record_train(self, load: int):
        self.train_loads.append(load)
        self.passengers_boarded += load

    def summary(self) -> Dict:
        if not self.wait_times:
            return {"error": "No data collected"}

        result = {
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

        # Append per-group average wait times to summary
        for g, waits in self.group_wait_times.items():
            result[f"group_{g}_avg_wait_min"] = round(np.mean(waits), 2) if waits else None

        return result


# ─────────────────────────────────────────────
# SIMPY PROCESSES
# ─────────────────────────────────────────────

def passenger_generator(env: simpy.Environment,
                         platform: simpy.PriorityResource,
                         metrics: Metrics,
                         lam_per_min: float):
    """
    Generate passengers according to a Poisson process (M in M/D/c).
    Each passenger is assigned a boarding group on arrival, which
    determines their priority in the PriorityResource queue.
    """
    while True:
        inter_arrival = random.expovariate(lam_per_min)
        yield env.timeout(inter_arrival)
        env.process(passenger(env, platform, metrics))


def passenger(env: simpy.Environment,
              platform: simpy.PriorityResource,
              metrics: Metrics):
    """
    A single passenger arrives, gets assigned a boarding group based on
    their arrival time, then waits in priority queue.

    SimPy PriorityResource: lower priority value = served first.
    Group 1 (earliest arrivers) gets priority=1 → boards before Group 5.
    """
    arrive_time = env.now

    # Assign boarding group based on when in the window this passenger arrives
    group = assign_boarding_group(arrive_time, SIM_DURATION_MIN)

    # Record queue snapshot before joining
    metrics.record_queue(arrive_time, len(platform.queue))

    # Request platform with priority = group number
    # Group 1 (priority=1) is served before Group 5 (priority=5)
    with platform.request(priority=group) as req:
        yield req
        wait = env.now - arrive_time
        metrics.record_wait(wait, group)   # track wait by group
        yield env.timeout(0)               # boarding absorbed into train dwell


def train_dispatcher(env: simpy.Environment,
                     platform: simpy.PriorityResource,
                     metrics: Metrics,
                     headway_min: float,
                     sim_duration: float):
    """
    Dispatch trains on a fixed headway (D in M/D/c — deterministic service).
    """
    train_id = 0
    while env.now < sim_duration:
        yield env.timeout(headway_min)
        train_id += 1
        env.process(train_service(env, platform, metrics, f"Train {train_id}"))


def train_service(env: simpy.Environment,
                  platform: simpy.PriorityResource,
                  metrics: Metrics,
                  name: str):
    """
    A single train's service cycle. Deterministic dwell time.
    PriorityResource ensures group 1 passengers board before group 5.
    """
    arrive = env.now
    print(f"  [{arrive:6.1f} min] {name} arrives | Queue: {len(platform.queue)}")

    # Fixed dwell — deterministic service time (the D in M/D/c)
    yield env.timeout(DWELL_TIME_MIN)

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
    Uses PriorityResource so boarding groups are respected.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
    rail_demand = shift["estimated_rail_ridership"]
    avg_lam = rail_demand / PRE_GAME_WINDOW_MIN

    mu = TRAIN_CAPACITY / DWELL_TIME_MIN
    c = num_platforms
    p_wait_analytical = erlang_c(c, avg_lam, mu)

    env = simpy.Environment()
    metrics = Metrics()

    # KEY CHANGE: PriorityResource instead of Resource
    # This is what enforces boarding group order —
    # passengers with lower group numbers (earlier arrivals) get served first
    platform = simpy.PriorityResource(env, capacity=num_platforms)

    env.process(passenger_generator(env, platform, metrics, avg_lam))
    env.process(train_dispatcher(env, platform, metrics, headway_min, SIM_DURATION_MIN))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {game['name']} | {game['day']}")
        print(f"  Fill Rate: {game['fill_rate']*100:.0f}% | Est. Rail Riders: {rail_demand:,}")
        print(f"  Headway: {headway_min} min | Platforms: {num_platforms}")
        print(f"  Boarding Groups: {NUM_BOARDING_GROUPS} (by arrival time window)")
        print(f"  Erlang C P(wait) [analytical]: {p_wait_analytical*100:.2f}%")
        print(f"{'='*60}")

    env.run(until=SIM_DURATION_MIN)

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
    """
    shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
    lam = shift["estimated_rail_ridership"] / PRE_GAME_WINDOW_MIN
    mu = TRAIN_CAPACITY / DWELL_TIME_MIN
    c = NUM_PLATFORMS

    results = []
    for hw in headway_range:
        p_wait = erlang_c(c, lam, mu)
        rho = lam / (c * mu)
        results.append({
            "headway_min": hw,
            "p_wait_pct": round(p_wait * 100, 2),
            "rho": round(rho, 4),
            "feasible": p_wait <= target_p_wait
        })

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
    """
    express_per_hour = 60 / headway_express_min
    total_trains_per_hour = express_per_hour + regular_trains_per_hour
    platform_utilization = total_trains_per_hour / (NUM_PLATFORMS * (60 / DWELL_TIME_MIN))

    delay_risk = platform_utilization > 0.85
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
    print(f"  Boarding Groups: {NUM_BOARDING_GROUPS} (by arrival time window)")
    print("="*60)

    # ── 1. Mode Shift Analysis ────────────────────────────────────
    print("\n── SECTION 1: MODE SHIFT ANALYSIS ──────────────────────────")
    for game in GAME_PROFILES:
        shift = mode_shift_analysis(STADIUM_CAPACITY, game["fill_rate"], PARKING_REDUCTION)
        print(f"\n{game['name']} ({game['Departure Time']}, {game['day']})")
        print(f"  Attendance:               {shift['attendance']:,}")
        print(f"  Reduced parking spots:    {shift['reduced_parking']:,}  (from {shift['original_parking']:,})")
        print(f"  Persons displaced by car: {shift['riders_displaced']:,}")
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
    print("\n\n── SECTION 3: SIMPY SIMULATION BOSTON STADIUM MATCH ─────────")
    for game in GAME_PROFILES:
        result = run_scenario(game, headway_min=15, num_platforms=NUM_PLATFORMS)
        print(f"\n  Simulation Results for {game['name']}:")
        for k, v in result.items():
            if k != "game":
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
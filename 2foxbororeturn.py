"""
MBTA Foxboro → South Station Return Simulation
===============================================
Models the post-game departure from Foxboro after a World Cup match.

Key assumptions:
  - Service begins 30 minutes after the final whistle
  - 1 platform available at Foxboro for boarding
  - 14 trains total dispatched (FCFS — no boarding groups)
  - Train capacity: 180 passengers
  - Dwell time: 5 minutes (deterministic)
  - Travel time back to South Station: 55 minutes
  - Passengers arrive at Foxboro station via Poisson process

Outputs:
  - Console log of each train departure
  - Cumulative departure plot (passengers vs. time)
  - Required arrival rate λ to fill all 14 trains
  - Total time to clear the stadium via rail
"""

import simpy
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import List, Dict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RANDOM_SEED = 42

# Train / Service Parameters
TRAIN_CAPACITY       = 180     # passengers per train
DWELL_TIME_MIN       = 5.0     # fixed dwell at Foxboro platform (deterministic)
TRAVEL_TIME_MIN      = 55.0    # Foxboro → South Station
NUM_TRAINS           = 14      # total return trains to dispatch
NUM_PLATFORMS        = 1       # only 1 platform at Foxboro for return service

# Post-game timing
WHISTLE_TO_SERVICE   = 30      # minutes after final whistle before first train departs
HEADWAY_MIN          = 10      # minutes between trains (fixed schedule)

# Total simulation window: time from whistle to last possible train departure
# 30 min delay + (14 trains × 10 min headway) + buffer
SIM_DURATION_MIN     = WHISTLE_TO_SERVICE + (NUM_TRAINS * HEADWAY_MIN) + 30


# ─────────────────────────────────────────────
# ARRIVAL RATE ANALYSIS
# ─────────────────────────────────────────────

def required_arrival_rate(num_trains: int,
                           train_capacity: int,
                           service_start_min: int,
                           headway_min: float) -> Dict:
    """
    Calculate the arrival rate λ needed to fill all num_trains trains.

    Total demand = num_trains × train_capacity passengers.
    These passengers must arrive before the last train departs.

    Last train departs at: service_start + (num_trains - 1) × headway + dwell
    So passengers must arrive across that full window.

    Returns
    -------
    dict with λ, total demand, window length, and trains × capacity breakdown
    """
    total_demand      = num_trains * train_capacity
    last_departure    = service_start_min + (num_trains - 1) * headway_min + DWELL_TIME_MIN
    arrival_window    = last_departure  # passengers arrive from t=0 (whistle) onward

    # λ needed so that E[arrivals] = total_demand over the window
    lam_per_min       = total_demand / arrival_window

    # Also compute how long it takes all 14 trains to complete the trip to South Station
    last_train_arrival_ss = last_departure + TRAVEL_TIME_MIN

    return {
        "total_passengers_needed":    total_demand,
        "arrival_window_min":         round(arrival_window, 1),
        "required_lam_per_min":       round(lam_per_min, 3),
        "required_lam_per_hour":      round(lam_per_min * 60, 1),
        "first_train_departs_min":    service_start_min + DWELL_TIME_MIN,
        "last_train_departs_min":     round(last_departure, 1),
        "last_train_arrival_SS_min":  round(last_train_arrival_ss, 1),
        "total_elapsed_min":          round(last_train_arrival_ss, 1),
    }


# ─────────────────────────────────────────────
# METRICS COLLECTOR
# ─────────────────────────────────────────────

@dataclass
class ReturnMetrics:
    """Tracks per-train and per-passenger data for the return trip."""
    wait_times:           List[float] = field(default_factory=list)
    queue_at_departure:   List[int]   = field(default_factory=list)
    train_loads:          List[int]   = field(default_factory=list)
    departure_times:      List[float] = field(default_factory=list)  # sim time of departure
    arrival_times_ss:     List[float] = field(default_factory=list)  # sim time at South Station
    passengers_boarded:   int         = 0
    passengers_stranded:  int         = 0  # still in queue when last train departs

    def record_wait(self, wait: float):
        self.wait_times.append(wait)

    def record_departure(self, t: float, load: int, queue_remaining: int):
        self.train_loads.append(load)
        self.departure_times.append(t)
        self.arrival_times_ss.append(t + TRAVEL_TIME_MIN)
        self.queue_at_departure.append(queue_remaining)
        self.passengers_boarded += load

    def summary(self) -> Dict:
        if not self.wait_times:
            return {"error": "No passengers recorded"}
        return {
            "total_trains_dispatched":   len(self.train_loads),
            "passengers_boarded":        self.passengers_boarded,
            "passengers_stranded":       self.passengers_stranded,
            "avg_train_load":            round(np.mean(self.train_loads), 1),
            "trains_at_full_capacity":   sum(1 for l in self.train_loads if l == TRAIN_CAPACITY),
            "avg_wait_min":              round(np.mean(self.wait_times), 2),
            "p95_wait_min":              round(np.percentile(self.wait_times, 95), 2),
            "max_wait_min":              round(np.max(self.wait_times), 2),
            "first_departure_min":       round(self.departure_times[0], 1)  if self.departure_times else None,
            "last_departure_min":        round(self.departure_times[-1], 1) if self.departure_times else None,
            "last_arrival_SS_min":       round(self.arrival_times_ss[-1], 1) if self.arrival_times_ss else None,
        }


# ─────────────────────────────────────────────
# SIMPY PROCESSES
# ─────────────────────────────────────────────

def passenger_generator(env: simpy.Environment,
                         platform: simpy.Resource,
                         metrics: ReturnMetrics,
                         lam_per_min: float,
                         start_time: float = 0.0):
    """
    Generate post-game passengers via Poisson process (FCFS — no priority).
    Passengers begin arriving at the whistle (t=0), not at service start.
    They queue and wait until a train is available.
    """
    # Wait until the whistle before generating arrivals
    yield env.timeout(start_time)

    while True:
        inter_arrival = random.expovariate(lam_per_min)
        yield env.timeout(inter_arrival)
        env.process(passenger_process(env, platform, metrics))


def passenger_process(env: simpy.Environment,
                       platform: simpy.Resource,
                       metrics: ReturnMetrics):
    """
    A single returning passenger: arrives, joins FCFS queue, boards next train.
    """
    arrive_time = env.now
    with platform.request() as req:
        yield req
        wait = env.now - arrive_time
        metrics.record_wait(wait)
        yield env.timeout(0)  # boarding absorbed into train dwell


def train_dispatcher(env: simpy.Environment,
                      platform: simpy.Resource,
                      metrics: ReturnMetrics,
                      service_start: float,
                      headway_min: float,
                      num_trains: int):
    """
    Dispatch num_trains return trains on a fixed headway starting at service_start.
    After all trains depart, record stranded passengers.
    """
    # Wait until service begins (30 min after whistle)
    yield env.timeout(service_start)

    for train_id in range(1, num_trains + 1):
        env.process(train_service(env, platform, metrics, f"Return Train {train_id}"))
        if train_id < num_trains:
            yield env.timeout(headway_min)  # wait before dispatching next

    # After last train dispatched, record stranded passengers
    # Give the last train time to finish dwell before counting
    yield env.timeout(DWELL_TIME_MIN + 1)
    metrics.passengers_stranded = len(platform.queue)


def train_service(env: simpy.Environment,
                   platform: simpy.Resource,
                   metrics: ReturnMetrics,
                   name: str):
    """
    A single return train: dwells for DWELL_TIME_MIN, boards up to TRAIN_CAPACITY,
    then departs for South Station.
    """
    arrive = env.now
    print(f"  [{arrive:6.1f} min] {name} ready for boarding | Queue: {len(platform.queue)}")

    # Deterministic dwell
    yield env.timeout(DWELL_TIME_MIN)

    load = min(platform.capacity, TRAIN_CAPACITY)
    queue_remaining = len(platform.queue)
    metrics.record_departure(env.now, load, queue_remaining)

    print(f"  [{env.now:6.1f} min] {name} departs Foxboro → South Station "
          f"| Boarded: {load} | Queue remaining: {queue_remaining} "
          f"| Arrives SS at: {env.now + TRAVEL_TIME_MIN:.1f} min")


# ─────────────────────────────────────────────
# CUMULATIVE DEPARTURE PLOT
# ─────────────────────────────────────────────

def plot_cumulative_departures(metrics: ReturnMetrics,
                                lam_per_min: float,
                                save_path: str = "return_departures.png"):
    """
    Plot cumulative passengers departed from Foxboro and cumulative arrivals
    at South Station over simulation time.

    Also overlays the theoretical cumulative arrival curve so you can
    see when demand meets capacity.
    """
    dep_times  = np.array(metrics.departure_times)
    arr_times  = np.array(metrics.arrival_times_ss)
    loads      = np.array(metrics.train_loads)

    cum_departed  = np.cumsum(loads)
    cum_arrived   = np.cumsum(loads)  # same passengers, just shifted by travel time

    # Theoretical cumulative arrivals at Foxboro station
    t_range = np.linspace(0, dep_times[-1] + 10, 300)
    cum_demand = lam_per_min * t_range  # E[arrivals] under Poisson

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), facecolor="#0d1117")
    fig.suptitle("MBTA Foxboro Return Service — 2026 FIFA World Cup\nCumulative Passenger Flow",
                 fontsize=14, color="white", fontweight="bold", y=0.98)

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")

    # ── Panel 1: Departures from Foxboro ─────────────────────────
    ax1.step(dep_times, cum_departed, where="post",
             color="#58a6ff", linewidth=2.5, label="Cumulative boarded (Foxboro)")
    ax1.plot(t_range, np.minimum(cum_demand, NUM_TRAINS * TRAIN_CAPACITY),
             color="#f78166", linewidth=1.5, linestyle="--", label="Expected demand (Poisson)")

    # Annotate each train departure
    for i, (t, c) in enumerate(zip(dep_times, cum_departed)):
        ax1.annotate(f"T{i+1}\n{loads[i]} pax",
                     xy=(t, c), xytext=(t + 1, c - 15),
                     fontsize=7, color="#8b949e",
                     arrowprops=dict(arrowstyle="-", color="#30363d", lw=0.8))

    ax1.axvline(WHISTLE_TO_SERVICE, color="#3fb950", linewidth=1.2,
                linestyle=":", label=f"Service start (+{WHISTLE_TO_SERVICE} min)")
    ax1.set_xlabel("Minutes after final whistle")
    ax1.set_ylabel("Cumulative passengers")
    ax1.set_title("Foxboro Platform — Cumulative Boardings")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(TRAIN_CAPACITY))

    # ── Panel 2: Arrivals at South Station ───────────────────────
    ax2.step(arr_times, cum_arrived, where="post",
             color="#3fb950", linewidth=2.5, label="Cumulative arrived (South Station)")

    for i, (t, c) in enumerate(zip(arr_times, cum_arrived)):
        ax2.annotate(f"T{i+1}",
                     xy=(t, c), xytext=(t + 1, c - 12),
                     fontsize=7, color="#8b949e")

    ax2.set_xlabel("Minutes after final whistle")
    ax2.set_ylabel("Cumulative passengers")
    ax2.set_title("South Station — Cumulative Arrivals")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(TRAIN_CAPACITY))

    # ── Shared x-axis labels for key events ──────────────────────
    for ax in axes:
        ax.set_xlim(left=0)
        xticks = list(ax.get_xticks())
        ax.set_xticks(sorted(set(xticks)))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*65)
    print("  MBTA FOXBORO RETURN SERVICE — POST-GAME SIMULATION")
    print("="*65)

    # ── Step 1: Required arrival rate ────────────────────────────
    print("\n── SECTION 1: REQUIRED ARRIVAL RATE TO FILL ALL 14 TRAINS ──")
    rate_info = required_arrival_rate(
        num_trains     = NUM_TRAINS,
        train_capacity = TRAIN_CAPACITY,
        service_start_min = WHISTLE_TO_SERVICE,
        headway_min    = HEADWAY_MIN,
    )
    print(f"\n  Total passengers needed:       {rate_info['total_passengers_needed']:,}")
    print(f"  Arrival window:                {rate_info['arrival_window_min']} min")
    print(f"  Required λ:                    {rate_info['required_lam_per_min']} pax/min")
    print(f"                                 ({rate_info['required_lam_per_hour']} pax/hr)")
    print(f"  First train departs at:        +{rate_info['first_train_departs_min']} min")
    print(f"  Last train departs Foxboro:    +{rate_info['last_train_departs_min']} min")
    print(f"  Last train arrives S. Station: +{rate_info['last_train_arrival_SS_min']} min")
    print(f"  Total elapsed time:            {rate_info['total_elapsed_min']} min "
          f"({rate_info['total_elapsed_min']/60:.1f} hrs after whistle)")

    # ── Step 2: SimPy Simulation ──────────────────────────────────
    print("\n── SECTION 2: SIMPY RETURN SIMULATION ──────────────────────")
    lam = rate_info["required_lam_per_min"]

    env      = simpy.Environment()
    metrics  = ReturnMetrics()
    platform = simpy.Resource(env, capacity=NUM_PLATFORMS)

    # Passengers start arriving at t=0 (the whistle)
    # Trains start at t=WHISTLE_TO_SERVICE
    env.process(passenger_generator(env, platform, metrics,
                                     lam_per_min=lam,
                                     start_time=0.0))
    env.process(train_dispatcher(env, platform, metrics,
                                  service_start=WHISTLE_TO_SERVICE,
                                  headway_min=HEADWAY_MIN,
                                  num_trains=NUM_TRAINS))

    print(f"\n  λ = {lam} pax/min | Headway = {HEADWAY_MIN} min | "
          f"Trains = {NUM_TRAINS} | Platform = {NUM_PLATFORMS}\n")
    env.run(until=SIM_DURATION_MIN)

    # ── Step 3: Results summary ───────────────────────────────────
    print("\n── SECTION 3: SIMULATION RESULTS ───────────────────────────")
    summary = metrics.summary()
    for k, v in summary.items():
        print(f"  {k:<35} {v}")

    # ── Step 4: Cumulative plot ───────────────────────────────────
    print("\n── SECTION 4: GENERATING CUMULATIVE DEPARTURE PLOT ─────────")
    plot_cumulative_departures(metrics, lam_per_min=lam,
                                save_path="return_departures.png")

    print("\n" + "="*65)
    print("  Return simulation complete.")
    print("="*65 + "\n")
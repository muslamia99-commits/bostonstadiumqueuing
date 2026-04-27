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
    Calculate the arrival rate λ needed to fill all trains.

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
    wait_times:           List[float] = field(default_factory=list)
    queue_at_departure:   List[int]   = field(default_factory=list)
    train_loads:          List[int]   = field(default_factory=list)
    departure_times:      List[float] = field(default_factory=list)
    arrival_times_ss:     List[float] = field(default_factory=list)
    platform_arrivals:    List[float] = field(default_factory=list)  # NEW: timestamp of each arrival
    passengers_boarded:   int         = 0
    passengers_stranded:  int         = 0

    def record_arrival(self, t: float):                              # NEW
        self.platform_arrivals.append(t)

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
            "last_train_arrives_SS_min": round(self.arrival_times_ss[-1], 1) if self.arrival_times_ss else None,
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
    Generates passengers with a time-varying rate to model the post-game surge:
      - t=0 to t=30 (pre-service): slow trickle as fans exit the stadium
      - t=30 (whistle + service start): large surge as crowd floods platform
      - t=30 onward: rate decays exponentially as crowd thins out
    """
    yield env.timeout(start_time)

    while True:
        t = env.now

        # Surge model: peak at service start, exponential decay afterward
        # Before service: slow trickle at 20% of base rate
        # After whistle: spike to 3x base rate, decaying with half-life of 20 min
        if t < WHISTLE_TO_SERVICE:
            effective_lam = lam_per_min * 0.20
        else:
            time_since_service = t - WHISTLE_TO_SERVICE
            effective_lam = lam_per_min * 3.0 * math.exp(-0.035 * time_since_service)
            effective_lam = max(effective_lam, lam_per_min * 0.05)  # floor: trickle never stops

        inter_arrival = random.expovariate(effective_lam)
        yield env.timeout(inter_arrival)
        env.process(passenger_process(env, platform, metrics))

def passenger_process(env: simpy.Environment,
                       platform: simpy.Resource,
                       metrics: ReturnMetrics):
    """
    A single returning passenger: arrives, joins FCFS queue, boards next train.
    """
    arrive_time = env.now
    metrics.record_arrival(arrive_time)   # NEW: log platform arrival timestamp
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
    arrive = env.now
    queue_before = len(platform.queue)
    print(f"  [{arrive:6.1f} min] {name} ready for boarding | Queue: {queue_before}")

    # Board passengers one by one up to train capacity during dwell
    boarded = 0
    boarding_events = []
    while boarded < TRAIN_CAPACITY and len(platform.queue) > 0:
        boarding_events.append(platform.queue[0].proc)
        boarded += 1

    yield env.timeout(DWELL_TIME_MIN)

    load = min(queue_before, TRAIN_CAPACITY)
    queue_remaining = len(platform.queue)
    metrics.record_departure(env.now, load, queue_remaining)

    print(f"  [{env.now:6.1f} min] {name} departs | Boarded: {load} "
          f"| Queue remaining: {queue_remaining} "
          f"| Arrives SS: {env.now + TRAVEL_TIME_MIN:.1f} min")
# ─────────────────────────────────────────────
# CUMULATIVE DEPARTURE PLOT
# ─────────────────────────────────────────────
def plot_cumulative_departures(metrics: ReturnMetrics,
                                lam_per_min: float,
                                save_path: str = "return_departures.png"):
    """
    Two lines:
      - Orange: cumulative passenger arrivals to Foxboro platform
      - Blue:   cumulative passengers that actually board and depart
    X-axis: minutes after final whistle
    Y-axis: cumulative passengers
    """
    # ── Arrival curve ─────────────────────────────────────────────
    arrival_times_sorted = np.sort(metrics.platform_arrivals)
    cum_arrivals = np.arange(1, len(arrival_times_sorted) + 1)

    # ── Boarding staircase ────────────────────────────────────────
    loads     = np.array(metrics.train_loads)
    dep_times = np.array(metrics.departure_times)
    cum_boarded = np.cumsum(loads)

    # Prepend zero so staircase starts at origin
    dep_times_plot  = np.concatenate([[0], dep_times])
    cum_boarded_plot = np.concatenate([[0], cum_boarded])

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.7, linestyle="--")

    # Orange: arrivals to platform
    ax.plot(arrival_times_sorted, cum_arrivals,
            color="#f78166", linewidth=2.0, label="Cumulative arrivals to platform")

    # Blue: passengers that board each train (staircase steps up by load per train)
    ax.step(dep_times_plot, cum_boarded_plot, where="post",
            color="#58a6ff", linewidth=2.5, label="Cumulative passengers boarded")

    # Annotate each step with train number and load
    for i, (t, c, load) in enumerate(zip(dep_times, cum_boarded, loads)):
        ax.annotate(f"T{i+1}: {load} pax",
                    xy=(t, c), xytext=(t + 0.5, c + 15),
                    fontsize=7, color="#8b949e",
                    arrowprops=dict(arrowstyle="-", color="#30363d", lw=0.6))

    # Vertical markers
    ax.axvline(0, color="#3fb950", linewidth=1.0, linestyle=":",
               label="Final whistle (t=0)")
    ax.axvline(WHISTLE_TO_SERVICE, color="#e3b341", linewidth=1.0, linestyle=":",
               label=f"First train (+{WHISTLE_TO_SERVICE} min)")

    ax.set_xlabel("Minutes after final whistle", color="white")
    ax.set_ylabel("Cumulative passengers", color="white")
    ax.set_title("MBTA Foxboro Return Service — 2026 FIFA World Cup\n"
                 "Platform Arrivals vs. Passengers Boarded",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  Plot saved → {save_path}")
    import os
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"  File exists after save: {os.path.exists(save_path)}")
print(f"  File size: {os.path.getsize(save_path)} bytes")
print(f"  Saved to: {os.path.abspath(save_path)}")
plt.show()
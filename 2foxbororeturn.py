"""
MBTA Foxboro → South Station Return Simulation
===============================================
Models the post-game departure from Foxboro after a World Cup match.

Key assumptions:
  - Service begins 30 minutes after the final whistle
  - 1 platform available at Foxboro for boarding
  - 14 trains total dispatched (FCFS — no boarding groups)
  - Train capacity: 1,440 passengers
  - Dwell time: 5 minutes (deterministic)
  - Travel time back to South Station: 55 minutes
  - Headway: 15 minutes between trains

Three arrival scenarios are modeled:
  1. Normal distribution peak at t=30 (first train, early surge)
  2. Normal distribution peak at t=120 (around train 7, late surge)
  3. Uniform arrival rate across the entire 14-train service window
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import norm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

TRAIN_CAPACITY     = 1440     # passengers per train
NUM_TRAINS         = 14       # total return trains
FIRST_DEPARTURE    = 30       # minutes after whistle for train 1
HEADWAY_MIN        = 15       # minutes between trains
TRAVEL_TIME_MIN    = 55       # Foxboro → South Station

TOTAL_PASSENGERS   = NUM_TRAINS * TRAIN_CAPACITY   # 20,160

# Derived timing
LAST_DEPARTURE     = FIRST_DEPARTURE + (NUM_TRAINS - 1) * HEADWAY_MIN   # 225 min
T_MAX              = LAST_DEPARTURE + 20                                  # plot window

# Fine time axis for smooth curves
T = np.linspace(0, T_MAX, 2000)

# Train departure times
TRAIN_DEPARTS = [FIRST_DEPARTURE + i * HEADWAY_MIN for i in range(NUM_TRAINS)]


# ─────────────────────────────────────────────
# ARRIVAL DENSITY FUNCTIONS
# ─────────────────────────────────────────────

def arrival_normal(t, mu, sigma):
    """Normal (Gaussian) density, evaluated at times t."""
    return norm.pdf(t, loc=mu, scale=sigma)

def arrival_uniform(t, t_start=0, t_end=None):
    """Uniform density from t_start to t_end."""
    if t_end is None:
        t_end = LAST_DEPARTURE
    inside = (t >= t_start) & (t <= t_end)
    density = np.zeros_like(t, dtype=float)
    density[inside] = 1.0 / (t_end - t_start)
    return density


from scipy.integrate import cumulative_trapezoid

# ... (other code) ...

def cumulative_arrivals(density, t, total=TOTAL_PASSENGERS):
    """
    Integrate density over t, then scale so A(T_MAX) = total passengers.
    Returns a cumulative array aligned with t.
    """
    # Use cumulative_trapezoid to get the running integral
    # initial=0 ensures the output array is the same length as t
    running = cumulative_trapezoid(density, t, initial=0)
    
    # Scale the curve so the final value matches your total passenger count
    if running[-1] > 0:
        return (running / running[-1]) * total
    return running


# ─────────────────────────────────────────────
# CUMULATIVE DEPARTURE CURVE  D(t)
# ─────────────────────────────────────────────

def cumulative_departures(cum_arr, t, train_departs=TRAIN_DEPARTS,
                           capacity=TRAIN_CAPACITY):
    """
    Step-function departure curve: each train boards min(queue, capacity) passengers.
    Queue at train i departure = cum_arr(t_depart_i) - passengers already boarded.
    """
    dep = np.zeros(len(t))
    total_boarded = 0
    schedule = []   # (departure_time, cumulative_boarded_after_this_train)

    for d_time in train_departs:
        # How many passengers have arrived by this train's departure?
        idx = np.searchsorted(t, d_time)
        arrived_so_far = cum_arr[min(idx, len(cum_arr) - 1)]
        queue = arrived_so_far - total_boarded
        this_train = min(queue, capacity)
        total_boarded += this_train
        schedule.append((d_time, total_boarded))

    # Build step curve
    for i, tv in enumerate(t):
        boarded = 0
        for (d_time, cum_b) in schedule:
            if tv >= d_time:
                boarded = cum_b
        dep[i] = boarded

    return dep, schedule


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

COLORS = {
    "arrival":   "#1a6bbd",   # blue
    "departure": "#d85a30",   # coral/orange
    "fill":      "#1a6bbd",
    "train":     "#aaaaaa",   # gray dashed verticals
}

def plot_scenario(ax, cum_arr, cum_dep, title, subtitle):
    """
    Draw A(t) and D(t) on ax, with vertical dashed lines for each train.
    """
    # Shade gap between A(t) and D(t)  →  represents queue / total wait
    ax.fill_between(T, cum_arr, cum_dep,
                     where=(cum_arr >= cum_dep),
                     color=COLORS["arrival"], alpha=0.08, label="_nolegend_")

    # A(t) — cumulative arrivals
    ax.plot(T, cum_arr, color=COLORS["arrival"], lw=2, label="A(t)  cumulative arrivals")

    # D(t) — cumulative departures (step)
    ax.step(T, cum_dep, where="pre", color=COLORS["departure"], lw=2.5,
            label="D(t)  cumulative departures")

    # Vertical lines for each train departure
    for i, td in enumerate(TRAIN_DEPARTS):
        ax.axvline(td, color=COLORS["train"], lw=0.8, ls="--", alpha=0.6)
        # Label trains 1, 3, 5, 7, 9, 11, 13, 14 to avoid clutter
        if i % 2 == 0 or i == NUM_TRAINS - 1:
            ax.text(td + 0.5, ax.get_ylim()[1] * 0.97,
                    f"T{i+1}", fontsize=7, color="#888888",
                    ha="left", va="top")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Minutes after final whistle", fontsize=9)
    ax.set_ylabel("Cumulative passengers", fontsize=9)
    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, TOTAL_PASSENGERS * 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.4)
    ax.text(0.99, 0.04, subtitle, transform=ax.transAxes,
            fontsize=8, color="#666666", ha="right", va="bottom",
            style="italic")


def plot_all_scenarios():
    # ── Scenario 1: Normal peak at t=30 (σ=22 min) ──
    d1 = arrival_normal(T, mu=FIRST_DEPARTURE, sigma=22)
    ca1 = cumulative_arrivals(d1, T)
    cd1, _ = cumulative_departures(ca1, T)

    # ── Scenario 2: Normal peak at train 7 departure (t=120, σ=28 min) ──
    train7_time = FIRST_DEPARTURE + 6 * HEADWAY_MIN   # 120 min
    d2 = arrival_normal(T, mu=train7_time, sigma=28)
    ca2 = cumulative_arrivals(d2, T)
    cd2, _ = cumulative_departures(ca2, T)

    # ── Scenario 3: Uniform arrivals across full service window ──
    d3 = arrival_uniform(T, t_start=0, t_end=LAST_DEPARTURE)
    ca3 = cumulative_arrivals(d3, T)
    cd3, _ = cumulative_departures(ca3, T)

    # ── Figure ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), facecolor="#ffffff")
    fig.suptitle(
        "MBTA Foxboro → South Station  |  Post-Game Cumulative Arrival & Departure Diagrams\n"
        f"14 trains  ·  {TRAIN_CAPACITY:,} pax/train  ·  {TOTAL_PASSENGERS:,} total  "
        f"·  First departure t=30 min  ·  15-min headway",
        fontsize=11, fontweight="bold", y=1.005
    )

    plot_scenario(
        axes[0], ca1, cd1,
        title="Scenario 1 — Early surge: Normal peak at t = 30 min (first departure)",
        subtitle="Most fans rush to the platform immediately; queue clears quickly"
    )
    plot_scenario(
        axes[1], ca2, cd2,
        title=f"Scenario 2 — Late surge: Normal peak at t = {train7_time} min (train 7 departure)",
        subtitle="Fans linger in stadium; large backlog builds then clears after midpoint"
    )
    plot_scenario(
        axes[2], ca3, cd3,
        title="Scenario 3 — Uniform arrivals across full 14-train service window",
        subtitle="Steady trickle throughout; departure curve tracks A(t) closely but with constant lag"
    )

    plt.tight_layout()
    plt.savefig("foxboro_cumulative_diagrams.png", dpi=150, bbox_inches="tight")
    print("Saved: foxboro_cumulative_diagrams.png")
    plt.show()


if __name__ == "__main__":
    plot_all_scenarios()
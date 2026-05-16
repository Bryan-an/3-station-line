"""Discrete event simulation of a 3-station serial assembly line.

State machine for each station:
  E1: BUSY | BLOCKED          (never FREE — input is infinite)
  E2: BUSY | BLOCKED | FREE
  E3: BUSY | FREE             (never BLOCKED — output is infinite)

Buffer semantics: B1 (between E1 and E2) holds pieces waiting; the piece
inside E2 is NOT counted in B1. Same for B2 / E3.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

from src.rng import LCG, exponential


# Event types
FIN_E1 = "FIN_E1"
FIN_E2 = "FIN_E2"
FIN_E3 = "FIN_E3"

# Priority for simultaneous events: downstream first (frees space upstream)
PRIORITY = {FIN_E3: 0, FIN_E2: 1, FIN_E1: 2}


@dataclass(frozen=True)
class SimulationConfig:
    k1: int
    k2: int
    rate: float = 0.25        # λ (pieces/minute)
    duration: float = 480.0   # minutes
    seed: int = 12_345


@dataclass
class SimulationResult:
    throughput: float                       # pieces / hour
    flow_times: list[float]                 # one per completed piece
    avg_wip: float                          # time-weighted (B1 + B2)
    pct_blocked_e1: float
    pct_blocked_e2: float
    pct_starved_e2: float
    pct_starved_e3: float
    busy_e1: float
    busy_e2: float
    busy_e3: float
    pieces_completed: int
    wip_trace: list[tuple[float, int]]      # (t, B1+B2) after each event
    event_log: list[dict]                   # per-event snapshot


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run one replication. Pure: same config → same result."""
    rng = LCG(state=config.seed)

    def sample_duration() -> tuple[float, float]:
        """Returns (u_used, duration)."""
        u = rng.next_uniform()
        while u == 0.0:
            u = rng.next_uniform()
        return u, -math.log(u) / config.rate

    # --- State ---
    t = 0.0
    fel: list[tuple[float, int, int, str, int]] = []
    tiebreaker = 0  # ensures stable order for equal (t, priority)

    e1_state, e1_piece = "BUSY", None
    e2_state, e2_piece = "FREE", None
    e3_state, e3_piece = "FREE", None
    b1: deque[int] = deque()
    b2: deque[int] = deque()

    next_piece_id = 1
    pieces_completed = 0

    t_busy_e1 = t_busy_e2 = t_busy_e3 = 0.0
    t_blocked_e1 = t_blocked_e2 = 0.0
    t_starved_e2 = t_starved_e3 = 0.0
    area_wip = 0.0

    piece_entry: dict[int, float] = {}
    piece_exit: dict[int, float] = {}

    wip_trace: list[tuple[float, int]] = [(0.0, 0)]
    event_log: list[dict] = []

    def push_event(when: float, event_type: str, piece: int) -> None:
        nonlocal tiebreaker
        heapq.heappush(
            fel, (when, PRIORITY[event_type], tiebreaker, event_type, piece)
        )
        tiebreaker += 1

    def accumulate_until(t_next: float) -> None:
        """Add (t_next - t) to the relevant accumulators based on current state."""
        nonlocal area_wip, t_busy_e1, t_busy_e2, t_busy_e3
        nonlocal t_blocked_e1, t_blocked_e2, t_starved_e2, t_starved_e3
        delta = t_next - t
        if delta <= 0.0:
            return
        if e1_state == "BUSY":
            t_busy_e1 += delta
        elif e1_state == "BLOCKED":
            t_blocked_e1 += delta
        if e2_state == "BUSY":
            t_busy_e2 += delta
        elif e2_state == "BLOCKED":
            t_blocked_e2 += delta
        elif e2_state == "FREE":
            t_starved_e2 += delta
        if e3_state == "BUSY":
            t_busy_e3 += delta
        elif e3_state == "FREE":
            t_starved_e3 += delta
        area_wip += (len(b1) + len(b2)) * delta

    def snapshot(event_type: str, piece_id, u_used, duration_used, note: str) -> None:
        event_log.append({
            "t": t,
            "event_type": event_type,
            "piece_id": piece_id,
            "u": u_used,
            "duration": duration_used,
            "e1_state": e1_state,
            "e2_state": e2_state,
            "e3_state": e3_state,
            "e1_piece": e1_piece,
            "e2_piece": e2_piece,
            "e3_piece": e3_piece,
            "b1_count": len(b1),
            "b2_count": len(b2),
            "wip": len(b1) + len(b2),
            "pieces_completed": pieces_completed,
            "next_piece_id": next_piece_id,
            "note": note,
        })

    # --- Initialize: E1 takes piece 1 immediately ---
    e1_piece = next_piece_id
    piece_entry[next_piece_id] = 0.0
    next_piece_id += 1
    u0, d0 = sample_duration()
    push_event(d0, FIN_E1, e1_piece)
    snapshot("INIT", e1_piece, u0, d0, "E1 starts piece 1")

    # --- Main loop ---
    while fel:
        t_next, _prio, _tb, event_type, piece = heapq.heappop(fel)
        if t_next >= config.duration:
            accumulate_until(config.duration)
            t = config.duration
            break

        accumulate_until(t_next)
        t = t_next

        if event_type == FIN_E1:
            if len(b1) < config.k1:
                b1.append(e1_piece)
                e1_piece = next_piece_id
                piece_entry[next_piece_id] = t
                next_piece_id += 1
                u, d = sample_duration()
                push_event(t + d, FIN_E1, e1_piece)
                # E1 remains BUSY
                snapshot(FIN_E1, piece, u, d, "E1 → B1; started next piece")
                if e2_state == "FREE":
                    e2_piece = b1.popleft()
                    e2_state = "BUSY"
                    u2, d2 = sample_duration()
                    push_event(t + d2, FIN_E2, e2_piece)
                    snapshot(FIN_E1, piece, u2, d2, "E2 took from B1")
            else:
                e1_state = "BLOCKED"
                snapshot(FIN_E1, piece, None, None, "B1 full → E1 BLOCKED")

        elif event_type == FIN_E2:
            if len(b2) < config.k2:
                b2.append(e2_piece)
                # Pre-pop B1 for E2 (without starting E2 yet) — this frees a slot
                # before any E1-unblock push, preventing transient B1 overflow.
                if len(b1) > 0:
                    next_for_e2 = b1.popleft()
                else:
                    next_for_e2 = None
                # E1 unblock cascade: B1 now has space (was full only if E1 was BLOCKED)
                if e1_state == "BLOCKED":
                    b1.append(e1_piece)
                    e1_piece = next_piece_id
                    piece_entry[next_piece_id] = t
                    next_piece_id += 1
                    u_e1, d_e1 = sample_duration()
                    push_event(t + d_e1, FIN_E1, e1_piece)
                    e1_state = "BUSY"
                    snapshot(FIN_E2, piece, u_e1, d_e1, "E1 unblocked")
                # Now actually start E2 with its next piece (if any)
                if next_for_e2 is not None:
                    e2_piece = next_for_e2
                    u_e2, d_e2 = sample_duration()
                    push_event(t + d_e2, FIN_E2, e2_piece)
                    e2_state = "BUSY"
                    snapshot(FIN_E2, piece, u_e2, d_e2, "E2 took next from B1")
                else:
                    e2_state = "FREE"
                    e2_piece = None
                    snapshot(FIN_E2, piece, None, None, "E2 FREE (starved)")
                if e3_state == "FREE":
                    e3_piece = b2.popleft()
                    e3_state = "BUSY"
                    u_e3, d_e3 = sample_duration()
                    push_event(t + d_e3, FIN_E3, e3_piece)
                    snapshot(FIN_E2, piece, u_e3, d_e3, "E3 took from B2")
            else:
                e2_state = "BLOCKED"
                snapshot(FIN_E2, piece, None, None, "B2 full → E2 BLOCKED")

        elif event_type == FIN_E3:
            piece_exit[e3_piece] = t
            pieces_completed += 1
            completed_piece = e3_piece
            e3_piece = None
            e3_state = "FREE"
            # Cascade: E2 was blocked
            if e2_state == "BLOCKED":
                # E2 blocked because b2 was full. Pre-pop b2 (for E3 to take), then push e2_piece.
                next_for_e3 = b2.popleft()
                b2.append(e2_piece)
                if len(b1) > 0:
                    next_for_e2 = b1.popleft()
                    u_e2, d_e2 = sample_duration()        # u_e2 FIRST (preserves original order)
                    # E1 cascade
                    if e1_state == "BLOCKED":
                        b1.append(e1_piece)
                        e1_piece = next_piece_id
                        piece_entry[next_piece_id] = t
                        next_piece_id += 1
                        u_e1, d_e1 = sample_duration()    # u_e1 SECOND
                        push_event(t + d_e1, FIN_E1, e1_piece)
                        e1_state = "BUSY"
                        snapshot(FIN_E3, completed_piece, u_e1, d_e1, "E1 unblocked (cascade)")
                    # Now apply E2 with sampled u_e2
                    e2_piece = next_for_e2
                    push_event(t + d_e2, FIN_E2, e2_piece)
                    e2_state = "BUSY"
                    snapshot(FIN_E3, completed_piece, u_e2, d_e2, "E2 unblocked, took from B1")
                else:
                    e2_state = "FREE"
                    e2_piece = None
                    snapshot(FIN_E3, completed_piece, None, None, "E2 unblocked but B1 empty")
                # Apply E3 with the pre-popped piece
                e3_piece = next_for_e3
                e3_state = "BUSY"
                u_e3, d_e3 = sample_duration()             # u_e3 THIRD (preserves original order)
                push_event(t + d_e3, FIN_E3, e3_piece)
                snapshot(FIN_E3, completed_piece, u_e3, d_e3, "E3 took next from B2")
            else:
                # E2 was not blocked. E3 just checks b2.
                if len(b2) > 0:
                    e3_piece = b2.popleft()
                    e3_state = "BUSY"
                    u_e3, d_e3 = sample_duration()
                    push_event(t + d_e3, FIN_E3, e3_piece)
                    snapshot(FIN_E3, completed_piece, u_e3, d_e3, "E3 took next from B2")
                else:
                    snapshot(FIN_E3, completed_piece, None, None, "E3 FREE (starved)")

        wip_trace.append((t, len(b1) + len(b2)))

    # --- Final stats ---
    duration = config.duration
    flow_times = [piece_exit[pid] - piece_entry[pid] for pid in piece_exit]
    throughput = pieces_completed / (duration / 60.0)
    avg_wip = area_wip / duration

    return SimulationResult(
        throughput=throughput,
        flow_times=flow_times,
        avg_wip=avg_wip,
        pct_blocked_e1=t_blocked_e1 / duration,
        pct_blocked_e2=t_blocked_e2 / duration,
        pct_starved_e2=t_starved_e2 / duration,
        pct_starved_e3=t_starved_e3 / duration,
        busy_e1=t_busy_e1 / duration,
        busy_e2=t_busy_e2 / duration,
        busy_e3=t_busy_e3 / duration,
        pieces_completed=pieces_completed,
        wip_trace=wip_trace,
        event_log=event_log,
    )

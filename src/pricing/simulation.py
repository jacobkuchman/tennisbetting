from __future__ import annotations

import random


def simulate_set(hold_p1: float, hold_p2: float) -> tuple[int, int]:
    g1 = g2 = 0
    server = 1
    while True:
        if server == 1:
            g1 += 1 if random.random() < hold_p1 else 0
            g2 += 0 if random.random() < hold_p1 else 1
            server = 2
        else:
            g2 += 1 if random.random() < hold_p2 else 0
            g1 += 0 if random.random() < hold_p2 else 1
            server = 1

        if (g1 >= 6 or g2 >= 6) and abs(g1 - g2) >= 2:
            return g1, g2
        if g1 == 7 or g2 == 7:
            return g1, g2


def simulate_match_markets(hold_p1: float, hold_p2: float, best_of: int = 3, n_sims: int = 5000) -> dict:
    p1_match_wins = 0
    p1_set1_wins = 0
    margin_samples = []
    totals_samples = []
    score_counts = {}

    for _ in range(n_sims):
        s1 = s2 = 0
        total_games = 0
        first_set_done = False

        while s1 < (best_of // 2 + 1) and s2 < (best_of // 2 + 1):
            g1, g2 = simulate_set(hold_p1, hold_p2)
            total_games += g1 + g2
            if not first_set_done:
                p1_set1_wins += 1 if g1 > g2 else 0
                first_set_done = True
            if g1 > g2:
                s1 += 1
            else:
                s2 += 1

        p1_match_wins += 1 if s1 > s2 else 0
        margin_samples.append((s1 - s2) * 6)
        totals_samples.append(total_games)
        score_counts[(s1, s2)] = score_counts.get((s1, s2), 0) + 1

    return {
        "p1_match_win": p1_match_wins / n_sims,
        "p1_set1_win": p1_set1_wins / n_sims,
        "exp_game_margin": sum(margin_samples) / len(margin_samples),
        "exp_total_games": sum(totals_samples) / len(totals_samples),
        "correct_score_probs": {k: v / n_sims for k, v in score_counts.items()},
    }

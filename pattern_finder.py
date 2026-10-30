import numpy as np
import random
import time
from itertools import combinations

# ---------- SETTINGS ----------
N = 198
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

TARGET_B_SHIFT = 66      # brightest cue: +120 deg
TARGET_C_SHIFT = 132     # pattern cue: +240 deg, UI/intuitive convention

A_BRIGHT_LED = 0
A_DIM_LEVEL = 0.64
B_COPY_LEVEL = 0.64
LEVELS_EXTRA = [0.64]

MAX_A_DIM_LEDS = 4
MAX_EXTRA_B_LEDS = 5

MAX_VECTOR_ERROR_DEG = 10.0
MAX_OVERLAP_ERROR_LEDS = 4
MAX_A_SECOND_BEST_OVERLAP = 0.75

PRINT_EVERY_A = 100
RANDOM_B_TRIALS_PER_A = 50000

random.seed(1)
np.random.seed(1)

# Aesthetic Pattern A candidate LEDs: 30-degree grid
A_CANDIDATE_LEDS = sorted(set([
    17,   # 30.91 deg
    33,   # 60.00 deg
    50,   # 90.91 deg
    66,   # 120.00 deg
    83,   # 150.91 deg
    99,   # 180.00 deg
    116,  # 210.91 deg
    132,  # 240.00 deg
    149,  # 270.91 deg
    165,  # 300.00 deg
    182,  # 330.91 deg
]))

B_COMP_CANDIDATE_LEDS = list(range(N))


# ---------- FUNCTIONS ----------
def circdiff(a, b):
    return np.angle(np.exp(1j * (a - b)))


def vec_angle(P):
    z = np.sum(P * np.exp(1j * theta))
    if abs(z) < 1e-12:
        return None
    return np.angle(z)


def brightest_index(P):
    return int(np.argmax(P))


def zcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def circular_led_diff(a, b):
    d = (a - b) % N
    return min(d, N - d)


def full_overlap_shift(A, B):
    """
    UI/intuitive convention:
    returns k such that rotating Pattern A by k best matches Pattern B.

    In words:
        "How much must A rotate to look like B?"
    """
    corrs = np.array([zcorr(B, np.roll(A, k)) for k in range(N)])
    k = int(np.argmax(corrs))
    return k, float(corrs[k])


def local_overlap_shift(A, B, target_shift, window=4):
    """
    Same convention as full_overlap_shift:
    rotate A by k and compare to B.
    """
    vals = []
    for k in range(target_shift - window, target_shift + window + 1):
        kk = k % N
        vals.append((kk, zcorr(B, np.roll(A, kk))))
    return max(vals, key=lambda x: x[1])


def print_pattern(name, P):
    print(f"\n{name}:")
    for i in np.where(P > 0)[0]:
        print(f"LED {i:3d} | angle {i * 360 / N:7.2f} deg | intensity {P[i]:.2f}")


# ---------- GENERATE PATTERN A CANDIDATES ----------
A_candidates = []

for n_dim in range(2, MAX_A_DIM_LEDS + 1):
    for dims in combinations(A_CANDIDATE_LEDS, n_dim):

        if A_BRIGHT_LED in dims:
            continue

        A = np.zeros(N)
        A[A_BRIGHT_LED] = 1.0

        for d in dims:
            A[d] = A_DIM_LEVEL

        if np.sum(A == A.max()) != 1:
            continue

        VA = vec_angle(A)
        if VA is None:
            continue

        # Self-overlap ambiguity: rotate A onto itself.
        all_corrs = np.array([zcorr(A, np.roll(A, k)) for k in range(N)])
        all_corrs[0] = -999
        second_best = float(np.max(all_corrs))

        if second_best > MAX_A_SECOND_BEST_OVERLAP:
            continue

        A_candidates.append((second_best, A.copy()))

A_candidates = sorted(A_candidates, key=lambda x: x[0])

print(f"Generated {len(A_candidates)} Pattern A candidates.")
print("Searching joint A/B space...\n")


# ---------- JOINT SEARCH ----------
best = []
near = []
start_time = time.time()

for a_idx, (A_second_best, A) in enumerate(A_candidates, start=1):

    VA = vec_angle(A)
    BA = brightest_index(A)

    target_B_index = (BA + TARGET_B_SHIFT) % N

    B_seed = np.zeros(N)

    # Rotated dim copy of Pattern A at intended pattern-match shift.
    # Since overlap convention is now "rotate A to B", this directly matches TARGET_C_SHIFT.
    for i in np.where(A > 0)[0]:
        j = (i + TARGET_C_SHIFT) % N
        B_seed[j] = B_COPY_LEVEL

    # Unique brightest cue at +120 deg
    B_seed[target_B_index] = 1.0

    # Keep copied LEDs sub-brightest
    for i in np.where(B_seed > 0)[0]:
        if i != target_B_index:
            B_seed[i] = min(B_seed[i], B_COPY_LEVEL)

    used = set(np.where(B_seed > 0)[0])
    comp_leds = [i for i in B_COMP_CANDIDATE_LEDS if i not in used]

    for _ in range(RANDOM_B_TRIALS_PER_A):

        n_extra = random.randint(0, MAX_EXTRA_B_LEDS)
        extra = random.sample(comp_leds, n_extra)
        vals = [random.choice(LEVELS_EXTRA) for _ in range(n_extra)]

        B = B_seed.copy()

        for i, v in zip(extra, vals):
            B[i] = v

        if np.sum(B == B.max()) != 1:
            continue

        if brightest_index(B) != target_B_index:
            continue

        VB = vec_angle(B)
        if VB is None:
            continue

        dV_deg = np.rad2deg(abs(circdiff(VB, VA)))

        if dV_deg > MAX_VECTOR_ERROR_DEG:
            continue

        local_shift, local_overlap = local_overlap_shift(
            A,
            B,
            TARGET_C_SHIFT,
            window=MAX_OVERLAP_ERROR_LEDS,
        )

        local_error = circular_led_diff(local_shift, TARGET_C_SHIFT)

        if local_error > MAX_OVERLAP_ERROR_LEDS:
            continue

        full_shift, full_overlap = full_overlap_shift(A, B)
        full_error = circular_led_diff(full_shift, TARGET_C_SHIFT)

        A_on = int(np.sum(A > 0))
        B_on = int(np.sum(B > 0))
        total_on = A_on + B_on

        if full_error > MAX_OVERLAP_ERROR_LEDS:

            near_score = (
                10 * dV_deg
                + 10 * full_error
                + 20 * total_on
                - 10 * full_overlap
                + 10 * A_second_best
            )

            near.append((
                near_score,
                total_on,
                A_on,
                B_on,
                dV_deg,
                full_shift,
                full_error,
                full_overlap,
                A_second_best,
                A.copy(),
                B.copy(),
            ))

            near = sorted(near, key=lambda x: x[0])[:20]

            continue

        score = (
            100 * total_on
            + 30 * B_on
            + 20 * dV_deg
            + 20 * full_error
            - 10 * full_overlap
            + 50 * A_second_best
        )

        best.append((
            score,
            total_on,
            A_on,
            B_on,
            dV_deg,
            full_shift,
            full_overlap,
            A_second_best,
            A.copy(),
            B.copy(),
        ))

        print(
            f"ACCEPTED | A={a_idx:,}/{len(A_candidates):,} "
            f"| total={total_on} A={A_on} B={B_on} "
            f"| dV={dV_deg:.2f} deg "
            f"| shift={full_shift} "
            f"| shift_deg={full_shift * 360 / N:.2f} "
            f"| overlap={full_overlap:.3f} "
            f"| A_ambiguity={A_second_best:.3f} "
            f"| score={score:.2f}"
        )

    if a_idx % PRINT_EVERY_A == 0:
        elapsed = time.time() - start_time
        print(
            f"PROGRESS | A={a_idx:,}/{len(A_candidates):,} "
            f"| elapsed={elapsed:.1f}s "
            f"| accepted={len(best)} "
            f"| near={len(near)}"
        )


# ---------- FINAL ----------
print("\nJoint search finished.")

if len(best) == 0:

    print("\nNo accepted joint A/B solution found.")

    if len(near) > 0:

        print("\nBest near misses:")

        for item in near[:5]:

            (
                near_score,
                total_on,
                A_on,
                B_on,
                dV_deg,
                full_shift,
                full_error,
                full_overlap,
                A_second_best,
                A_near,
                B_near,
            ) = item

            print("\n--- Near miss ---")
            print(f"Score: {near_score:.2f}")
            print(f"Total LEDs: {total_on} | A: {A_on} | B: {B_on}")
            print(f"Vector error: {dV_deg:.2f} deg")
            print(f"Pattern shift: {full_shift} LEDs = {full_shift * 360 / N:.2f} deg")
            print(f"Pattern error: {full_error} LEDs = {full_error * 360 / N:.2f} deg")
            print(f"Overlap: {full_overlap:.3f}")
            print(f"A ambiguity: {A_second_best:.3f}")

            print_pattern("Near Pattern A", A_near)
            print_pattern("Near Pattern B", B_near)

    raise SystemExit


best = sorted(
    best,
    key=lambda x: (
        x[1],      # total LEDs
        x[3],      # Pattern B LEDs
        x[7],      # Pattern A ambiguity
        x[4],      # vector error
        -x[6],     # overlap strength
    ),
)

(
    score,
    total_on,
    A_on,
    B_on,
    dV_deg,
    full_shift,
    full_overlap,
    A_second_best,
    A,
    B,
) = best[0]

VA = vec_angle(A)
VB = vec_angle(B)
BA = brightest_index(A)
BB = brightest_index(B)

brightest_delta_leds = (BB - BA) % N
vector_delta_signed = np.rad2deg(circdiff(VB, VA))
vector_delta_display = vector_delta_signed % 360
pattern_delta_deg = full_shift * 360 / N

print("\nBest joint solution:")
print(f"Score: {score:.2f}")
print(f"Total LEDs on: {total_on}")
print(f"Pattern A LEDs on: {A_on}")
print(f"Pattern B LEDs on: {B_on}")
print(f"Vector error from 0 deg: {dV_deg:.2f}")
print(f"Brightest delta: {brightest_delta_leds} LEDs = {brightest_delta_leds * 360 / N:.2f} deg")
print(f"Pattern overlap shift: {full_shift} LEDs = {pattern_delta_deg:.2f} deg")
print(f"Pattern overlap strength: {full_overlap:.3f}")
print(f"Pattern A second-best self-overlap: {A_second_best:.3f}")

print_pattern("Pattern A", A)
print_pattern("Pattern B", B)

print("\nPredicted deltas:")
print(f"Brightest cue: {brightest_delta_leds} LEDs = {brightest_delta_leds * 360 / N:.2f} deg")
print(f"Vector sum:    {vector_delta_display:.2f} deg")
print(f"Pattern match: {full_shift} LEDs = {pattern_delta_deg:.2f} deg")
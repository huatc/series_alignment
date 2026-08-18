#!/usr/bin/env python
"""Measure the noise floor of the SAXS objective by repeating one evaluation.

The optimisation only varied by ~3.6% across 22 evaluations, so before that
spread can be read as signal we need to know how much the score moves when
NOTHING changes. There are two independent noise sources:

  * the MD itself -- HOOMD's Langevin integrator is seeded (seed=42), so a
    repeated run with identical parameters should be bit-identical;
  * the SAXS forward model -- the Monte Carlo scattering draws SAXS_MC_SAMPLES
    (1e7) samples per frame with no seed, so I(q) differs between repeats even
    for the very same configuration.

Two modes separate them:

  analysis  (default, cheap, no HOOMD)
      Re-run the SAXS forward model + scoring N times on ONE existing
      trajectory. Isolates the Monte Carlo noise.

  full      (expensive, needs HOOMD)
      Re-run objective() N times with identical parameters. Includes the MD.

Examples
--------
    python repeatability_saxs.py --gsd path/to/DNA_assembly_*.gsd --repeats 5
    HOOMD_MODE=--mode=cpu python repeatability_saxs.py --mode full \
        --params 0.003 150 2.25 12 8 --repeats 3
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Bridge_20_Hyak_250803 as B     # noqa: E402


def section(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def score_saxs_only(exp_saxs, sim_saxs):
    """SAXS-only aligned cost per curve, plus the monotone path."""
    cost = B.build_cost_matrix(exp_saxs, sim_saxs, B.shape_distance_saxs,
                               label='saxs', verbose=False)
    idx, total = B.series_alignment.align_monotone_min(cost)
    return total / len(exp_saxs), [int(v) for v in idx], cost


def repeat_analysis(gsd_path, repeats, workdir, n_samples=None):
    """Re-run the SAXS forward model on one fixed trajectory."""
    exp_saxs = B.load_experimental_saxs()
    positions, lattice, box_L = B.read_trajectory(gsd_path)
    frames = B.select_frame_indices(len(positions), B.N_SERIES_FRAMES)
    print(f"trajectory : {os.path.basename(gsd_path)}")
    print(f"             {len(positions)} frames, scoring {len(frames)}")
    print(f"experiment : {len(exp_saxs)} SAXS curves")
    print(f"MC samples : {n_samples or B.SAXS_MC_SAMPLES:,} per frame\n")

    scores, paths, curves = [], [], []
    for r in range(repeats):
        out = os.path.join(workdir, f'repeat_{r}')
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        kw = {} if n_samples is None else {'n_samples': n_samples}
        sim = B.convert_to_SAXS_series(lattice, frames, out, **kw)
        s, path, _ = score_saxs_only(exp_saxs, sim)
        scores.append(s)
        paths.append(path)
        curves.append([c.copy() for c in sim])
        print(f"  repeat {r + 1}/{repeats}: score = {s:.6f}   "
              f"({time.time() - t0:.0f}s)")
    return np.array(scores), paths, curves


def repeat_full(params, repeats, workdir):
    """Re-run the whole objective, MD included."""
    exp_saxs = B.load_experimental_saxs()
    exp_dls = B.load_experimental_dls()
    names = ('Density', 'U_0', 'r0', 'm', 'gap')
    print(f"parameters : {dict(zip(names, params))}\n")

    scores = []
    for r in range(repeats):
        B.SCORE_PARTS.clear()
        t0 = time.time()
        B.objective(os.path.join(workdir, f'repeat_{r}'), 0, 0, names, *params,
                    exp_saxs=exp_saxs, exp_dls=exp_dls)
        part = B.SCORE_PARTS[-1] if B.SCORE_PARTS else {}
        s = part.get('saxs_score', np.nan)
        scores.append(s)
        print(f"  repeat {r + 1}/{repeats}: saxs_score = {s:.6f}   "
              f"({time.time() - t0:.0f}s)")
    return np.array(scores), [], []


def report(scores, paths, curves, signal_spread):
    section('score reproducibility')
    finite = scores[np.isfinite(scores)]
    if len(finite) < 2:
        print("  fewer than two successful repeats; nothing to compare")
        return

    sd, rng = finite.std(ddof=1), finite.max() - finite.min()
    print(f"  scores : {'  '.join(f'{v:.6f}' for v in finite)}")
    print(f"  mean   : {finite.mean():.6f}")
    print(f"  std    : {sd:.6f}   ({sd / finite.mean() * 100:.2f}% of the mean)")
    print(f"  range  : {rng:.6f}   ({rng / finite.mean() * 100:.2f}%)")

    section('noise floor vs the spread seen during optimisation')
    print(f"  repeatability spread (this test) : {rng:.6f}")
    print(f"  spread across BO evaluations     : {signal_spread:.6f}")
    if rng > 0:
        ratio = signal_spread / rng
        print(f"  signal-to-noise                  : {ratio:.1f}x")
        if ratio < 1.5:
            print("\n  WARNING: the objective moves about as much when nothing")
            print("  changes as it does across different parameters. The")
            print("  optimisation cannot be distinguished from fitting noise.")
        elif ratio < 3:
            print("\n  MARGINAL: real variation exists but is only a few times")
            print("  the noise. Averaging repeats per evaluation would help.")
        else:
            print("\n  OK: parameter-driven variation clearly exceeds the noise.")
    else:
        print("  repeats are IDENTICAL -> the pipeline is deterministic, so the")
        print("  spread seen during optimisation is all parameter-driven.")

    if paths:
        section('alignment path stability')
        uniq = {tuple(p) for p in paths}
        for p in paths:
            print(f"  {p}")
        print(f"  -> {len(uniq)} distinct path(s) across {len(paths)} repeats")
        if len(uniq) > 1:
            print("  paths differ between identical inputs, so the score can")
            print("  jump discontinuously; that is extra noise on top of the")
            print("  per-curve variation above.")

    if curves and len(curves) > 1:
        section('per-curve S(q) reproducibility')
        # Score the SAME frame from two different repeats against each other,
        # using the very metric the objective uses.  That puts the Monte Carlo
        # noise in the same units as the score, and confines it to the q window
        # actually compared -- the raw curves extend well past it, where S(q)
        # is small and log-noise is large but irrelevant to the fit.
        n_f = min(len(c) for c in curves)
        same = []
        for k in range(n_f):
            for a in range(len(curves)):
                for b in range(a + 1, len(curves)):
                    try:
                        same.append(B.shape_distance_saxs(curves[a][k],
                                                          curves[b][k]))
                    except Exception:
                        pass
        if same:
            same = np.array(same)
            print(f"  same frame, different repeat, scored with the objective's")
            print(f"  own metric ({len(same)} pairs over {n_f} frames):")
            print(f"    min {same.min():.3e}   median {np.median(same):.3e}   "
                  f"max {same.max():.3e}")
            print(f"  aligned exp-vs-sim score for comparison: "
                  f"{finite.mean():.3e}")
            print(f"  -> Monte Carlo noise is about "
                  f"{np.median(same) / finite.mean() * 100:.1f}% of the quantity"
                  f" being minimised.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['analysis', 'full'], default='analysis')
    ap.add_argument('--gsd', help='trajectory to reuse (analysis mode)')
    ap.add_argument('--params', nargs=5, type=float,
                    metavar=('DENSITY', 'U_0', 'r0', 'm', 'gap'),
                    help='parameters to repeat (full mode)')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--mc-samples', type=int, default=None,
                    help='override SAXS_MC_SAMPLES to test its effect')
    ap.add_argument('--signal-spread', type=float, default=0.00303,
                    help='score spread observed across BO evaluations')
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--keep', action='store_true', help='keep intermediate files')
    args = ap.parse_args()

    work = args.workdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'repeatability_out')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    section(f'SAXS repeatability -- {args.mode} mode, {args.repeats} repeats')
    print(f"  SAXS_METRIC = {B.SAXS_METRIC}")

    if args.mode == 'analysis':
        if not args.gsd:
            ap.error('--gsd is required in analysis mode')
        scores, paths, curves = repeat_analysis(
            args.gsd, args.repeats, work, args.mc_samples)
    else:
        if not args.params:
            ap.error('--params is required in full mode')
        scores, paths, curves = repeat_full(args.params, args.repeats, work)

    report(scores, paths, curves, args.signal_spread)

    pd.DataFrame({'repeat': np.arange(len(scores)), 'saxs_score': scores}).to_csv(
        os.path.join(work, 'repeatability_scores.csv'), index=False)
    print(f"\nscores written to {os.path.join(work, 'repeatability_scores.csv')}")
    if not args.keep:
        for r in range(args.repeats):
            shutil.rmtree(os.path.join(work, f'repeat_{r}'), ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""Check that a HOOMD v4 environment can actually run Bridge_20's simulation.

Run inside the hoomd4_gpu environment:

    python test_hoomd4.py                # GPU
    python test_hoomd4.py --device cpu   # CPU
    python test_hoomd4.py --device both  # both, with a TPS comparison

This deliberately exercises the paths `simulation()` needs rather than a
generic Lennard-Jones run, because those are the ones that broke under
HOOMD 2.9.4 on sm_86:

  1. device creation and GPU detection
  2. hoomd.md.nlist.Cell            <- the CUB DeviceScan that failed on v2
  3. hoomd.md.pair.Table            <- the tabulated (n, m) LJ, not pair.lj
  4. Langevin integration           <- must stay finite, no blow-up
  5. hoomd.write.GSD                <- trajectory written
  6. gsd.hoomd.open round-trip      <- what read_trajectory() consumes

Exit status is 0 only if every stage passes.
"""
import argparse
import os
import sys
import tempfile
import traceback

import numpy as np

# --------------------------------------------------------------------------
# The potential from Bridge_20_Hyak_250803.py, kept identical on purpose.
# --------------------------------------------------------------------------
U_0, R0, N_EXP, M_EXP = 10.0, 2.2, 12.0, 6.0
R_MIN, R_CUT, WIDTH = 0.75 * R0, 5.0, 1000


def modified_LJ(r, U_0, n, m, r0):
    """Generalized (n, m) Lennard-Jones potential and force."""
    U = U_0 / (n - m) * (m * (r0 / r) ** n - n * (r0 / r) ** m)
    F = U_0 * m * n * ((r0 / r) ** n - (r0 / r) ** m) / ((n - m) * r)
    return U, F


def make_positions(N, L, rng):
    """Loose lattice placement; spacing is comfortably above the core."""
    side = int(np.ceil(N ** (1 / 3)))
    spacing = L / side
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3), -1).reshape(-1, 3)[:N]
    pos = (grid + 0.5) * spacing - L / 2
    return pos + rng.uniform(-0.05, 0.05, pos.shape)


def run_one(device_kind, N, steps, gsd_period, verbose):
    """Run the full stack on one device. Returns (tps, gsd_path)."""
    import hoomd

    stage = 'device creation'
    if device_kind == 'gpu':
        device = hoomd.device.GPU()
    else:
        device = hoomd.device.CPU()
    print(f"  [{stage}] ok -> {type(device).__name__}")

    rng = np.random.default_rng(3)
    density = 0.005
    L = (N / density) ** (1 / 3)

    stage = 'state creation'
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = N
        snap.particles.types = ['A']
        snap.particles.position[:] = make_positions(N, L, rng)
        snap.particles.typeid[:] = 0
    sim = hoomd.Simulation(device=device, seed=1)
    sim.create_state_from_snapshot(snap)
    print(f"  [{stage}] ok -> N={N}, box L={L:.2f}")

    stage = 'nlist.Cell'
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    print(f"  [{stage}] ok")

    stage = 'pair.Table'
    r = np.linspace(R_MIN, R_CUT, WIDTH)
    U, F = modified_LJ(r, U_0, N_EXP, M_EXP, R0)
    if not (np.all(np.isfinite(U)) and np.all(np.isfinite(F))):
        raise RuntimeError("tabulated U/F contain non-finite values")
    table = hoomd.md.pair.Table(nlist=nlist, default_r_cut=R_CUT)
    table.params[('A', 'A')] = dict(r_min=R_MIN, U=U, F=F)
    print(f"  [{stage}] ok -> |F|max={np.abs(F).max():.3e}, "
          f"dx/step={np.abs(F).max() * 0.001 ** 2:.2e} sigma")

    stage = 'integrator'
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    langevin.gamma['A'] = 1.0
    sim.operations.integrator = hoomd.md.Integrator(
        dt=0.001, methods=[langevin], forces=[table])
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    print(f"  [{stage}] ok")

    stage = 'write.GSD'
    tmpdir = tempfile.mkdtemp(prefix='hoomd4_test_')
    gsd_path = os.path.join(tmpdir, f'traj_{device_kind}.gsd')
    writer = hoomd.write.GSD(filename=gsd_path,
                            trigger=hoomd.trigger.Periodic(gsd_period),
                            filter=hoomd.filter.All(),
                            mode='wb')
    sim.operations.writers.append(writer)
    print(f"  [{stage}] ok -> {gsd_path}")

    stage = f'run ({steps} steps)'
    sim.run(steps)
    tps = sim.tps
    pe = thermo.potential_energy
    print(f"  [{stage}] ok -> {tps:.0f} TPS, potential_energy={pe:.4f}")
    if pe is not None and not np.isfinite(pe):
        raise RuntimeError("potential energy is not finite -- the run blew up")

    stage = 'flush GSD'
    try:
        writer.flush()
    except AttributeError:
        sim.operations.writers.clear()   # older v4: closes on teardown
    del sim, writer
    print(f"  [{stage}] ok")

    return tps, gsd_path


def check_gsd(path, verbose):
    """Read the trajectory exactly the way read_trajectory() does."""
    import gsd.hoomd

    with gsd.hoomd.open(name=path, mode='r') as traj:
        frames = len(traj)
        if frames == 0:
            raise RuntimeError("trajectory contains no frames")
        first = traj[0]
        pos = np.asarray(first.particles.position)
        box = np.asarray(first.configuration.box)
        has_orientation = getattr(first.particles, 'orientation', None) is not None
    print(f"  [gsd round-trip] ok -> {frames} frames, position{pos.shape}, "
          f"box L={box[0]:.2f}, orientation={'yes' if has_orientation else 'no'}")
    if not np.all(np.isfinite(pos)):
        raise RuntimeError("positions in the GSD file are not finite")
    return frames


def report_environment():
    import hoomd
    print("=" * 70)
    print("environment")
    print("=" * 70)
    print(f"  python        {sys.version.split()[0]}")
    print(f"  hoomd         {hoomd.version.version}")
    print(f"  gpu_enabled   {hoomd.version.gpu_enabled}")
    print(f"  gpu_platform  {getattr(hoomd.version, 'gpu_platform', 'n/a')}")
    try:
        import gsd
        print(f"  gsd           {gsd.__version__}")
    except Exception as exc:
        print(f"  gsd           UNAVAILABLE ({exc})")

    if hoomd.version.gpu_enabled:
        try:
            devices = hoomd.device.GPU.get_available_devices()
            print(f"  visible GPUs  {len(devices)}")
            for d in devices:
                print(f"                {d}")
        except Exception as exc:
            print(f"  visible GPUs  query failed: {exc}")
    else:
        print("  NOTE: this HOOMD build has no GPU support compiled in.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['gpu', 'cpu', 'both'], default='gpu')
    ap.add_argument('--n-particles', type=int, default=5000,
                    help='matches simulation()\'s default N')
    ap.add_argument('--steps', type=int, default=20000,
                    help='enough to measure a stable TPS')
    ap.add_argument('--gsd-period', type=int, default=5000)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    try:
        import hoomd  # noqa: F401
    except Exception:
        print("FAILED: cannot import hoomd -- wrong environment?")
        traceback.print_exc()
        return 1

    report_environment()

    kinds = ['gpu', 'cpu'] if args.device == 'both' else [args.device]
    results, failures = {}, []

    for kind in kinds:
        print("=" * 70)
        print(f"{kind.upper()} run")
        print("=" * 70)
        try:
            tps, gsd_path = run_one(kind, args.n_particles, args.steps,
                                    args.gsd_period, args.verbose)
            check_gsd(gsd_path, args.verbose)
            results[kind] = tps
            print(f"  RESULT: {kind.upper()} PASSED\n")
        except Exception as exc:
            failures.append(kind)
            print(f"  RESULT: {kind.upper()} FAILED -> "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            print()

    print("=" * 70)
    print("summary")
    print("=" * 70)
    for kind in kinds:
        if kind in results:
            tps = results[kind]
            print(f"  {kind:<4} PASS  {tps:>9.0f} TPS")
        else:
            print(f"  {kind:<4} FAIL")

    if 'gpu' in results and 'cpu' in results:
        speedup = results['gpu'] / results['cpu']
        print(f"\n  GPU speedup: {speedup:.1f}x")

    # Extrapolate to a production run so the cost is concrete.
    for kind, tps in results.items():
        for steps in (1_000_000, 15_000_000):
            hours = steps / tps / 3600
            print(f"  {kind}: {steps:>12,d} steps -> {hours:6.2f} h "
                  f"({hours * 100:7.1f} h for 100 BO iterations)")

    if failures:
        print(f"\nFAILED on: {', '.join(failures)}")
        return 1
    print("\nALL STAGES PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())

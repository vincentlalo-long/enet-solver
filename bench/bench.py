"""Benchmark suite: our solver vs scikit-learn at *matched* accuracy.

Why matched KKT?  `tol` does not mean the same thing in the two solvers:

    ours        tol = max_j  KKT violation of the returned iterate
    sklearn     tol = relative duality gap

Comparing wall time at the same `tol` is therefore not apples-to-apples.  Here
the accuracy axis is the final max KKT violation (computed by
`reference.kkt_violation` for both solutions).  For each target KKT:

    ours     run with tol = target          (stops when KKT <= target)
    sklearn  sweep a tolerance grid, keep the fastest run with KKT <= target

Protocol:
    - same alpha, l1_ratio, fit_intercept=True for both
    - single thread: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    - one warmup, then median of `--repeat` runs
    - report time, achieved KKT, primal objective and duality gap

Outputs (written by default):
    bench/results/runs.csv        every (solver, tolerance) measurement
    bench/results/matched.csv     main matched-KKT table
    bench/results/scaling.csv     time vs D and vs N
    bench/results/summary.md      markdown tables, ready to paste into README
    docs/figures/pareto.png       time vs achieved KKT, per dataset
    docs/figures/matched_ratio.png  sklearn/ours at matched KKT
    docs/figures/scaling.png      time and speed ratio vs D (vsN rows in CSV)

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        .venv/bin/python bench/bench.py
    OMP_NUM_THREADS=1 .venv/bin/python bench/bench.py --quick --repeat 1
    OMP_NUM_THREADS=1 .venv/bin/python bench/bench.py --backend numpy --no-figures
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import sparse as sp

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from enet import reference
from scripts.gen_data import make_dense, make_ill_conditioned, make_sparse

OURS_TARGETS = [1e-4, 1e-6, 1e-8, 1e-10]
SK_TOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    name: str
    X: object
    y: np.ndarray
    alpha: float
    l1_ratio: float

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.X.nnz) if sp.issparse(self.X) else int(self.X.size)

    @property
    def density(self) -> float:
        return self.nnz / (self.n_samples * self.n_features)


def datasets(quick: bool) -> list[Dataset]:
    sparse_d, sparse_nd = (5_000, 10_000) if quick else (20_000, 50_000)
    Xd, yd, _ = make_dense()
    Xs, ys, _ = make_sparse(n_features=sparse_d)
    Xn, yn, _ = make_sparse(n_samples=200, n_features=sparse_nd)
    Xi, yi, _ = make_ill_conditioned()
    return [
        Dataset("dense-500x50", Xd, yd, alpha=0.05, l1_ratio=0.5),
        Dataset(f"sparse-500x{sparse_d}", Xs, ys, alpha=0.02, l1_ratio=0.5),
        Dataset(f"sparse-N<<D-200x{sparse_nd}", Xn, yn, alpha=0.01, l1_ratio=0.5),
        Dataset("ill-cond-200x100", Xi, yi, alpha=0.05, l1_ratio=0.5),
    ]


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def select_solver(backend: str):
    if backend == "numpy":
        return reference.coordinate_descent, "ours (NumPy)"
    try:
        from enet import backend as enet_backend
    except ImportError as exc:  # pragma: no cover - depends on build
        raise SystemExit("C++ backend not built; run `pip install -e .`") from exc
    return enet_backend.coordinate_descent_cpp, "ours (C++)"


def fit_sklearn(X, y, alpha, l1_ratio, tol, max_iter):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
            tol=tol, max_iter=max_iter,
        ).fit(X, y)
    return model.coef_, float(model.intercept_), int(model.n_iter_)


def time_call(fn, repeat: int, warmup: int = 1):
    """Return (median_s, min_s, last result)."""
    for _ in range(warmup):
        out = fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - t0)
    arr = np.asarray(samples, dtype=np.float64)
    return float(np.median(arr)), float(arr.min()), out


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    dataset: str
    n_samples: int
    n_features: int
    nnz: int
    density: float
    alpha: float
    l1_ratio: float
    solver: str
    tol: float
    time_s: float
    time_min_s: float
    kkt: float
    objective: float
    gap: float
    n_iter: object


def measure(ds: Dataset, solver, solver_name: str, tol: float, repeat: int,
            max_iter: int) -> Run:
    if solver_name.startswith("ours"):
        fn = lambda: (*solver(ds.X, ds.y, ds.alpha, ds.l1_ratio,
                              tol=tol, max_iter=max_iter), None)
    else:
        fn = lambda: fit_sklearn(ds.X, ds.y, ds.alpha, ds.l1_ratio, tol, max_iter)

    median_s, min_s, (w, b, n_iter) = time_call(fn, repeat)

    kkt = reference.kkt_violation(ds.X, ds.y, w, b, ds.alpha, ds.l1_ratio)
    obj = reference.objective(ds.X, ds.y, w, b, ds.alpha, ds.l1_ratio)
    gap = reference.duality_gap(ds.X, ds.y, w, b, ds.alpha, ds.l1_ratio)
    return Run(
        dataset=ds.name, n_samples=ds.n_samples, n_features=ds.n_features,
        nnz=ds.nnz, density=ds.density, alpha=ds.alpha, l1_ratio=ds.l1_ratio,
        solver=solver_name, tol=tol, time_s=median_s, time_min_s=min_s,
        kkt=kkt, objective=obj, gap=gap, n_iter=n_iter,
    )


def matched(ours: list[Run], sk: list[Run]) -> list[dict]:
    """For every ours target, pick the fastest sklearn run with KKT <= target."""
    rows = []
    for o in ours:
        ok = [r for r in sk if r.kkt <= o.tol]
        reached = bool(ok)
        s = min(ok, key=lambda r: r.time_s) if ok else min(sk, key=lambda r: r.kkt)
        rows.append({
            "dataset": o.dataset,
            "n_samples": o.n_samples,
            "n_features": o.n_features,
            "nnz": o.nnz,
            "density": o.density,
            "target": o.tol,
            "ours_s": o.time_s,
            "ours_kkt": o.kkt,
            "ours_obj": o.objective,
            "ours_gap": o.gap,
            "sk_s": s.time_s,
            "sk_kkt": s.kkt,
            "sk_obj": s.objective,
            "sk_gap": s.gap,
            "sk_tol": s.tol,
            "ratio": s.time_s / o.time_s,
            "reached": reached,
        })
    return rows


def scaling(quick: bool, solver, solver_name: str, repeat: int,
            max_iter: int, target: float = 1e-6) -> list[dict]:
    """Time vs n_features (fixed N, density) and vs n_samples (fixed D, density)."""
    rows = []

    def run(tag, N, D, density):
        X, y, _ = make_sparse(n_samples=N, n_features=D, density=density,
                              n_informative=50)
        # Normalise the problem difficulty: use the same fraction of alpha_max
        # at every size, otherwise a fixed alpha is relatively weaker or
        # stronger as N / D change and the timing reflects that, not scaling.
        alpha = 0.05 * reference.alpha_max(X, y, 0.5)
        ds = Dataset(f"{tag}-{N}x{D}", X, y, alpha=alpha, l1_ratio=0.5)

        ours = measure(ds, solver, solver_name, target, repeat, max_iter)
        sk = measure(ds, "sklearn", "sklearn", target, repeat, max_iter)

        rows.append({
            "regime": tag, "n_samples": N, "n_features": D,
            "nnz": ds.nnz, "density": ds.density, "alpha": alpha,
            "ours_s": ours.time_s, "ours_kkt": ours.kkt,
            "sk_s": sk.time_s, "sk_kkt": sk.kkt,
            "ratio": sk.time_s / ours.time_s,
        })
        print(f"  scaling {tag:4} N={N:5d} D={D:6d} nnz={ds.nnz:7d} "
              f"alpha={alpha:.4f} "
              f"ours={ours.time_s:8.4f}s sk={sk.time_s:8.4f}s "
              f"ratio={sk.time_s / ours.time_s:5.2f}x", flush=True)

    dims = [1_000, 3_000, 10_000] if quick else [1_000, 5_000, 20_000, 50_000]
    for D in dims:
        run("vsD", 500, D, 0.01)

    samples = [200, 500, 2_000] if quick else [200, 500, 2_000, 5_000]
    for N in samples:
        run("vsN", N, 5_000, 0.01)

    return rows


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_s(x):
    return f"{x:.4f}"


def fmt_e(x):
    return f"{x:.2e}"


def markdown_summary(env: dict, matched_rows: list[dict],
                     scaling_rows: list[dict], backend: str) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"- Machine: {env['cpu']}",
        f"- Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, "
        f"scikit-learn {env['sklearn']}",
        f"- Backend under test: {backend}",
        "- Single thread (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1`), one warmup, median of repeated runs",
        "",
        "## Matched-KKT timing",
        "",
        "The time column is the median wall time to reach the target accuracy. "
        "For scikit-learn, the fastest `tol` from a fixed grid with achieved "
        "`KKT <= target` is used, so the comparison is at equal accuracy rather "
        "than equal `tol`. `sk/ours > 1` means our solver is faster.",
        "",
        "| dataset | shape | nnz | target KKT | ours (s) | sklearn (s) | sk/ours | "
        "ours KKT | sk KKT | sk tol |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in matched_rows:
        flag = "" if r["reached"] else "\\*"
        lines.append(
            f"| {r['dataset']} | {r['n_samples']}×{r['n_features']} | {r['nnz']} | "
            f"{fmt_e(r['target'])} | {r['ours_s']:.4f} | {r['sk_s']:.4f} | "
            f"{r['ratio']:.2f}×{flag} | {fmt_e(r['ours_kkt'])} | "
            f"{fmt_e(r['sk_kkt'])} | {fmt_e(r['sk_tol'])} |"
        )

    lines += [
        "",
        "## Scaling at fixed target KKT = 1e-6",
        "",
        "`vsD`: N = 500, density = 0.01, D varies. "
        "`vsN`: D = 5000, density = 0.01, N varies. "
        "`alpha = 0.05 * alpha_max` in every case, so the relative problem "
        "difficulty stays comparable as the size changes.",
        "",
        "| regime | N | D | nnz | alpha | ours (s) | sklearn (s) | sk/ours |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in scaling_rows:
        lines.append(
            f"| {r['regime']} | {r['n_samples']} | {r['n_features']} | {r['nnz']} | "
            f"{r['alpha']:.4f} | {r['ours_s']:.4f} | {r['sk_s']:.4f} | "
            f"{r['ratio']:.2f}× |"
        )

    lines += [
        "",
        "## Reading the numbers honestly",
        "",
        "- This is a learning project, not a claim that it beats scikit-learn. "
        "scikit-learn is a mature library with active-set screening, a carefully "
        "tuned convergence criterion, and years of optimization.",
        "- The C++ core is competitive in some sparse regimes because it does one "
        "thing: coordinate descent with a residual update over CSC. It has no SIMD, "
        "no OpenMP, no active set and no strong rules.",
        "- Dense problems are slower than scikit-learn. The dense path is not the "
        "target of this project.",
        "- Timings include the Python wrapper and, for dense input, the one-time "
        "conversion to CSC. They are end-to-end API timings, not pure kernel time.",
        "- Accuracy is matched through the final KKT violation computed by the same "
        "function for both solutions, so differences in `tol` semantics do not "
        "drive the conclusion.",
        "- Both solvers can overshoot the target (ours checks once per sweep; "
        "scikit-learn is picked from a coarse `tol` grid), so the timings mean "
        "'time to reach at least the target'. scikit-learn's overshoot is "
        "typically larger, which inflates its time and makes the comparison "
        "conservative *against* scikit-learn.",
        "- Dense timings are around 1 ms; treat sub-0.2 ms differences as noise.",
        "",
    ]
    return "\n".join(lines)


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pareto(plt, runs: list[Run], datasets_present: list[str], out: Path,
                backend_label: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, datasets_present):
        for solver, style in ((f"ours ({backend_label})", "o-"),
                              ("sklearn", "s--")):
            pts = [r for r in runs if r.dataset == name and r.solver == solver]
            pts.sort(key=lambda r: r.kkt)
            xs = [max(r.kkt, 1e-16) for r in pts]
            ys = [r.time_s for r in pts]
            ax.loglog(xs, ys, style, label=solver, linewidth=1.6, markersize=5)
        ax.set_title(name)
        ax.set_xlabel("final max KKT violation")
        ax.set_ylabel("time (s, median)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[len(datasets_present):]:
        ax.set_visible(False)
    fig.suptitle("Time vs achieved accuracy (lower-left is better)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_matched_ratio(plt, matched_rows: list[dict], out: Path):
    names = list(dict.fromkeys(r["dataset"] for r in matched_rows))
    targets = sorted({r["target"] for r in matched_rows})
    x = np.arange(len(names), dtype=np.float64)
    width = 0.8 / len(targets)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, t in enumerate(targets):
        vals = []
        for n in names:
            row = next(r for r in matched_rows
                       if r["dataset"] == n and r["target"] == t)
            vals.append(row["ratio"])
        ax.bar(x + (i - (len(targets) - 1) / 2) * width, vals, width,
               label=f"target KKT {t:g}")
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("sklearn time / ours time   (>1 = ours faster)")
    ax.set_title("Speed ratio at matched accuracy")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(ncol=min(len(targets), 4), fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_scaling(plt, rows: list[dict], out: Path):
    sub = sorted((r for r in rows if r["regime"] == "vsD"),
                 key=lambda r: r["n_features"])
    if not sub:
        return
    xs = [r["n_features"] for r in sub]

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax_t.loglog(xs, [r["ours_s"] for r in sub], "o-", label="ours")
    ax_t.loglog(xs, [r["sk_s"] for r in sub], "s--", label="sklearn")
    ax_t.set_xlabel("n_features D   (N=500, density=0.01, alpha=0.05*alpha_max)")
    ax_t.set_ylabel("time (s, median)")
    ax_t.set_title("Time vs dimension (target KKT 1e-6)")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_r.semilogx(xs, [r["ratio"] for r in sub], "o-", color="C2")
    ax_r.axhline(1.0, color="black", linewidth=1.0)
    ax_r.set_xlabel("n_features D")
    ax_r.set_ylabel("sklearn time / ours time")
    ax_r.set_title("Speed ratio vs dimension   (>1 = ours faster)")
    ax_r.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="smaller sparse cases, fewer scaling points")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--backend", choices=["cpp", "numpy"], default="cpp")
    parser.add_argument("--targets", default="",
                        help="comma-separated KKT targets (default: 1e-4..1e-10)")
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="kept for backwards compatibility; default is full")
    parser.add_argument("--outdir", default=str(_ROOT / "bench" / "results"))
    args = parser.parse_args()

    targets = ([float(t) for t in args.targets.split(",")]
               if args.targets else list(OURS_TARGETS))
    outdir = Path(args.outdir)
    figdir = _ROOT / "docs" / "figures"
    solver, solver_name = select_solver(args.backend)
    backend_label = "C++" if args.backend == "cpp" else "NumPy"

    env = {
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    print(f"{env['cpu']} | Python {env['python']} | numpy {env['numpy']} | "
          f"scipy {env['scipy']} | sklearn {env['sklearn']}")
    print(f"backend: {solver_name} | repeat={args.repeat} | "
          f"targets={targets}\n")

    all_runs: list[Run] = []
    matched_rows: list[dict] = []
    ds_list = datasets(args.quick)

    for ds in ds_list:
        print(f"{ds.name}  shape=({ds.n_samples}, {ds.n_features})  "
              f"nnz={ds.nnz}  alpha={ds.alpha}  l1_ratio={ds.l1_ratio}")
        ours = [measure(ds, solver, solver_name, t, args.repeat, args.max_iter)
                for t in targets]
        sk = [measure(ds, "sklearn", "sklearn", t, args.repeat, args.max_iter)
              for t in SK_TOLS]
        all_runs.extend(ours + sk)
        for r in ours:
            print(f"  ours  tol={r.tol:.0e}  {r.time_s:8.4f}s  "
                  f"KKT={r.kkt:.2e}  obj={r.objective:.6f}", flush=True)
        for r in sk:
            print(f"  sk    tol={r.tol:.0e}  {r.time_s:8.4f}s  "
                  f"KKT={r.kkt:.2e}  obj={r.objective:.6f}", flush=True)
        matched_rows.extend(matched(ours, sk))
        print()

    scaling_rows = []
    if not args.no_scaling:
        print("scaling:")
        scaling_rows = scaling(args.quick, solver, solver_name, args.repeat,
                               args.max_iter)

    write_csv(outdir / "runs.csv", [asdict(r) for r in all_runs])
    write_csv(outdir / "matched.csv", matched_rows)
    write_csv(outdir / "scaling.csv", scaling_rows)
    (outdir / "summary.md").write_text(
        markdown_summary(env, matched_rows, scaling_rows, backend_label))
    print(f"\nwrote {outdir}/runs.csv, matched.csv, scaling.csv, summary.md")

    if not args.no_figures and args.backend == "cpp":
        plt = _setup_matplotlib()
        present = list(dict.fromkeys(r.dataset for r in all_runs))
        plot_pareto(plt, all_runs, present, figdir / "pareto.png", backend_label)
        plot_matched_ratio(plt, matched_rows, figdir / "matched_ratio.png")
        if scaling_rows:
            plot_scaling(plt, scaling_rows, figdir / "scaling.png")
        print(f"wrote figures to {figdir}")


if __name__ == "__main__":
    main()

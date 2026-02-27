from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QxSeries:
    year: int
    qx: np.ndarray  # indexed by exact age: qx[60] = q60


def load_qx_csv(path: str | Path, *, year: int) -> QxSeries:
    """
    Expected columns: Year, x, q(x)
    Rows: one per exact age, 0..ultimate_age (often 119 with q=1).
    """
    df = pd.read_csv(path)

    # Normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Year", "x", "q(x)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)} (found {df.columns.tolist()})")

    df = df[df["Year"] == year].copy()
    if df.empty:
        raise ValueError(f"{path}: no rows found for Year={year}")

    df["x"] = pd.to_numeric(df["x"], errors="raise").astype(int)
    df["q(x)"] = pd.to_numeric(df["q(x)"], errors="raise").astype(float)

    df.sort_values("x", inplace=True)

    ages = df["x"].to_numpy(dtype=int)
    if ages.min() != 0:
        raise ValueError(f"{path}: expected ages to start at 0; got min age={ages.min()}")

    expected = np.arange(0, ages.max() + 1, dtype=int)
    if not np.array_equal(ages, expected):
        missing_ages = sorted(set(expected) - set(ages))
        raise ValueError(f"{path}: age gaps detected; missing ages: {missing_ages[:20]}...")

    qx = df["q(x)"].to_numpy(dtype=float)

    # Basic qx validation
    if (qx < 0).any() or (qx > 1).any():
        bad = np.where((qx < 0) | (qx > 1))[0][:10].tolist()
        raise ValueError(f"{path}: q(x) out of [0,1] at indices (ages) {bad}")

    return QxSeries(year=year, qx=qx)


def _format_np_constant(name: str, arr: np.ndarray) -> str:
    vals = []
    for v in arr:
        if v >= 1.0:
            vals.append("1.0")
        else:
            vals.append(f"{v:.6f}".rstrip("0").rstrip("."))

    lines = []
    for i in range(0, len(vals), 10):
        chunk = ", ".join(vals[i : i + 10])
        suffix = "," if (i + 10) < len(vals) else ""
        lines.append(f"    {chunk}{suffix}")

    return f"{name} = np.array([\n" + "\n".join(lines) + "\n], dtype=float)\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--male_csv", required=True)
    ap.add_argument("--female_csv", required=True)
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--out_dir", default="simulation/domain")
    ap.add_argument("--out_py", default="us_mortality_tables_2022.py")
    args = ap.parse_args()

    year = args.year
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    male = load_qx_csv(args.male_csv, year=year)
    female = load_qx_csv(args.female_csv, year=year)

    # Save npy
    np.save(out_dir / f"qx_us_male_{year}.npy", male.qx)
    np.save(out_dir / f"qx_us_female_{year}.npy", female.qx)

    # Write drop-in python module
    py_path = out_dir / args.out_py
    parts = [
        "from __future__ import annotations\n",
        "import numpy as np\n\n",
        "# Source: SSA historical mortality probabilities (Trustees Report basis)\n",
        f"# Extracted year: {year}\n\n",
        _format_np_constant(f"QX_US_MALE_{year}", male.qx),
        "\n",
        _format_np_constant(f"QX_US_FEMALE_{year}", female.qx),
    ]
    py_path.write_text("".join(parts), encoding="utf-8")

    print(f"Wrote: {py_path}")
    print(f"Wrote: {out_dir / f'qx_us_male_{year}.npy'}")
    print(f"Wrote: {out_dir / f'qx_us_female_{year}.npy'}")
    print(f"Sanity: male q60={male.qx[60]:.6f}, female q60={female.qx[60]:.6f}")
    print(f"Sanity: male last q={male.qx[-1]:.6f}, female last q={female.qx[-1]:.6f}")


if __name__ == "__main__":
    main()
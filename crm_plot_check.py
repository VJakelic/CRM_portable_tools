#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine-specific paths, working filenames, run names, measured OES targets and
PID-distance mapping are supplied at runtime. The default CRM layout convention
is:

    <crm-root>/bin/CRM
    <crm-root>/input/input_nofit.txt
    <crm-root>/exp/

Use ``python3 crm_plot_check.py --help`` for all options.
"""

from __future__ import print_function

import argparse
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Runtime config populated by configure_from_args().
CRM_ROOT = None
EXP_DIR = None
ACTIVE_INPUT_FILE = None
CRM_CONFIG = None
CRM_EXECUTABLE = None
CRM_RUN_NAME = None
CRM_OUTPUT_FILE = None
OUTPUT_ROOT = None

SUPPORTED_CS = (11, 12, 21, 22)
OES_COLUMNS = ["Ha-s", "Hb-s", "Hg-s", "Fulcher-s"]
MEASURED = {}
PID_TO_DISTANCE_CM = {}

INPUT_DENSITY_COLUMNS = [
    "ne", "nH", "nH2", "nHp", "nH2p", "nH3p", "nHm"
]
CRM_HYDROGEN_COLUMNS = [
    "H", "H+", "e-", "H2+", "H3+", "H-", "H2"
]


def trapezoid_integral(y, x):
    """NumPy-version-compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _looks_like_crm_root(path):
    path = os.path.abspath(os.path.expanduser(path))
    return (
        os.path.isdir(os.path.join(path, "exp"))
        and os.path.isdir(os.path.join(path, "input"))
        and os.path.isfile(os.path.join(path, "bin", "CRM"))
    )


def discover_crm_root(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = os.environ.get("CRM_ROOT")
    if env_root:
        candidates.append(env_root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([script_dir, os.path.dirname(script_dir), os.getcwd()])

    seen = set()
    for item in candidates:
        path = os.path.abspath(os.path.expanduser(item))
        if path in seen:
            continue
        seen.add(path)
        if _looks_like_crm_root(path):
            return path

    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    raise RuntimeError(
        "Could not locate the CRM root. Use --crm-root /path/to/CRM or set "
        "the CRM_ROOT environment variable."
    )


def resolve_root_path(value, crm_root):
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(crm_root, value))


def load_distance_map(path):
    path = os.path.abspath(os.path.expanduser(path))
    ensure_exists(path, "PID-distance map")
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
        lower = {str(c).strip().lower(): c for c in df.columns}
        if "pid" in lower and "distance_cm" in lower:
            out = df[[lower["pid"], lower["distance_cm"]]].copy()
            out.columns = ["PID", "distance_cm"]
        else:
            raise ValueError
    except Exception:
        out = pd.read_csv(
            path,
            sep=r"[\s,]+",
            engine="python",
            comment="#",
            header=None,
            names=["PID", "distance_cm"],
        )
    out = out.dropna(how="all")
    out["PID"] = pd.to_numeric(out["PID"], errors="raise").astype(int)
    out["distance_cm"] = pd.to_numeric(
        out["distance_cm"], errors="raise"
    ).astype(float)
    if out["PID"].duplicated().any():
        dup = out.loc[out["PID"].duplicated(), "PID"].tolist()
        raise RuntimeError("Duplicate PID(s) in distance map: {}".format(dup))
    if not np.all(np.isfinite(out["distance_cm"].to_numpy(dtype=float))):
        raise RuntimeError("Distance map contains non-finite distance_cm values.")
    if out["distance_cm"].duplicated().any():
        dup = out.loc[out["distance_cm"].duplicated(), "distance_cm"].tolist()
        raise RuntimeError(
            "Distance map contains duplicate spatial positions: {}".format(dup)
        )
    return dict(zip(out["PID"].tolist(), out["distance_cm"].tolist()))


def load_targets_file(path):
    path = os.path.abspath(os.path.expanduser(path))
    ensure_exists(path, "OES targets file")
    values = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,=]+", line)
            if len(parts) >= 2 and parts[0] in OES_COLUMNS:
                values[parts[0]] = float(parts[1])
    missing = [k for k in OES_COLUMNS if k not in values]
    if missing:
        raise RuntimeError(
            "Targets file is missing: {}".format(", ".join(missing))
        )
    return values



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run/plot one CRM input and calculate corrected OES errors."
    )
    parser.add_argument("input_file", help="CRM input file to evaluate.")
    parser.add_argument(
        "--crm-root",
        default=None,
        help="CRM root. Can also be supplied by CRM_ROOT environment variable.",
    )
    parser.add_argument(
        "--active-input",
        required=True,
        help=(
            "CRM working input file expected by the CRM config. Absolute path "
            "or path relative to CRM root, e.g. exp/my_input.txt."
        ),
    )
    parser.add_argument(
        "--config",
        default="input/input_nofit.txt",
        help="CRM config path, relative to CRM root unless absolute.",
    )
    parser.add_argument(
        "--executable",
        default="bin/CRM",
        help="CRM executable path, relative to CRM root unless absolute.",
    )
    parser.add_argument(
        "--run-name",
        default="check",
        help="Run name passed to CRM. Default: check.",
    )
    parser.add_argument(
        "--crm-output",
        default=None,
        help="CRM output file. Default: <crm-root>/CRM_res_<run-name>.txt.",
    )
    parser.add_argument(
        "--output-root",
        default="crm_quick_checks",
        help="Results directory, relative to CRM root unless absolute.",
    )
    parser.add_argument(
        "--distance-map",
        required=True,
        help="Text/CSV file with columns PID and distance_cm.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--targets",
        nargs=4,
        type=float,
        metavar=("HA", "HB", "HG", "FULCHER"),
        help="Experimental OES targets in ph m^-2 s^-1.",
    )
    target_group.add_argument(
        "--targets-file",
        help="Text file containing Ha-s, Hb-s, Hg-s and Fulcher-s targets.",
    )
    parser.add_argument(
        "--cs",
        type=int,
        choices=SUPPORTED_CS,
        default=None,
        help="Temporarily use k_eiae 11, 12, 21 or 22; config is restored after run.",
    )
    return parser.parse_args()


def configure_from_args(args):
    global CRM_ROOT, EXP_DIR, ACTIVE_INPUT_FILE, CRM_CONFIG, CRM_EXECUTABLE
    global CRM_RUN_NAME, CRM_OUTPUT_FILE, OUTPUT_ROOT, MEASURED, PID_TO_DISTANCE_CM

    CRM_ROOT = discover_crm_root(args.crm_root)
    EXP_DIR = os.path.join(CRM_ROOT, "exp")
    ACTIVE_INPUT_FILE = resolve_root_path(args.active_input, CRM_ROOT)
    CRM_CONFIG = resolve_root_path(args.config, CRM_ROOT)
    CRM_EXECUTABLE = resolve_root_path(args.executable, CRM_ROOT)
    CRM_RUN_NAME = str(args.run_name)
    CRM_OUTPUT_FILE = (
        resolve_root_path(args.crm_output, CRM_ROOT)
        if args.crm_output
        else os.path.join(CRM_ROOT, "CRM_res_{}.txt".format(CRM_RUN_NAME))
    )
    OUTPUT_ROOT = resolve_root_path(args.output_root, CRM_ROOT)
    PID_TO_DISTANCE_CM = load_distance_map(args.distance_map)

    if args.targets_file:
        MEASURED = load_targets_file(args.targets_file)
    else:
        MEASURED = dict(zip(OES_COLUMNS, [float(v) for v in args.targets]))

    for line in OES_COLUMNS:
        value = float(MEASURED[line])
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                "OES target {} must be finite and > 0; got {}.".format(
                    line, value
                )
            )


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise IOError("{} not found:\n{}".format(label, path))


def add_distance(df):
    df = df.copy()
    df["PID"] = pd.to_numeric(df["PID"], errors="raise").astype(int)
    df["distance_cm"] = df["PID"].map(PID_TO_DISTANCE_CM)

    if df["distance_cm"].isnull().any():
        bad = df.loc[df["distance_cm"].isnull(), "PID"].tolist()
        raise RuntimeError(
            "No distance mapping defined for PID(s): {}".format(bad)
        )

    return df.sort_values("distance_cm").reset_index(drop=True)


def find_active_k_eiae(path):
    pattern = re.compile(r"^\s*k_eiae\s*=\s*(11|12|21|22)\b")
    found = []

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("!")
                or stripped.startswith(";")
            ):
                continue

            m = pattern.match(line)
            if m:
                found.append(int(m.group(1)))

    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        return None

    raise RuntimeError(
        "More than one active k_eiae line found in {}: {}".format(path, found)
    )


def set_k_eiae(path, setting):
    with open(path, "r") as f:
        lines = f.readlines()

    pattern = re.compile(r"^(\s*k_eiae\s*=\s*)(11|12|21|22)(\b.*)$")
    matches = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("!")
            or stripped.startswith(";")
        ):
            continue

        m = pattern.match(line)
        if m:
            matches.append((i, m))

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one active k_eiae line in {}, found {}.".format(
                path, len(matches)
            )
        )

    i, m = matches[0]
    newline = "\n" if lines[i].endswith("\n") else ""
    suffix = m.group(3).rstrip("\r\n")

    lines[i] = "{}{}{}{}".format(
        m.group(1), int(setting), suffix, newline
    )

    with open(path, "w") as f:
        f.writelines(lines)


def load_input(path):
    df = pd.read_csv(path, sep=r"\s+")

    required = [
        "PID", "ne", "Te", "nH", "nH2",
        "nHp", "nH2p", "nH3p", "nHm",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError("Input file is missing columns: {}".format(missing))

    df["PID"] = pd.to_numeric(df["PID"], errors="raise").astype(int)
    if df["PID"].duplicated().any():
        dup = df.loc[df["PID"].duplicated(), "PID"].tolist()
        raise RuntimeError("Input contains duplicate PID(s): {}".format(dup))
    if len(df) < 2:
        raise RuntimeError("Input must contain at least two PID rows.")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        if not np.all(np.isfinite(df[col].to_numpy(dtype=float))):
            raise RuntimeError(
                "Input column {} contains non-finite values.".format(col)
            )

    if np.any(df["ne"].to_numpy(dtype=float) <= 0.0):
        raise RuntimeError("Input electron density ne must be > 0.")
    if np.any(df["Te"].to_numpy(dtype=float) <= 0.0):
        raise RuntimeError("Input electron temperature Te must be > 0.")
    for col in ["nH", "nH2", "nHp", "nH2p", "nH3p", "nHm"]:
        if np.any(df[col].to_numpy(dtype=float) < 0.0):
            raise RuntimeError("Input density {} must be >= 0.".format(col))

    out = add_distance(df)
    if out["distance_cm"].duplicated().any():
        raise RuntimeError("Input maps two or more PIDs to the same distance_cm.")
    return out


def run_crm():
    if os.path.exists(CRM_OUTPUT_FILE):
        os.remove(CRM_OUTPUT_FILE)

    t0 = time.time()

    process = subprocess.Popen(
        [CRM_EXECUTABLE, CRM_CONFIG, CRM_RUN_NAME],
        cwd=CRM_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    stdout, _ = process.communicate()
    runtime = time.time() - t0

    if process.returncode != 0:
        raise RuntimeError(
            "CRM failed with return code {}:\n{}".format(
                process.returncode, stdout
            )
        )

    if not os.path.isfile(CRM_OUTPUT_FILE):
        raise RuntimeError(
            "CRM output was not produced:\n{}".format(CRM_OUTPUT_FILE)
        )

    return runtime, stdout


def load_crm_output(expected_pids):
    df = pd.read_csv(CRM_OUTPUT_FILE, sep=r"\s+")

    required = ["PID", "opl"] + OES_COLUMNS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "CRM output is missing columns: {}".format(missing)
        )

    df["PID"] = pd.to_numeric(df["PID"], errors="raise").astype(int)
    expected_pids = sorted(int(pid) for pid in expected_pids)
    df = df[df["PID"].isin(expected_pids)].copy()

    found = sorted(df["PID"].tolist())
    if found != expected_pids:
        raise RuntimeError(
            "CRM output does not contain exactly the requested input PID set. "
            "Expected {}, found {}.".format(expected_pids, found)
        )

    df = add_distance(df)

    df["opl"] = pd.to_numeric(df["opl"], errors="raise").astype(float)
    opl = df["opl"].to_numpy(dtype=float)

    if (
        not np.all(np.isfinite(opl))
        or np.any(opl <= 0.0)
    ):
        raise RuntimeError(
            "CRM output contains invalid opl values: {}".format(
                df["opl"].tolist()
            )
        )

    for col in OES_COLUMNS:
        raw = pd.to_numeric(
            df[col], errors="raise"
        ).to_numpy(dtype=float)

        if not np.all(np.isfinite(raw)):
            raise RuntimeError(
                "CRM output contains non-finite values in {}.".format(col)
            )

        df[col + "_vol"] = raw / opl

    return df


def calculate_oes(crm_df):
    x_m = crm_df["distance_cm"].to_numpy(dtype=float) / 100.0
    results = {}

    for col in OES_COLUMNS:
        emissivity = crm_df[col + "_vol"].to_numpy(dtype=float)

        calculated = float(
            trapezoid_integral(emissivity, x_m)
        )

        measured = float(MEASURED[col])
        rel = (calculated - measured) / measured

        results[col] = {
            "calculated": calculated,
            "measured": measured,
            "relative_error": rel,
            "relative_error_percent": 100.0 * rel,
            "absolute_relative_error_percent": 100.0 * abs(rel),
        }

    return results


def plot_input_densities(df, path):
    x = df["distance_cm"].to_numpy(dtype=float)

    plt.figure(figsize=(11, 7))

    for col in INPUT_DENSITY_COLUMNS:
        y = df[col].to_numpy(dtype=float)

        plt.plot(
            x,
            np.where(y > 0.0, y, np.nan),
            "o-",
            label=col
        )

    plt.xlabel("Distance [cm]")
    plt.ylabel("Density [m$^{-3}$]")
    plt.title("CRM input density profiles")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_te(df, path):
    x = df["distance_cm"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["Te"].to_numpy(dtype=float), "o-")
    plt.xlabel("Distance [cm]")
    plt.ylabel("Te")
    plt.title("Input electron-temperature profile")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ion_fractions(df, path):
    x = df["distance_cm"].to_numpy(dtype=float)

    nHp = df["nHp"].to_numpy(dtype=float)
    nH2p = df["nH2p"].to_numpy(dtype=float)
    nH3p = df["nH3p"].to_numpy(dtype=float)

    total = nHp + nH2p + nH3p

    fHp = np.divide(
        nHp, total,
        out=np.full_like(total, np.nan),
        where=total > 0.0
    )
    fH2p = np.divide(
        nH2p, total,
        out=np.full_like(total, np.nan),
        where=total > 0.0
    )
    fH3p = np.divide(
        nH3p, total,
        out=np.full_like(total, np.nan),
        where=total > 0.0
    )

    plt.figure(figsize=(10, 6))
    plt.plot(x, 100.0 * fHp, "o-", label="H+")
    plt.plot(x, 100.0 * fH2p, "o-", label="H2+")
    plt.plot(x, 100.0 * fH3p, "o-", label="H3+")
    plt.xlabel("Distance [cm]")
    plt.ylabel("Positive-ion fraction [%]")
    plt.title("Input positive-ion composition")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_quasineutrality(df, path):
    x = df["distance_cm"].to_numpy(dtype=float)

    positive = (
        df["nHp"].to_numpy(dtype=float)
        + df["nH2p"].to_numpy(dtype=float)
        + df["nH3p"].to_numpy(dtype=float)
    )

    required = (
        df["ne"].to_numpy(dtype=float)
        + df["nHm"].to_numpy(dtype=float)
    )

    rel = (
        positive - required
    ) / np.maximum(required, 1.0e-300)

    plt.figure(figsize=(10, 6))
    plt.plot(x, 100.0 * rel, "o-")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Distance [cm]")
    plt.ylabel("Quasineutrality residual [%]")
    plt.title("(positive ions - ne - nH-) / (ne + nH-)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_oes_emissivity(df, path):
    x = df["distance_cm"].to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))

    for col in OES_COLUMNS:
        plt.plot(
            x,
            df[col + "_vol"].to_numpy(dtype=float),
            "o-",
            label=col
        )

    plt.xlabel("Distance [cm]")
    plt.ylabel("Volumetric emissivity [ph m$^{-3}$ s$^{-1}$]")
    plt.title("CRM OES emissivity")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_crm_densities(df, path):
    available = [
        c for c in CRM_HYDROGEN_COLUMNS
        if c in df.columns
    ]

    if not available:
        return False

    x = df["distance_cm"].to_numpy(dtype=float)

    plt.figure(figsize=(11, 7))
    plotted = False

    for col in available:
        y = pd.to_numeric(
            df[col], errors="coerce"
        ).to_numpy(dtype=float)

        if np.any(np.isfinite(y) & (y > 0.0)):
            plotted = True
            plt.plot(
                x,
                np.where(y > 0.0, y, np.nan),
                "o-",
                label=col
            )

    if not plotted:
        plt.close()
        return False

    plt.xlabel("Distance [cm]")
    plt.ylabel("Density [m$^{-3}$]")
    plt.title("CRM hydrogen/electron output profiles")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

    return True


def plot_relative_errors(results, path):
    labels = OES_COLUMNS
    values = [
        results[c]["relative_error_percent"]
        for c in labels
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 6))
    plt.bar(x, values)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, labels)
    plt.ylabel("Signed relative error [%]")
    plt.title("CRM prediction relative to measured OES")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_outputs(
    output_dir,
    source_file,
    input_df,
    crm_df,
    results,
    runtime,
    used_cs,
    crm_stdout
):
    input_df.to_csv(
        os.path.join(output_dir, "input_profiles.csv"),
        index=False
    )

    crm_df.to_csv(
        os.path.join(output_dir, "CRM_result.csv"),
        index=False
    )

    oes_cols = ["PID", "distance_cm", "opl"]
    for col in OES_COLUMNS:
        oes_cols.extend([col, col + "_vol"])

    crm_df[oes_cols].to_csv(
        os.path.join(output_dir, "OES_profiles.csv"),
        index=False
    )

    error_rows = []

    for col in OES_COLUMNS:
        info = results[col]
        error_rows.append({
            "line": col,
            "calculated_ph_m-2_s-1": info["calculated"],
            "experimental_ph_m-2_s-1": info["measured"],
            "relative_error": info["relative_error"],
            "relative_error_percent": info["relative_error_percent"],
            "absolute_relative_error_percent":
                info["absolute_relative_error_percent"],
        })

    pd.DataFrame(error_rows).to_csv(
        os.path.join(output_dir, "OES_relative_errors.csv"),
        index=False
    )

    with open(
        os.path.join(output_dir, "CRM_stdout.txt"), "w"
    ) as f:
        f.write(crm_stdout)

    positive = (
        input_df["nHp"].to_numpy(dtype=float)
        + input_df["nH2p"].to_numpy(dtype=float)
        + input_df["nH3p"].to_numpy(dtype=float)
    )

    required = (
        input_df["ne"].to_numpy(dtype=float)
        + input_df["nHm"].to_numpy(dtype=float)
    )

    qn_rel = (
        positive - required
    ) / np.maximum(required, 1.0e-300)

    with open(
        os.path.join(output_dir, "summary.txt"), "w"
    ) as f:
        f.write("General CRM input check\n")
        f.write("=" * 78 + "\n\n")
        f.write("Input: {}\n".format(source_file))
        f.write("k_eiae: {}\n".format(used_cs))
        f.write("CRM runtime: {:.2f} s\n".format(runtime))
        f.write(
            "PID range: {} to {}\n".format(
                int(input_df["PID"].min()),
                int(input_df["PID"].max())
            )
        )
        f.write(
            "Distance range: {:.3f} to {:.3f} cm\n".format(
                float(input_df["distance_cm"].min()),
                float(input_df["distance_cm"].max())
            )
        )
        f.write(
            "PID integration span: {:.6f} m\n".format(
                (
                    float(crm_df["distance_cm"].max())
                    - float(crm_df["distance_cm"].min())
                ) / 100.0
            )
        )
        f.write(
            "CRM opl range: {:.6f} to {:.6f} m\n".format(
                float(crm_df["opl"].min()),
                float(crm_df["opl"].max())
            )
        )
        f.write(
            "Maximum |quasineutrality residual|: {:.6E}\n".format(
                float(np.max(np.abs(qn_rel)))
            )
        )
        f.write("\n")
        f.write("OES treatment:\n")
        f.write("  emissivity = CRM line signal / opl\n")
        f.write("  prediction = trapz(emissivity, PID distance)\n")
        f.write("\nOES RELATIVE ERRORS\n")
        f.write("-" * 78 + "\n")

        for col in OES_COLUMNS:
            info = results[col]
            f.write(
                "{:<10s} calc={:.8E}  exp={:.8E}  "
                "rel.error={:+.3f}%  |rel|={:.3f}%\n".format(
                    col,
                    info["calculated"],
                    info["measured"],
                    info["relative_error_percent"],
                    info["absolute_relative_error_percent"],
                )
            )


def main():
    args = parse_args()
    configure_from_args(args)

    source_file = os.path.abspath(
        os.path.expanduser(args.input_file)
    )

    ensure_exists(CRM_ROOT, "CRM root")
    ensure_exists(EXP_DIR, "CRM exp directory")
    ensure_exists(source_file, "Input file")
    ensure_exists(CRM_CONFIG, "CRM configuration")
    ensure_exists(CRM_EXECUTABLE, "CRM executable")

    source_abs = os.path.abspath(source_file)
    active_abs = os.path.abspath(ACTIVE_INPUT_FILE)
    output_abs = os.path.abspath(CRM_OUTPUT_FILE)
    config_abs = os.path.abspath(CRM_CONFIG)
    if output_abs in {source_abs, active_abs, config_abs}:
        raise RuntimeError(
            "CRM output path collides with the source input, active input, or config."
        )

    active_parent = os.path.dirname(ACTIVE_INPUT_FILE)
    if not os.path.isdir(active_parent):
        raise IOError("Active-input directory not found:\n{}".format(active_parent))

    input_df = load_input(source_file)

    if not os.path.isdir(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    source_stem = os.path.splitext(
        os.path.basename(source_file)
    )[0]

    output_dir = os.path.join(
        OUTPUT_ROOT,
        source_stem
    )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    config_backup = os.path.join(
        output_dir,
        "_backup_input_nofit.txt"
    )
    active_backup = os.path.join(
        output_dir,
        "_backup_active_input.txt"
    )
    output_backup = os.path.join(
        output_dir,
        "_backup_CRM_output.txt"
    )

    active_existed = os.path.isfile(ACTIVE_INPUT_FILE)
    output_existed = os.path.isfile(CRM_OUTPUT_FILE)

    shutil.copyfile(
        CRM_CONFIG,
        config_backup
    )

    if active_existed:
        shutil.copyfile(
            ACTIVE_INPUT_FILE,
            active_backup
        )

    if output_existed:
        shutil.copyfile(
            CRM_OUTPUT_FILE,
            output_backup
        )

    try:
        if args.cs is not None:
            set_k_eiae(
                CRM_CONFIG,
                args.cs
            )

        used_cs = find_active_k_eiae(
            CRM_CONFIG
        )
        if used_cs is None:
            raise RuntimeError(
                "No active k_eiae=11/12/21/22 line was found in the CRM config."
            )

        if source_abs != active_abs:
            shutil.copyfile(
                source_file,
                ACTIVE_INPUT_FILE
            )

        runtime, crm_stdout = run_crm()

        shutil.copyfile(
            CRM_OUTPUT_FILE,
            os.path.join(
                output_dir,
                "CRM_result_raw.txt"
            )
        )

        input_pids = input_df["PID"].astype(int).tolist()
        crm_df = load_crm_output(input_pids)

        crm_df = crm_df.sort_values(
            "distance_cm"
        ).reset_index(drop=True)

        results = calculate_oes(
            crm_df
        )

        plot_input_densities(
            input_df,
            os.path.join(
                output_dir,
                "input_density_profiles.png"
            )
        )

        plot_te(
            input_df,
            os.path.join(
                output_dir,
                "input_Te_profile.png"
            )
        )

        plot_ion_fractions(
            input_df,
            os.path.join(
                output_dir,
                "input_positive_ion_fractions.png"
            )
        )

        plot_quasineutrality(
            input_df,
            os.path.join(
                output_dir,
                "input_quasineutrality_residual.png"
            )
        )

        plot_oes_emissivity(
            crm_df,
            os.path.join(
                output_dir,
                "OES_volumetric_emissivity.png"
            )
        )

        plot_crm_densities(
            crm_df,
            os.path.join(
                output_dir,
                "CRM_density_profiles.png"
            )
        )

        plot_relative_errors(
            results,
            os.path.join(
                output_dir,
                "OES_relative_errors.png"
            )
        )

        save_outputs(
            output_dir,
            source_file,
            input_df,
            crm_df,
            results,
            runtime,
            used_cs,
            crm_stdout
        )

        print("")
        print("=" * 78)
        print("GENERAL CRM INPUT CHECK")
        print("=" * 78)
        print("Input: {}".format(source_file))
        print("k_eiae: {}".format(used_cs))
        print(
            "CRM runtime: {:.2f} s".format(runtime)
        )
        print(
            "PID integration span: {:.6f} m".format(
                (
                    float(crm_df["distance_cm"].max())
                    - float(crm_df["distance_cm"].min())
                ) / 100.0
            )
        )
        print(
            "CRM opl: {:.6f} .. {:.6f} m".format(
                float(crm_df["opl"].min()),
                float(crm_df["opl"].max())
            )
        )

        print("")
        print("Corrected OES relative errors :")
        print(
            "  {:<10s} {:>14s} {:>14s} {:>12s} {:>12s}".format(
                "line",
                "calculated",
                "experimental",
                "rel.err %",
                "|rel| %"
            )
        )

        for col in OES_COLUMNS:
            info = results[col]
            print(
                "  {:<10s} {:>14.6E} {:>14.6E} "
                "{:>+12.2f} {:>12.2f}".format(
                    col,
                    info["calculated"],
                    info["measured"],
                    info["relative_error_percent"],
                    info["absolute_relative_error_percent"],
                )
            )

        print("")
        print("Saved plots/results to")
        print("  {}".format(output_dir))

    finally:
        if os.path.isfile(config_backup):
            shutil.copyfile(
                config_backup,
                CRM_CONFIG
            )

        if active_existed:
            if os.path.isfile(active_backup):
                shutil.copyfile(
                    active_backup,
                    ACTIVE_INPUT_FILE
                )
        elif os.path.exists(ACTIVE_INPUT_FILE):
            os.remove(ACTIVE_INPUT_FILE)

        if output_existed:
            if os.path.isfile(output_backup):
                shutil.copyfile(
                    output_backup,
                    CRM_OUTPUT_FILE
                )
        elif os.path.exists(CRM_OUTPUT_FILE):
            os.remove(CRM_OUTPUT_FILE)

    print("")
    print("CRM working state restored.")


if __name__ == "__main__":
    main()

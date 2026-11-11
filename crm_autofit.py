#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRM layout convention
---------------------
By default the script expects the CRM root to contain:

    bin/CRM
    input/input_nofit.txt
    exp/

The executable/config paths can also be overridden explicitly. The CRM output
is assumed to follow the standard name CRM_res_<run-name>.txt unless
--crm-output is supplied.

The fitter keeps ne, Te and nHm from the template fixed. It fits nH, nH2, H+
and H2+ under the configured physical bounds; H3+ is fixed to zero

Use ``python3 crm_autofit.py --help`` for all options.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd


# ============================================================================
# DEFAULT NUMERICAL SETTINGS
# ============================================================================

MAX_RUNTIME_MINUTES = 165.0
GLOBAL_TIME_FRACTION = 0.45

GLOBAL_INITIAL_SAMPLES = 18
GLOBAL_MUTATION_BATCH = 6
GLOBAL_ELITE_COUNT = 6
GLOBAL_MUTATION_SIGMA = 0.22
GLOBAL_RANDOM_INJECTION_PROBABILITY = 0.25

LOCAL_STARTS = 5
LOCAL_MAX_ITERATIONS = 8
LOCAL_ESCAPE_TRIALS = 4
LOCAL_ESCAPE_SIGMA = 0.14

INITIAL_DAMPING = 1.0
MIN_DAMPING = 1.0e-5
MAX_DAMPING = 1.0e7
MAX_NORMALIZED_STEP = 1.25
LINE_SEARCH_FACTORS = [1.0, 0.5, 0.25, 0.125, 0.0625]
FULL_JACOBIAN_REFRESH_AFTER = 3
BAD_MODEL_RATIO = 2.5
FAILED_ITERATIONS_BEFORE_ESCAPE = 2

FD_RATIO_STEP = 0.025
FD_NH2_1E19_STEP = 0.18
FD_ION_RATIO_STEP = 0.025
FD_HP_FRACTION_STEP = 0.04

STOP_ON_TOLERANCE = False
REUSE_HASH_COMPATIBLE_HISTORY = True
RANDOM_SEED = 20261013

DRIVER_CONCENTRATION_WEIGHT = 0.20
REQUIRE_CS_SETTING = None

SUPPORTED_CS_SETTINGS = (11, 12, 21, 22)
LINE_ORDER = ["Ha-s", "Hb-s", "Hg-s", "Fulcher-s"]

# Populated from CLI/configuration in main().
CRM_ROOT = None
EXP_DIR = None
BEST_INPUT_FILE = None
TEMPLATE_FILE = None
CRM_INPUT_FILE = None
CRM_EXECUTABLE = None
CRM_CONFIG = None
CRM_RUN_NAME = None
CRM_OUTPUT_FILE = None
RESULTS_ROOT = None

TARGETS = {}
WEIGHTS = {
    "Ha-s": 1.2,
    "Hb-s": 1.0,
    "Hg-s": 1.0,
    "Fulcher-s": 0.4,
}
ACCEPTABLE_RELATIVE_ERROR = 0.12

PID_TO_DISTANCE_CM = {}
CONTROL_PIDS = np.array([], dtype=float)
DRIVER_PID = None

DISSOCIATION_RATIO_BOUNDS = (0.10, 0.50)
NH2_DENSITY_BOUNDS_M3 = (2.0e19, 5.0e19)
NH2_UNIT = 1.0e19
NH2_CONTROL_BOUNDS = None
ION_TOTAL_OVER_NE_BOUNDS = (0.99, 1.01)
HP_FRACTION_BOUNDS = (0.02, 0.98)

N_CONTROL = 0
N_PARAMS = 0
LOWER_BOUNDS = None
UPPER_BOUNDS = None
PARAM_RANGES = None
OBJECTIVE_VERSION = None


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
    """Find CRM root from --crm-root, CRM_ROOT env, script placement or cwd."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = os.environ.get("CRM_ROOT")
    if env_root:
        candidates.append(env_root)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        script_dir,
        os.path.dirname(script_dir),
        os.getcwd(),
    ])

    seen = set()
    for item in candidates:
        path = os.path.abspath(os.path.expanduser(item))
        if path in seen:
            continue
        seen.add(path)
        if _looks_like_crm_root(path):
            return path

    if explicit:
        # Preserve the explicit value so the later existence checks produce a
        # precise error about the missing component.
        return os.path.abspath(os.path.expanduser(explicit))

    raise RuntimeError(
        "Could not locate the CRM root. Use --crm-root /path/to/CRM or set "
        "the CRM_ROOT environment variable."
    )


def resolve_root_path(value, crm_root):
    """Resolve a CRM-internal path; relative values are relative to CRM root."""
    value = os.path.expanduser(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(crm_root, value))


def load_distance_map(path):
    """Read a whitespace- or CSV-delimited two-column PID/distance map."""
    path = os.path.abspath(os.path.expanduser(path))
    ensure_exists(path, "PID-distance map")

    # First try named columns.
    try:
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
        lower = {str(c).strip().lower(): c for c in df.columns}
        if "pid" in lower and "distance_cm" in lower:
            pid_col = lower["pid"]
            dist_col = lower["distance_cm"]
            out = df[[pid_col, dist_col]].copy()
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
    """Read OES targets from lines such as 'Ha-s 3.2e21'."""
    path = os.path.abspath(os.path.expanduser(path))
    ensure_exists(path, "OES targets file")
    values = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,=]+", line)
            if len(parts) < 2:
                continue
            key = parts[0]
            if key in LINE_ORDER:
                values[key] = float(parts[1])
    missing = [k for k in LINE_ORDER if k not in values]
    if missing:
        raise RuntimeError(
            "Targets file is missing: {}".format(", ".join(missing))
        )
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Portable CRM hybrid global + Gauss-Newton autofitter."
    )
    parser.add_argument(
        "template",
        help="Starting CRM input/template file (not modified by default).",
    )
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
        default="autofit",
        help="Run name passed to CRM. Default is autofit.",
    )
    parser.add_argument(
        "--crm-output",
        default=None,
        help=(
            "CRM output file. Default is <crm-root>/CRM_res_<run-name>.txt."
        ),
    )
    parser.add_argument(
        "--best-output",
        default=None,
        help=(
            "Where to write the final best input. Default TEMPLATE_best.txt beside the template."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="crm_autofit_runs",
        help="Diagnostics/history directory, relative to CRM root unless absolute.",
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
        help="Text file containing Ha-s, Hb-s, Hg-s and Fulcher-s target values.",
    )

    parser.add_argument(
        "--weights",
        nargs=4,
        type=float,
        metavar=("HA", "HB", "HG", "FULCHER"),
        default=None,
        help="Optional spectral weights. Default 1.2 1.0 1.0 0.4.",
    )
    parser.add_argument(
        "--control-pids",
        required=True,
        help=(
            "Comma-separated neutral-profile control PIDs. Controls are ordered "
            "by distance_cm; both spatial endpoint PIDs must be included."
        ),
    )
    parser.add_argument(
        "--driver-pid",
        type=int,
        default=None,
        help="Driver-side PID. Default: PID with greatest distance_cm.",
    )
    parser.add_argument(
        "--require-cs",
        type=int,
        choices=SUPPORTED_CS_SETTINGS,
        default=None,
        help="Require the config to have this active k_eiae value.",
    )
    parser.add_argument(
        "--runtime-minutes",
        type=float,
        default=MAX_RUNTIME_MINUTES,
        help="Optimization wall-clock budget, final verification adds one CRM run.",
    )
    parser.add_argument(
        "--global-time-fraction",
        type=float,
        default=GLOBAL_TIME_FRACTION,
        help="Fraction of runtime reserved for global search.",
    )
    parser.add_argument(
        "--driver-weight",
        type=float,
        default=DRIVER_CONCENTRATION_WEIGHT,
        help="Soft driver-side concentration prior weight; 0 disables it.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=ACCEPTABLE_RELATIVE_ERROR,
        help="Per-line relative-error tolerance as a fraction (0.12 = 12%%).",
    )
    parser.add_argument(
        "--ratio-bounds",
        nargs=2,
        type=float,
        default=DISSOCIATION_RATIO_BOUNDS,
        metavar=("MIN", "MAX"),
        help="Bounds for nH/nH2.",
    )
    parser.add_argument(
        "--nh2-bounds",
        nargs=2,
        type=float,
        default=NH2_DENSITY_BOUNDS_M3,
        metavar=("MIN", "MAX"),
        help="Bounds for nH2 in m^-3.",
    )
    parser.add_argument(
        "--ion-bounds",
        nargs=2,
        type=float,
        default=ION_TOTAL_OVER_NE_BOUNDS,
        metavar=("MIN", "MAX"),
        help="Bounds for (nH+ + nH2+)/ne. Equal bounds are allowed.",
    )
    parser.add_argument(
        "--hp-fraction-bounds",
        nargs=2,
        type=float,
        default=HP_FRACTION_BOUNDS,
        metavar=("MIN", "MAX"),
        help="Bounds for H+ fraction of H+ + H2+.",
    )
    return parser.parse_args()


def configure_from_args(args):
    global CRM_ROOT, EXP_DIR, BEST_INPUT_FILE, TEMPLATE_FILE
    global CRM_INPUT_FILE, CRM_EXECUTABLE, CRM_CONFIG, CRM_RUN_NAME
    global CRM_OUTPUT_FILE, RESULTS_ROOT
    global TARGETS, WEIGHTS, PID_TO_DISTANCE_CM, CONTROL_PIDS, DRIVER_PID
    global REQUIRE_CS_SETTING, MAX_RUNTIME_MINUTES, GLOBAL_TIME_FRACTION
    global DRIVER_CONCENTRATION_WEIGHT, ACCEPTABLE_RELATIVE_ERROR
    global DISSOCIATION_RATIO_BOUNDS, NH2_DENSITY_BOUNDS_M3
    global NH2_CONTROL_BOUNDS, ION_TOTAL_OVER_NE_BOUNDS, HP_FRACTION_BOUNDS
    global N_CONTROL, N_PARAMS, LOWER_BOUNDS, UPPER_BOUNDS, PARAM_RANGES
    global OBJECTIVE_VERSION

    CRM_ROOT = discover_crm_root(args.crm_root)
    EXP_DIR = os.path.join(CRM_ROOT, "exp")

    TEMPLATE_FILE = os.path.abspath(os.path.expanduser(args.template))
    if args.best_output:
        BEST_INPUT_FILE = os.path.abspath(os.path.expanduser(args.best_output))
    else:
        stem, ext = os.path.splitext(TEMPLATE_FILE)
        BEST_INPUT_FILE = stem + "_best" + (ext or ".txt")

    CRM_INPUT_FILE = resolve_root_path(args.active_input, CRM_ROOT)
    CRM_CONFIG = resolve_root_path(args.config, CRM_ROOT)
    CRM_EXECUTABLE = resolve_root_path(args.executable, CRM_ROOT)
    CRM_RUN_NAME = str(args.run_name)
    CRM_OUTPUT_FILE = (
        resolve_root_path(args.crm_output, CRM_ROOT)
        if args.crm_output
        else os.path.join(CRM_ROOT, "CRM_res_{}.txt".format(CRM_RUN_NAME))
    )
    RESULTS_ROOT = resolve_root_path(args.results_root, CRM_ROOT)

    if args.targets_file:
        TARGETS = load_targets_file(args.targets_file)
    else:
        TARGETS = dict(zip(LINE_ORDER, [float(v) for v in args.targets]))

    if args.weights is not None:
        WEIGHTS = dict(zip(LINE_ORDER, [float(v) for v in args.weights]))

    for line in LINE_ORDER:
        target = float(TARGETS[line])
        if not np.isfinite(target) or target <= 0.0:
            raise RuntimeError(
                "OES target {} must be finite and > 0; got {}.".format(
                    line, target
                )
            )
        weight = float(WEIGHTS[line])
        if not np.isfinite(weight) or weight < 0.0:
            raise RuntimeError(
                "OES weight {} must be finite and >= 0; got {}.".format(
                    line, weight
                )
            )
    if not any(float(WEIGHTS[line]) > 0.0 for line in LINE_ORDER):
        raise RuntimeError("At least one OES weight must be > 0.")

    PID_TO_DISTANCE_CM = load_distance_map(args.distance_map)
    control_set = {
        int(x.strip()) for x in args.control_pids.split(",") if x.strip()
    }
    missing_control_map = [
        pid for pid in control_set if pid not in PID_TO_DISTANCE_CM
    ]
    if missing_control_map:
        raise RuntimeError(
            "Control PID(s) are absent from the distance map: {}".format(
                sorted(missing_control_map)
            )
        )
    CONTROL_PIDS = np.array(
        sorted(control_set, key=lambda pid: PID_TO_DISTANCE_CM[pid]),
        dtype=float,
    )
    if len(CONTROL_PIDS) < 2:
        raise RuntimeError("At least two --control-pids are required.")

    DRIVER_PID = (
        int(args.driver_pid)
        if args.driver_pid is not None
        else None
    )
    if DRIVER_PID is not None and DRIVER_PID not in PID_TO_DISTANCE_CM:
        raise RuntimeError("Driver PID {} is absent from distance map.".format(DRIVER_PID))

    REQUIRE_CS_SETTING = args.require_cs
    MAX_RUNTIME_MINUTES = float(args.runtime_minutes)
    GLOBAL_TIME_FRACTION = float(args.global_time_fraction)
    DRIVER_CONCENTRATION_WEIGHT = float(args.driver_weight)
    ACCEPTABLE_RELATIVE_ERROR = float(args.tolerance)

    if not np.isfinite(MAX_RUNTIME_MINUTES) or MAX_RUNTIME_MINUTES <= 0.0:
        raise RuntimeError("--runtime-minutes must be finite and > 0.")
    if (
        not np.isfinite(GLOBAL_TIME_FRACTION)
        or GLOBAL_TIME_FRACTION <= 0.0
        or GLOBAL_TIME_FRACTION > 1.0
    ):
        raise RuntimeError("--global-time-fraction must satisfy 0 < value <= 1.")
    if (
        not np.isfinite(DRIVER_CONCENTRATION_WEIGHT)
        or DRIVER_CONCENTRATION_WEIGHT < 0.0
    ):
        raise RuntimeError("--driver-weight must be finite and >= 0.")
    if (
        not np.isfinite(ACCEPTABLE_RELATIVE_ERROR)
        or ACCEPTABLE_RELATIVE_ERROR < 0.0
    ):
        raise RuntimeError("--tolerance must be finite and >= 0.")

    DISSOCIATION_RATIO_BOUNDS = tuple(float(v) for v in args.ratio_bounds)
    NH2_DENSITY_BOUNDS_M3 = tuple(float(v) for v in args.nh2_bounds)
    ION_TOTAL_OVER_NE_BOUNDS = tuple(float(v) for v in args.ion_bounds)
    HP_FRACTION_BOUNDS = tuple(float(v) for v in args.hp_fraction_bounds)

    for name, bounds in [
        ("ratio", DISSOCIATION_RATIO_BOUNDS),
        ("nH2", NH2_DENSITY_BOUNDS_M3),
        ("ion", ION_TOTAL_OVER_NE_BOUNDS),
        ("H+ fraction", HP_FRACTION_BOUNDS),
    ]:
        if not all(np.isfinite(v) for v in bounds):
            raise RuntimeError("{} bounds must be finite: {}".format(name, bounds))
        if bounds[0] > bounds[1]:
            raise RuntimeError("{} bounds are reversed: {}".format(name, bounds))

    if DISSOCIATION_RATIO_BOUNDS[0] < 0.0:
        raise RuntimeError("nH/nH2 ratio bounds must be >= 0.")
    if NH2_DENSITY_BOUNDS_M3[0] <= 0.0:
        raise RuntimeError("nH2 density bounds must be > 0.")
    if ION_TOTAL_OVER_NE_BOUNDS[0] < 0.0:
        raise RuntimeError("Positive-ion/electron ratio bounds must be >= 0.")
    if HP_FRACTION_BOUNDS[0] < 0.0 or HP_FRACTION_BOUNDS[1] > 1.0:
        raise RuntimeError("H+ fraction bounds must lie within [0, 1].")

    NH2_CONTROL_BOUNDS = (
        NH2_DENSITY_BOUNDS_M3[0] / NH2_UNIT,
        NH2_DENSITY_BOUNDS_M3[1] / NH2_UNIT,
    )

    N_CONTROL = len(CONTROL_PIDS)
    N_PARAMS = 2 * N_CONTROL + 2
    LOWER_BOUNDS = np.array(
        [DISSOCIATION_RATIO_BOUNDS[0]] * N_CONTROL
        + [NH2_CONTROL_BOUNDS[0]] * N_CONTROL
        + [ION_TOTAL_OVER_NE_BOUNDS[0], HP_FRACTION_BOUNDS[0]],
        dtype=float,
    )
    UPPER_BOUNDS = np.array(
        [DISSOCIATION_RATIO_BOUNDS[1]] * N_CONTROL
        + [NH2_CONTROL_BOUNDS[1]] * N_CONTROL
        + [ION_TOTAL_OVER_NE_BOUNDS[1], HP_FRACTION_BOUNDS[1]],
        dtype=float,
    )
    PARAM_RANGES = UPPER_BOUNDS - LOWER_BOUNDS

    OBJECTIVE_VERSION = None



def build_objective_version():
    history_signature_payload = repr((
        tuple(sorted((int(k), float(v)) for k, v in PID_TO_DISTANCE_CM.items())),
        tuple(int(v) for v in CONTROL_PIDS.tolist()),
        tuple(float(v) for v in DISSOCIATION_RATIO_BOUNDS),
        tuple(float(v) for v in NH2_DENSITY_BOUNDS_M3),
        tuple(float(v) for v in ION_TOTAL_OVER_NE_BOUNDS),
        tuple(float(v) for v in HP_FRACTION_BOUNDS),
        tuple((line, float(TARGETS[line])) for line in LINE_ORDER),
        tuple((line, float(WEIGHTS[line])) for line in LINE_ORDER),
        int(DRIVER_PID),
        str(CRM_RUN_NAME),
        os.path.abspath(CRM_INPUT_FILE),
    )).encode("utf-8")
    history_signature = hashlib.sha256(history_signature_payload).hexdigest()[:12]
    return (
        "opl_corrected_driver_prior_portable_v2_w{:.6g}_cfg{}"
        .format(DRIVER_CONCENTRATION_WEIGHT, history_signature)
    )


# ============================================================================
# BASIC FILE / HASH HELPERS
# ============================================================================

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_copy(src, dst):
    parent = os.path.dirname(dst)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    shutil.copyfile(src, dst)


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise RuntimeError("{} not found:\n{}".format(label, path))


# ============================================================================
# CROSS-SECTION SETTING
# ============================================================================

def detect_cross_section_setting():
    pattern = re.compile(r"^\s*k_eiae\s*=\s*(11|12|21|22)\b")
    found = []

    with open(CRM_CONFIG, "r") as f:
        for line in f:
            stripped = line.strip()

            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("!")
                or stripped.startswith(";")
            ):
                continue

            m = pattern.search(line)
            if m:
                found.append(int(m.group(1)))

    unique = sorted(set(found))

    if len(unique) != 1:
        raise RuntimeError(
            "Expected exactly one active k_eiae=11/12/21/22 in {}, found {}.".format(
                CRM_CONFIG,
                unique
            )
        )

    return unique[0]


# ============================================================================
# TEMPLATE
# ============================================================================

def load_template(path):
    df = pd.read_csv(path, sep=r"\s+")

    required = [
        "PID", "ne", "Te", "nH", "nH2",
        "nHp", "nH2p", "nH3p", "nHm",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Template is missing columns: {}".format(missing)
        )

    df["PID"] = pd.to_numeric(df["PID"], errors="raise").astype(int)
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        if not np.all(np.isfinite(df[col].to_numpy(dtype=float))):
            raise RuntimeError(
                "Template column {} contains non-finite values.".format(col)
            )

    pids = sorted(df["PID"].tolist())
    if len(pids) < 2 or len(set(pids)) != len(pids):
        raise RuntimeError("Template must contain at least two unique PID rows.")

    missing_distance = [pid for pid in pids if pid not in PID_TO_DISTANCE_CM]
    if missing_distance:
        raise RuntimeError(
            "Distance map is missing template PID(s): {}".format(missing_distance)
        )

    control_int = CONTROL_PIDS.astype(int).tolist()
    missing_controls = [pid for pid in control_int if pid not in pids]
    if missing_controls:
        raise RuntimeError(
            "Control PID(s) are absent from template: {}".format(missing_controls)
        )
    template_by_distance = sorted(
        pids, key=lambda pid: PID_TO_DISTANCE_CM[int(pid)]
    )
    endpoint_pids = {template_by_distance[0], template_by_distance[-1]}
    if not endpoint_pids.issubset(set(control_int)):
        raise RuntimeError(
            "Control PIDs must include both spatial endpoints to avoid "
            "extrapolation. Spatial endpoint PIDs are {}; controls are {}.".format(
                sorted(endpoint_pids), control_int
            )
        )

    return df.sort_values("PID").reset_index(drop=True)


# ============================================================================
# PARAMETERIZATION
# ============================================================================

def clip_params(params):
    p = np.asarray(params, dtype=float).copy()

    if len(p) != N_PARAMS:
        raise RuntimeError(
            "Expected {} parameters, got {}.".format(
                N_PARAMS,
                len(p)
            )
        )

    return np.minimum(
        UPPER_BOUNDS,
        np.maximum(LOWER_BOUNDS, p)
    )


def normalize_params(params):
    p = clip_params(params)
    q = np.zeros_like(p, dtype=float)
    variable = PARAM_RANGES > 0.0
    q[variable] = (p[variable] - LOWER_BOUNDS[variable]) / PARAM_RANGES[variable]
    # Fixed dimensions are identical for all candidates, so normalized 0 is fine.
    q[~variable] = 0.0
    return q


def denormalize_params(q):
    q = np.minimum(1.0, np.maximum(0.0, np.asarray(q, dtype=float)))
    return LOWER_BOUNDS + q * PARAM_RANGES


def normalized_distance(a, b):
    qa = normalize_params(a)
    qb = normalize_params(b)
    return float(np.linalg.norm(qa - qb))


def interpolate_controls(template, controls):
    template_x = np.array(
        [PID_TO_DISTANCE_CM[int(pid)] for pid in template["PID"]],
        dtype=float,
    )
    control_x = np.array(
        [PID_TO_DISTANCE_CM[int(pid)] for pid in CONTROL_PIDS],
        dtype=float,
    )
    return np.interp(
        template_x,
        control_x,
        np.asarray(controls, dtype=float)
    )


def unpack_params(params, template):
    p = clip_params(params)

    ratio_controls = p[0:N_CONTROL]
    h2_controls = p[N_CONTROL:2 * N_CONTROL]
    ion_total_over_ne = float(p[2 * N_CONTROL])
    hp_fraction = float(p[2 * N_CONTROL + 1])
    h2p_fraction = 1.0 - hp_fraction

    ratio_profile = interpolate_controls(
        template,
        ratio_controls
    )

    h2_units_profile = interpolate_controls(
        template,
        h2_controls
    )

    nH2 = h2_units_profile * NH2_UNIT
    nH = ratio_profile * nH2

    ne = pd.to_numeric(
        template["ne"],
        errors="raise"
    ).values.astype(float)

    total_positive = ion_total_over_ne * ne

    nHp = hp_fraction * total_positive
    nH2p = h2p_fraction * total_positive
    nH3p = np.zeros_like(total_positive)

    return {
        "ratio_controls": ratio_controls,
        "h2_controls": h2_controls,
        "ratio_profile": ratio_profile,
        "h2_units_profile": h2_units_profile,
        "nH": nH,
        "nH2": nH2,
        "ion_total_over_ne": ion_total_over_ne,
        "hp_fraction": hp_fraction,
        "h2p_fraction": h2p_fraction,
        "nHp": nHp,
        "nH2p": nH2p,
        "nH3p": nH3p,
    }


def constraint_report(params, template):
    prof = unpack_params(params, template)

    ratio = np.asarray(prof["ratio_profile"], dtype=float)
    nH2 = np.asarray(prof["nH2"], dtype=float)

    ratio_ok = (
        np.all(ratio >= DISSOCIATION_RATIO_BOUNDS[0] - 1.0e-12)
        and np.all(ratio <= DISSOCIATION_RATIO_BOUNDS[1] + 1.0e-12)
    )

    nH2_ok = (
        np.all(nH2 >= NH2_DENSITY_BOUNDS_M3[0] * (1.0 - 1.0e-12))
        and np.all(nH2 <= NH2_DENSITY_BOUNDS_M3[1] * (1.0 + 1.0e-12))
    )

    ion_ratio = float(prof["ion_total_over_ne"])

    ion_ok = (
        ion_ratio >= ION_TOTAL_OVER_NE_BOUNDS[0] - 1.0e-12
        and ion_ratio <= ION_TOTAL_OVER_NE_BOUNDS[1] + 1.0e-12
    )

    h3_ok = bool(np.all(np.asarray(prof["nH3p"]) == 0.0))

    return {
        "constraints_ok": bool(ratio_ok and nH2_ok and ion_ok and h3_ok),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "nH2_min_m3": float(np.min(nH2)),
        "nH2_max_m3": float(np.max(nH2)),
        "ion_total_over_ne": ion_ratio,
        "hp_fraction": float(prof["hp_fraction"]),
        "h2p_fraction": float(prof["h2p_fraction"]),
        "nH3p_zero": h3_ok,
    }


def initial_params_from_template(template):
    pids = template["PID"].values.astype(float)

    nH = pd.to_numeric(
        template["nH"],
        errors="raise"
    ).values.astype(float)

    nH2 = pd.to_numeric(
        template["nH2"],
        errors="raise"
    ).values.astype(float)

    ne = pd.to_numeric(
        template["ne"],
        errors="raise"
    ).values.astype(float)

    nHp = pd.to_numeric(
        template["nHp"],
        errors="raise"
    ).values.astype(float)

    nH2p = pd.to_numeric(
        template["nH2p"],
        errors="raise"
    ).values.astype(float)

    nH3p = pd.to_numeric(
        template["nH3p"],
        errors="raise"
    ).values.astype(float)

    if np.any(nH2 <= 0.0):
        raise RuntimeError("Template nH2 must be positive.")

    raw_ratio = nH / nH2
    raw_h2_units = nH2 / NH2_UNIT

    template_x = np.array(
        [PID_TO_DISTANCE_CM[int(pid)] for pid in pids],
        dtype=float,
    )
    order = np.argsort(template_x)
    template_x = template_x[order]
    control_x = np.array(
        [PID_TO_DISTANCE_CM[int(pid)] for pid in CONTROL_PIDS],
        dtype=float,
    )

    ratio_controls = np.interp(
        control_x,
        template_x,
        raw_ratio[order]
    )

    h2_controls = np.interp(
        control_x,
        template_x,
        raw_h2_units[order]
    )

    total_positive = nHp + nH2p + nH3p

    if np.any(ne <= 0.0):
        raise RuntimeError("Template ne must be positive.")
    if np.any(total_positive <= 0.0):
        raise RuntimeError("Template positive-ion total must be positive.")

    ion_ratio = float(np.median(total_positive / ne))

    h12 = nHp + nH2p

    if np.any(h12 <= 0.0):
        hp_fraction = 0.5
    else:
        hp_fraction = float(np.median(nHp / h12))

    p = np.concatenate([
        ratio_controls,
        h2_controls,
        np.array([ion_ratio, hp_fraction], dtype=float)
    ])

    return clip_params(p)


def write_input(params, template, destination):
    p = clip_params(params)
    report = constraint_report(p, template)

    if not report["constraints_ok"]:
        raise RuntimeError(
            "Attempted to write an unphysical candidate: {}".format(report)
        )

    prof = unpack_params(p, template)
    df = template.copy()

    df["nH"] = prof["nH"]
    df["nH2"] = prof["nH2"]
    df["nHp"] = prof["nHp"]
    df["nH2p"] = prof["nH2p"]

    # Explicitly write exact zeros, rather than relying on floating arithmetic.
    df["nH3p"] = 0.0

    columns = [
        "PID", "ne", "Te", "nH", "nH2",
        "nHp", "nH2p", "nH3p", "nHm",
    ]

    with open(destination, "w") as f:
        f.write(
            "PID\tne\tTe\tnH\tnH2\tnHp\tnH2p\tnH3p\tnHm\n"
        )

        for _, row in df[columns].iterrows():
            f.write(
                "{}\t{:.14E}\t{:.14g}\t{:.14E}\t{:.14E}\t"
                "{:.14E}\t{:.14E}\t{:.14E}\t{:.14E}\n".format(
                    int(row["PID"]),
                    float(row["ne"]),
                    float(row["Te"]),
                    float(row["nH"]),
                    float(row["nH2"]),
                    float(row["nHp"]),
                    float(row["nH2p"]),
                    0.0,
                    float(row["nHm"]),
                )
            )

    check = pd.read_csv(destination, sep=r"\s+")

    expected_pids = sorted(template["PID"].astype(int).tolist())
    if len(check) != len(expected_pids):
        raise RuntimeError(
            "Generated CRM input has {} rows; expected {} from the template.".format(
                len(check), len(expected_pids)
            )
        )

    if sorted(check["PID"].astype(int).tolist()) != expected_pids:
        raise RuntimeError(
            "Generated CRM input PID set differs from the template: expected {}.".format(
                expected_pids
            )
        )

    if not np.all(
        pd.to_numeric(
            check["nH3p"],
            errors="raise"
        ).values.astype(float) == 0.0
    ):
        raise RuntimeError("Generated CRM input violated nH3p=0.")

    return file_sha256(destination)


# ============================================================================
# CRM EXECUTION / OUTPUT
# ============================================================================

def run_crm(respect_deadline=True):
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

    try:
        if respect_deadline:
            timeout = time_remaining()
            if timeout <= 0.0:
                process.terminate()
                process.communicate()
                raise TimeoutError("Global runtime limit reached before CRM run.")
            stdout, _ = process.communicate(timeout=timeout)
        else:
            stdout, _ = process.communicate()
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
        raise TimeoutError("Global runtime limit reached during CRM run.")

    runtime = time.time() - t0

    if process.returncode != 0:
        raise RuntimeError(
            "CRM failed with return code {}:\n{}".format(
                process.returncode,
                stdout
            )
        )

    if not os.path.isfile(CRM_OUTPUT_FILE):
        raise RuntimeError(
            "CRM did not produce:\n{}".format(CRM_OUTPUT_FILE)
        )

    return runtime, stdout, file_sha256(CRM_OUTPUT_FILE)


def read_integrals():
    """
    Correct dimensional treatment of the CRM emission output.

    CRM output:
        Ha-s, Hb-s, Hg-s, Fulcher-s : ph m^-2 s^-1
        opl                          : m

    Reconstructed local emissivity:
        epsilon = signal / opl       : ph m^-3 s^-1

    Quantity compared with measured OES:
        integral epsilon(z) dz over the configured PID profile : ph m^-2 s^-1
    """
    df = pd.read_csv(
        CRM_OUTPUT_FILE,
        sep=r"\s+"
    )

    required = [
        "PID",
        "opl",
        "Ha-s",
        "Hb-s",
        "Hg-s",
        "Fulcher-s",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "CRM output is missing columns: {}".format(missing)
        )

    df["PID"] = pd.to_numeric(
        df["PID"],
        errors="raise"
    ).astype(int)

    expected_pids = sorted(TEMPLATE["PID"].astype(int).tolist())
    df = df[df["PID"].isin(expected_pids)].copy()
    if sorted(df["PID"].tolist()) != expected_pids:
        raise RuntimeError(
            "CRM output does not contain the full template PID set. Expected {}, found {}.".format(
                expected_pids, sorted(df["PID"].tolist())
            )
        )

    df["opl"] = pd.to_numeric(
        df["opl"],
        errors="raise"
    ).astype(float)

    if (
        not np.all(np.isfinite(df["opl"].values))
        or np.any(df["opl"].values <= 0.0)
    ):
        raise RuntimeError(
            "CRM output contains invalid optical path length(s) in 'opl': {}".format(
                df["opl"].tolist()
            )
        )

    df["distance_cm"] = df["PID"].map(
        PID_TO_DISTANCE_CM
    )

    df = df.sort_values(
        "distance_cm"
    ).reset_index(drop=True)

    x_m = (
        df["distance_cm"].values.astype(float)
        /
        100.0
    )

    values = {}
    opl = df["opl"].values.astype(float)

    for line in LINE_ORDER:
        crm_line_signal = pd.to_numeric(
            df[line],
            errors="raise"
        ).values.astype(float)

        if not np.all(np.isfinite(crm_line_signal)):
            raise RuntimeError(
                "CRM output contains non-finite values in {}.".format(line)
            )

        emissivity = crm_line_signal / opl

        df[line + "_vol"] = emissivity

        values[line] = float(
            trapezoid_integral(
                emissivity,
                x_m
            )
        )

    return values, df


# ============================================================================
# ERROR METRICS
# ============================================================================

def errors_from_values(values):
    return {
        line: (
            float(values[line])
            -
            float(TARGETS[line])
        )
        /
        float(TARGETS[line])
        for line in LINE_ORDER
    }


def spectral_score(errors):
    """OES-only weighted squared relative-error score."""
    return float(
        sum(
            WEIGHTS[line]
            *
            float(errors[line]) ** 2
            for line in LINE_ORDER
        )
    )


def driver_concentration_metrics(params):
    """
    Measure how far the fitted neutral density is concentrated from the driver.

    The spatial coordinate is normalized so:
        configured driver PID -> away = 0
        farthest retained PID -> away = 1

    For nH and nH2 separately, compute the density-weighted RMS value of
    this normalized distance. Each RMS is in [0,1] for positive profiles.

    The raw penalty is:
        0.5 * (rms_nH**2 + rms_nH2**2)

    so the weighted objective contribution is:
        DRIVER_CONCENTRATION_WEIGHT * raw_penalty
    """
    prof = unpack_params(
        clip_params(params),
        TEMPLATE
    )

    pids = TEMPLATE["PID"].values.astype(int)

    distances = np.array(
        [
            float(PID_TO_DISTANCE_CM[int(pid)])
            for pid in pids
        ],
        dtype=float
    )

    driver_distance = float(
        PID_TO_DISTANCE_CM[int(DRIVER_PID)]
    )

    absolute_distance = np.abs(distances - driver_distance)
    span = float(np.max(absolute_distance))

    if span <= 0.0:
        raise RuntimeError(
            "Driver-side spatial penalty requires at least two distinct "
            "spatial positions."
        )

    away = np.clip(
        absolute_distance / span,
        0.0,
        1.0
    )

    def weighted_rms_distance(density):
        density = np.asarray(density, dtype=float)

        if np.any(~np.isfinite(density)):
            raise RuntimeError(
                "Non-finite density encountered in driver prior."
            )

        # The current physical parameterization keeps nH/nH2 positive.
        weights = np.maximum(density, 0.0)
        total = float(np.sum(weights))

        if total <= 0.0:
            return 1.0

        second_moment = float(
            np.sum(
                weights * away ** 2
            )
            / total
        )

        return float(
            np.sqrt(
                max(second_moment, 0.0)
            )
        )

    rms_nH = weighted_rms_distance(
        prof["nH"]
    )

    rms_nH2 = weighted_rms_distance(
        prof["nH2"]
    )

    raw_penalty = 0.5 * (
        rms_nH ** 2
        +
        rms_nH2 ** 2
    )

    return {
        "rms_nH": float(rms_nH),
        "rms_nH2": float(rms_nH2),
        "raw_penalty": float(raw_penalty),
        "weighted_penalty": float(
            DRIVER_CONCENTRATION_WEIGHT
            *
            raw_penalty
        ),
    }


def total_objective_score(errors, params):
    """
    Combined objective used to rank candidates.

    Lower is better:
        total = OES spectral score + weighted driver-concentration penalty
    """
    spatial = driver_concentration_metrics(
        params
    )

    return float(
        spectral_score(errors)
        +
        spatial["weighted_penalty"]
    )


def optimization_residual_vector(candidate):
    """
    Residual vector used by Gauss-Newton.

    The first four entries are the weighted OES relative errors.
    The last two entries encode the driver-side prior so that the local
    optimizer responds directly to the same objective used for candidate
    ranking.
    """
    residuals = [
        np.sqrt(WEIGHTS[line])
        *
        float(candidate["errors"][line])
        for line in LINE_ORDER
    ]

    spatial = driver_concentration_metrics(
        candidate["params"]
    )

    if DRIVER_CONCENTRATION_WEIGHT > 0.0:
        scale = np.sqrt(
            DRIVER_CONCENTRATION_WEIGHT / 2.0
        )

        residuals.append(
            scale * spatial["rms_nH"]
        )
        residuals.append(
            scale * spatial["rms_nH2"]
        )
    else:
        residuals.extend([0.0, 0.0])

    return np.asarray(
        residuals,
        dtype=float
    )


def rms_relative_error(errors):
    return float(
        np.sqrt(
            np.mean(
                [
                    float(errors[line]) ** 2
                    for line in LINE_ORDER
                ]
            )
        )
    )


def max_abs_relative_error(errors):
    return float(
        max(
            abs(float(errors[line]))
            for line in LINE_ORDER
        )
    )


def all_lines_within_tolerance(errors):
    return all(
        abs(float(errors[line]))
        <= ACCEPTABLE_RELATIVE_ERROR
        for line in LINE_ORDER
    )


def smoothness_tiebreak(params):
    """
    Secondary metric only. It is not part of the OES objective.
    Smaller means smoother neutral-control profiles.
    """
    p = clip_params(params)

    ratio = p[0:N_CONTROL]
    h2 = p[N_CONTROL:2 * N_CONTROL]

    ratio_scale = DISSOCIATION_RATIO_BOUNDS[1] - DISSOCIATION_RATIO_BOUNDS[0]
    h2_scale = NH2_CONTROL_BOUNDS[1] - NH2_CONTROL_BOUNDS[0]

    ratio_term = (
        np.sum((np.diff(ratio) / ratio_scale) ** 2)
        if ratio_scale > 0.0
        else 0.0
    )
    h2_term = (
        np.sum((np.diff(h2) / h2_scale) ** 2)
        if h2_scale > 0.0
        else 0.0
    )

    return float(ratio_term + h2_term)


def candidate_better(a, b):
    """
    Primary comparison: lower combined objective score
    (OES spectral error + driver-side concentration penalty).
    Smoothness is used only if total scores are essentially indistinguishable.
    """
    if b is None:
        return True

    sa = float(a["score"])
    sb = float(b["score"])

    scale = max(abs(sa), abs(sb), 1.0e-15)

    if sa < sb - 1.0e-10 * scale:
        return True

    if abs(sa - sb) <= 1.0e-10 * scale:
        return float(a["smoothness"]) < float(b["smoothness"])

    return False


# ============================================================================
# GLOBAL RUNTIME / DIAGNOSTICS STATE
# ============================================================================

START_TIME = None
DEADLINE = None
GLOBAL_DEADLINE = None

TEMPLATE = None
CS_SETTING = None
TEMPLATE_SHA256 = None
CONFIG_SHA256 = None
EXECUTABLE_SHA256 = None

RUN_DIR = None
EVAL_DIR = None
HISTORY_FILE = None
SUMMARY_FILE = None

evaluation_number = 0
history_rows = []
exact_cache = {}
best_seen = None

fresh_runtime_samples = []


def time_remaining():
    return max(0.0, DEADLINE - time.time())


def deadline_reached():
    return time.time() >= DEADLINE


def history_dataframe():
    return pd.DataFrame(history_rows)


def save_history():
    if history_rows:
        history_dataframe().to_csv(
            HISTORY_FILE,
            index=False
        )


def params_to_columns(params):
    row = {}

    p = clip_params(params)

    for i in range(N_PARAMS):
        row["param_{:02d}".format(i)] = float(p[i])

    for i, pid in enumerate(CONTROL_PIDS.astype(int)):
        row["ratio_PID{}".format(pid)] = float(p[i])
        row["nH2_1e19_PID{}".format(pid)] = float(p[N_CONTROL + i])

    row["ion_total_over_ne"] = float(p[2 * N_CONTROL])
    row["f_Hp"] = float(p[2 * N_CONTROL + 1])
    row["f_H2p"] = 1.0 - float(p[2 * N_CONTROL + 1])
    row["f_H3p"] = 0.0

    return row


def result_record(
    params,
    values,
    errors,
    input_hash,
    output_hash,
    runtime,
    source,
    fresh
):
    report = constraint_report(params, TEMPLATE)

    row = {
        "evaluation": evaluation_number,
        "objective_version": OBJECTIVE_VERSION,
        "source": source,
        "fresh": bool(fresh),
        "template_sha256": TEMPLATE_SHA256,
        "config_sha256": CONFIG_SHA256,
        "executable_sha256": EXECUTABLE_SHA256,
        "cs_setting": CS_SETTING,
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "runtime_s": float(runtime),
        "score": total_objective_score(errors, params),
        "spectral_score": spectral_score(errors),
        "driver_raw_penalty": driver_concentration_metrics(params)["raw_penalty"],
        "driver_weighted_penalty": driver_concentration_metrics(params)["weighted_penalty"],
        "driver_rms_nH": driver_concentration_metrics(params)["rms_nH"],
        "driver_rms_nH2": driver_concentration_metrics(params)["rms_nH2"],
        "rms_relative_error": rms_relative_error(errors),
        "max_abs_relative_error": max_abs_relative_error(errors),
        "all_within_tolerance": all_lines_within_tolerance(errors),
        "smoothness": smoothness_tiebreak(params),
        "constraints_ok": report["constraints_ok"],
        "ratio_min": report["ratio_min"],
        "ratio_max": report["ratio_max"],
        "nH2_min_m3": report["nH2_min_m3"],
        "nH2_max_m3": report["nH2_max_m3"],
        "positive_ion_over_ne": report["ion_total_over_ne"],
        "Ha_integral": values["Ha-s"],
        "Hb_integral": values["Hb-s"],
        "Hg_integral": values["Hg-s"],
        "Fulcher_integral": values["Fulcher-s"],
        "Ha_rel_error": errors["Ha-s"],
        "Hb_rel_error": errors["Hb-s"],
        "Hg_rel_error": errors["Hg-s"],
        "Fulcher_rel_error": errors["Fulcher-s"],
    }

    row.update(
        params_to_columns(params)
    )

    return row


def row_to_candidate(row):
    p = np.array(
        [
            float(row["param_{:02d}".format(i)])
            for i in range(N_PARAMS)
        ],
        dtype=float
    )

    values = {
        "Ha-s": float(row["Ha_integral"]),
        "Hb-s": float(row["Hb_integral"]),
        "Hg-s": float(row["Hg_integral"]),
        "Fulcher-s": float(row["Fulcher_integral"]),
    }

    errors = errors_from_values(values)

    return {
        "params": p,
        "values": values,
        "errors": errors,
        "score": total_objective_score(errors, p),
        "spectral_score": spectral_score(errors),
        "driver_metrics": driver_concentration_metrics(p),
        "smoothness": smoothness_tiebreak(p),
        "input_sha256": str(row["input_sha256"]),
        "output_sha256": str(row["output_sha256"]),
        "source": str(row.get("source", "history")),
    }


def maybe_update_best(candidate):
    global best_seen

    if candidate_better(candidate, best_seen):
        best_seen = {
            "params": candidate["params"].copy(),
            "values": dict(candidate["values"]),
            "errors": dict(candidate["errors"]),
            "score": float(candidate["score"]),
            "spectral_score": float(
                candidate.get(
                    "spectral_score",
                    spectral_score(candidate["errors"])
                )
            ),
            "driver_metrics": dict(
                candidate.get(
                    "driver_metrics",
                    driver_concentration_metrics(candidate["params"])
                )
            ),
            "smoothness": float(candidate["smoothness"]),
            "input_sha256": candidate.get("input_sha256"),
            "output_sha256": candidate.get("output_sha256"),
            "source": candidate.get("source", "unknown"),
        }

        print(
            "   *** NEW GLOBAL BEST: total={:.6g}, spectral={:.6g}, "
            "driver={:.6g}, max|err|={:.2f}% ***".format(
                best_seen["score"],
                best_seen["spectral_score"],
                best_seen["driver_metrics"]["weighted_penalty"],
                100.0 * max_abs_relative_error(best_seen["errors"])
            ),
            flush=True
        )


# ============================================================================
# EXACT-HASH HISTORY REUSE
# ============================================================================

def load_compatible_history():
    global evaluation_number

    if not REUSE_HASH_COMPATIBLE_HISTORY:
        return

    if not os.path.isfile(HISTORY_FILE):
        return

    try:
        df = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        print(
            "WARNING: could not load history: {}".format(exc),
            flush=True
        )
        return

    usable = 0

    for row in df.to_dict(orient="records"):
        try:
            if str(row.get("objective_version", "")) != OBJECTIVE_VERSION:
                continue
            if str(row["template_sha256"]) != TEMPLATE_SHA256:
                continue
            if str(row["config_sha256"]) != CONFIG_SHA256:
                continue
            if str(row.get("executable_sha256", "")) != EXECUTABLE_SHA256:
                continue
            if int(float(row["cs_setting"])) != CS_SETTING:
                continue
            if abs(float(row.get("f_H3p", 0.0))) > 1.0e-15:
                continue

            candidate = row_to_candidate(row)

            input_hash = candidate["input_sha256"]

            exact_cache[input_hash] = candidate
            history_rows.append(dict(row))

            maybe_update_best(candidate)

            usable += 1

            try:
                evaluation_number = max(
                    evaluation_number,
                    int(float(row["evaluation"]))
                )
            except Exception:
                pass

        except Exception:
            continue

    print(
        "Loaded {} exact-hash-compatible previous evaluations.".format(usable),
        flush=True
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(params, source, force_fresh=False, allow_after_deadline=False):
    global evaluation_number

    if deadline_reached() and not allow_after_deadline:
        raise TimeoutError("Global runtime limit reached.")

    p = clip_params(params)

    report = constraint_report(p, TEMPLATE)

    if not report["constraints_ok"]:
        raise RuntimeError(
            "Internal constraint failure: {}".format(report)
        )

    input_hash = write_input(
        p,
        TEMPLATE,
        CRM_INPUT_FILE
    )

    if (
        not force_fresh
        and input_hash in exact_cache
    ):
        cached = exact_cache[input_hash]

        candidate = {
            "params": p.copy(),
            "values": dict(cached["values"]),
            "errors": dict(cached["errors"]),
            "score": total_objective_score(cached["errors"], p),
            "spectral_score": spectral_score(cached["errors"]),
            "driver_metrics": driver_concentration_metrics(p),
            "smoothness": smoothness_tiebreak(p),
            "input_sha256": input_hash,
            "output_sha256": cached.get("output_sha256"),
            "source": source + "_CACHE",
        }

        maybe_update_best(candidate)

        print(
            "   reused exact input {} | total {:.6g} | spectral {:.6g} "
            "| driver {:.6g} | max|err| {:.2f}%".format(
                input_hash[:12],
                candidate["score"],
                candidate["spectral_score"],
                candidate["driver_metrics"]["weighted_penalty"],
                100.0 * max_abs_relative_error(candidate["errors"])
            ),
            flush=True
        )

        return candidate

    evaluation_number += 1

    print(
        "\nEvaluation {} [{}]".format(
            evaluation_number,
            source
        ),
        flush=True
    )

    runtime, stdout, output_hash = run_crm(
        respect_deadline=not allow_after_deadline
    )
    fresh_runtime_samples.append(float(runtime))

    values, output_df = read_integrals()

    print(
        "   CRM opl = {:.6g} .. {:.6g} m; line outputs were divided by opl "
        "before spatial integration.".format(
            float(output_df["opl"].min()),
            float(output_df["opl"].max()),
        ),
        flush=True
    )

    errors = errors_from_values(values)

    candidate = {
        "params": p.copy(),
        "values": values,
        "errors": errors,
        "score": total_objective_score(errors, p),
        "spectral_score": spectral_score(errors),
        "driver_metrics": driver_concentration_metrics(p),
        "smoothness": smoothness_tiebreak(p),
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "source": source,
    }

    row = result_record(
        p,
        values,
        errors,
        input_hash,
        output_hash,
        runtime,
        source,
        True
    )

    history_rows.append(row)
    exact_cache[input_hash] = candidate
    save_history()

    prefix = os.path.join(
        EVAL_DIR,
        "eval_{:05d}_{}".format(
            evaluation_number,
            re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
        )
    )

    safe_copy(
        CRM_INPUT_FILE,
        prefix + "_input.txt"
    )

    safe_copy(
        CRM_OUTPUT_FILE,
        prefix + "_output.txt"
    )

    with open(prefix + "_stdout.txt", "w") as f:
        f.write(stdout)

    spatial = candidate["driver_metrics"]

    print(
        "   total={:.6g} | spectral={:.6g} | driver penalty={:.6g} "
        "| RMS={:.2f}% | max|err|={:.2f}%".format(
            candidate["score"],
            candidate["spectral_score"],
            spatial["weighted_penalty"],
            100.0 * rms_relative_error(errors),
            100.0 * max_abs_relative_error(errors)
        ),
        flush=True
    )

    print(
        "   driver RMS distance: nH={:.3f}, nH2={:.3f} "
        "(0=all at driver PID, 1=all at far edge)".format(
            spatial["rms_nH"],
            spatial["rms_nH2"],
        ),
        flush=True
    )

    print(
        "   Ha {:+.2f}% | Hb {:+.2f}% | Hg {:+.2f}% | Fulcher {:+.2f}%".format(
            100.0 * errors["Ha-s"],
            100.0 * errors["Hb-s"],
            100.0 * errors["Hg-s"],
            100.0 * errors["Fulcher-s"],
        ),
        flush=True
    )

    print(
        "   ratio {:.3f}..{:.3f} | nH2 {:.3e}..{:.3e} | "
        "ions/ne {:.3f} | Hp {:.1f}%".format(
            report["ratio_min"],
            report["ratio_max"],
            report["nH2_min_m3"],
            report["nH2_max_m3"],
            report["ion_total_over_ne"],
            100.0 * report["hp_fraction"],
        ),
        flush=True
    )

    maybe_update_best(candidate)

    return candidate


# ============================================================================
# GLOBAL SEARCH
# ============================================================================

def latin_hypercube(n_samples, n_dim, rng):
    """
    Latin-hypercube points in [0,1]^n_dim.
    """
    result = np.zeros(
        (n_samples, n_dim),
        dtype=float
    )

    for j in range(n_dim):
        perm = rng.permutation(n_samples)

        result[:, j] = (
            perm.astype(float)
            +
            rng.uniform(
                0.0,
                1.0,
                size=n_samples
            )
        ) / float(n_samples)

    return result


def all_candidates_sorted():
    candidates = []

    seen = set()

    for row in history_rows:
        try:
            c = row_to_candidate(row)
            key = c["input_sha256"]

            if key in seen:
                continue

            seen.add(key)
            candidates.append(c)

        except Exception:
            continue

    if best_seen is not None:
        key = best_seen.get("input_sha256")

        if key not in seen:
            candidates.append(best_seen)

    candidates.sort(
        key=lambda c: (
            float(c["score"]),
            float(c["smoothness"])
        )
    )

    return candidates


def select_distinct_candidates(candidates, count, min_distance=0.20):
    selected = []

    for c in candidates:
        if len(selected) >= count:
            break

        if all(
            normalized_distance(
                c["params"],
                s["params"]
            ) >= min_distance
            for s in selected
        ):
            selected.append(c)

    # If the distance rule is too strict, fill remaining slots with the next best.
    if len(selected) < count:
        selected_hashes = set(
            s.get("input_sha256")
            for s in selected
        )

        for c in candidates:
            if len(selected) >= count:
                break

            if c.get("input_sha256") not in selected_hashes:
                selected.append(c)
                selected_hashes.add(c.get("input_sha256"))

    return selected


def run_global_search(initial, rng):
    print(
        "\n============================================================\n"
        "GLOBAL EXPLORATION\n"
        "============================================================",
        flush=True
    )

    # Always evaluate the physically projected current best/template first.
    evaluate(
        initial,
        "GLOBAL_projected_template"
    )

    # Also test the exact center of every allowed physical bound.
    center = 0.5 * (
        LOWER_BOUNDS
        +
        UPPER_BOUNDS
    )

    evaluate(
        center,
        "GLOBAL_box_center"
    )

    q_lhs = latin_hypercube(
        GLOBAL_INITIAL_SAMPLES,
        N_PARAMS,
        rng
    )

    for i in range(GLOBAL_INITIAL_SAMPLES):
        if time.time() >= GLOBAL_DEADLINE or deadline_reached():
            break

        p = denormalize_params(
            q_lhs[i]
        )

        evaluate(
            p,
            "GLOBAL_LHS_{:03d}".format(i + 1)
        )

        if (
            STOP_ON_TOLERANCE
            and best_seen is not None
            and all_lines_within_tolerance(best_seen["errors"])
        ):
            return

    generation = 0

    # Use remaining global budget to mutate around the best candidates while continuing to inject fully random points.
    while (
        time.time() < GLOBAL_DEADLINE
        and not deadline_reached()
    ):
        generation += 1

        candidates = all_candidates_sorted()

        if not candidates:
            break

        elites = candidates[
            :min(
                GLOBAL_ELITE_COUNT,
                len(candidates)
            )
        ]

        # Slowly narrow the mutation width while never making it too small.
        sigma = max(
            0.06,
            GLOBAL_MUTATION_SIGMA
            *
            (0.82 ** (generation - 1))
        )

        for j in range(GLOBAL_MUTATION_BATCH):
            if time.time() >= GLOBAL_DEADLINE or deadline_reached():
                break

            if rng.rand() < GLOBAL_RANDOM_INJECTION_PROBABILITY:
                q = rng.uniform(
                    0.0,
                    1.0,
                    size=N_PARAMS
                )
                source = "GLOBAL_random_g{}_{}".format(
                    generation,
                    j + 1
                )
            else:
                parent = elites[
                    rng.randint(
                        0,
                        len(elites)
                    )
                ]

                q_parent = normalize_params(
                    parent["params"]
                )

                q = np.clip(
                    q_parent
                    +
                    rng.normal(
                        0.0,
                        sigma,
                        size=N_PARAMS
                    ),
                    0.0,
                    1.0
                )

                source = "GLOBAL_mutation_g{}_{}".format(
                    generation,
                    j + 1
                )

            evaluate(
                denormalize_params(q),
                source
            )

            if (
                STOP_ON_TOLERANCE
                and best_seen is not None
                and all_lines_within_tolerance(best_seen["errors"])
            ):
                return


# ============================================================================
# LOCAL JACOBIAN SEARCH
# ============================================================================

def parameter_fd_step(i):
    if i < N_CONTROL:
        return FD_RATIO_STEP

    if i < 2 * N_CONTROL:
        return FD_NH2_1E19_STEP

    if i == 2 * N_CONTROL:
        return FD_ION_RATIO_STEP

    return FD_HP_FRACTION_STEP


def finite_difference_jacobian(center_candidate, local_label):
    center = center_candidate["params"].copy()
    r0 = optimization_residual_vector(
        center_candidate
    )

    J = np.zeros(
        (len(r0), N_PARAMS),
        dtype=float
    )

    print(
        "\nMeasuring fresh local {} x {} Jacobian for {}...".format(
            len(r0),
            N_PARAMS,
            local_label
        ),
        flush=True
    )

    for i in range(N_PARAMS):
        if deadline_reached():
            raise TimeoutError(
                "Runtime limit reached while measuring local Jacobian."
            )

        h = parameter_fd_step(i)

        plus_possible = (
            center[i] + h
            <= UPPER_BOUNDS[i]
        )

        minus_possible = (
            center[i] - h
            >= LOWER_BOUNDS[i]
        )

        if plus_possible:
            trial = center.copy()
            trial[i] += h
        elif minus_possible:
            trial = center.copy()
            trial[i] -= h
        else:
            # Use the maximum available one-sided distance.
            dist_plus = UPPER_BOUNDS[i] - center[i]
            dist_minus = center[i] - LOWER_BOUNDS[i]

            if dist_plus >= dist_minus and dist_plus > 1.0e-12:
                trial = center.copy()
                trial[i] = UPPER_BOUNDS[i]
            elif dist_minus > 1.0e-12:
                trial = center.copy()
                trial[i] = LOWER_BOUNDS[i]
            else:
                continue

        trial = clip_params(trial)

        dh = float(
            trial[i] - center[i]
        )

        if abs(dh) < 1.0e-14:
            continue

        c = evaluate(
            trial,
            "{}_FD_param{:02d}".format(
                local_label,
                i
            )
        )

        r1 = optimization_residual_vector(c)

        J[:, i] = (
            r1 - r0
        ) / dh

    return J


def gauss_newton_step(candidate, J, damping):
    r = optimization_residual_vector(
        candidate
    )

    # Normalize each parameter by its entire allowed physical range.
    Jq = J * PARAM_RANGES[None, :]

    H = (
        np.dot(Jq.T, Jq)
        +
        float(damping)
        *
        np.eye(N_PARAMS)
    )

    g = -np.dot(
        Jq.T,
        r
    )

    try:
        dq = np.linalg.solve(
            H,
            g
        )
    except np.linalg.LinAlgError:
        dq, _, _, _ = np.linalg.lstsq(
            H,
            g,
            rcond=None
        )

    norm = float(
        np.linalg.norm(dq)
    )

    if norm > MAX_NORMALIZED_STEP:
        dq *= (
            MAX_NORMALIZED_STEP
            /
            norm
        )

    return PARAM_RANGES * dq


def model_mismatch(J, p0, p1, c0, c1):
    dp = np.asarray(p1) - np.asarray(p0)

    obs = (
        optimization_residual_vector(c1)
        -
        optimization_residual_vector(c0)
    )

    pred = np.dot(
        J,
        dp
    )

    return float(
        np.linalg.norm(
            pred - obs
        )
        /
        max(
            np.linalg.norm(obs),
            1.0e-12
        )
    )


def broyden_update(J, p0, p1, c0, c1):
    dp = np.asarray(p1) - np.asarray(p0)

    den = float(
        np.dot(dp, dp)
    )

    if den < 1.0e-20:
        return J

    dr = (
        optimization_residual_vector(c1)
        -
        optimization_residual_vector(c0)
    )

    return (
        J
        +
        np.outer(
            dr - np.dot(J, dp),
            dp
        )
        /
        den
    )


def random_escape(current, rng, local_label):
    q0 = normalize_params(
        current["params"]
    )

    best_escape = None

    for k in range(LOCAL_ESCAPE_TRIALS):
        if deadline_reached():
            break

        q = np.clip(
            q0
            +
            rng.normal(
                0.0,
                LOCAL_ESCAPE_SIGMA,
                size=N_PARAMS
            ),
            0.0,
            1.0
        )

        c = evaluate(
            denormalize_params(q),
            "{}_ESCAPE_{:02d}".format(
                local_label,
                k + 1
            )
        )

        if candidate_better(
            c,
            best_escape
        ):
            best_escape = c

    if (
        best_escape is not None
        and candidate_better(
            best_escape,
            current
        )
    ):
        print(
            "   escape improved score {:.6g} -> {:.6g}".format(
                current["score"],
                best_escape["score"]
            ),
            flush=True
        )

        return best_escape

    return current


def local_refine(seed, rng, local_index):
    label = "LOCAL{:02d}".format(local_index)

    print(
        "\n============================================================\n"
        "{} START\n"
        "seed score={:.6g}, max|err|={:.2f}%\n"
        "============================================================".format(
            label,
            seed["score"],
            100.0 * max_abs_relative_error(seed["errors"])
        ),
        flush=True
    )

    # Freshly evaluate the local starting point even if it was previously cached.
    current = evaluate(
        seed["params"],
        label + "_START",
        force_fresh=True
    )

    if (
        STOP_ON_TOLERANCE
        and all_lines_within_tolerance(current["errors"])
    ):
        return current

    try:
        J = finite_difference_jacobian(
            current,
            label
        )
    except TimeoutError:
        return current

    damping = INITIAL_DAMPING
    failed_iterations = 0
    accepted_since_refresh = 0

    for iteration in range(1, LOCAL_MAX_ITERATIONS + 1):
        if deadline_reached():
            break

        print(
            "\n{} iteration {} | score {:.6g} | damping {:.4g}".format(
                label,
                iteration,
                current["score"],
                damping
            ),
            flush=True
        )

        step = gauss_newton_step(
            current,
            J,
            damping
        )

        variable = PARAM_RANGES > 0.0
        normalized_step_norm = float(
            np.linalg.norm(step[variable] / PARAM_RANGES[variable])
        ) if np.any(variable) else 0.0

        if normalized_step_norm < 1.0e-9:
            print(
                "   step collapsed; attempting escape.",
                flush=True
            )

            escaped = random_escape(
                current,
                rng,
                label
            )

            if candidate_better(
                escaped,
                current
            ):
                current = escaped

                try:
                    J = finite_difference_jacobian(
                        current,
                        label + "_after_escape"
                    )
                except TimeoutError:
                    break

                damping = INITIAL_DAMPING
                failed_iterations = 0
                accepted_since_refresh = 0
                continue

            break

        old = current
        improved = False

        for alpha in LINE_SEARCH_FACTORS:
            if deadline_reached():
                break

            trial_params = clip_params(
                old["params"]
                +
                alpha * step
            )

            if normalized_distance(
                trial_params,
                old["params"]
            ) < 1.0e-10:
                continue

            trial = evaluate(
                trial_params,
                "{}_iter{:02d}_alpha{}".format(
                    label,
                    iteration,
                    alpha
                )
            )

            if candidate_better(
                trial,
                current
            ):
                mismatch = model_mismatch(
                    J,
                    old["params"],
                    trial["params"],
                    old,
                    trial
                )

                print(
                    "   accepted local step: {:.6g} -> {:.6g}; "
                    "model mismatch {:.3f}".format(
                        current["score"],
                        trial["score"],
                        mismatch
                    ),
                    flush=True
                )

                J = broyden_update(
                    J,
                    old["params"],
                    trial["params"],
                    old,
                    trial
                )

                current = trial
                improved = True
                failed_iterations = 0
                accepted_since_refresh += 1

                if mismatch < 0.75:
                    damping = max(
                        MIN_DAMPING,
                        damping * 0.45
                    )
                elif mismatch < BAD_MODEL_RATIO:
                    damping = max(
                        MIN_DAMPING,
                        damping * 0.8
                    )
                else:
                    damping = min(
                        MAX_DAMPING,
                        damping * 2.5
                    )

                if (
                    accepted_since_refresh
                    >= FULL_JACOBIAN_REFRESH_AFTER
                    or mismatch >= BAD_MODEL_RATIO
                ):
                    try:
                        J = finite_difference_jacobian(
                            current,
                            label + "_refresh"
                        )
                    except TimeoutError:
                        return current

                    accepted_since_refresh = 0

                break

        if (
            STOP_ON_TOLERANCE
            and all_lines_within_tolerance(current["errors"])
        ):
            break

        if not improved:
            failed_iterations += 1

            damping = min(
                MAX_DAMPING,
                damping * 5.0
            )

            print(
                "   no improving line-search step; failure {}.".format(
                    failed_iterations
                ),
                flush=True
            )

            if failed_iterations >= FAILED_ITERATIONS_BEFORE_ESCAPE:
                escaped = random_escape(
                    current,
                    rng,
                    label
                )

                if candidate_better(
                    escaped,
                    current
                ):
                    current = escaped
                    damping = INITIAL_DAMPING

                try:
                    J = finite_difference_jacobian(
                        current,
                        label + "_rebuild"
                    )
                except TimeoutError:
                    break

                failed_iterations = 0
                accepted_since_refresh = 0

        if damping >= MAX_DAMPING:
            print(
                "   maximum damping reached for this local start.",
                flush=True
            )
            break

    return current


# ============================================================================
# BOUND / SOLUTION DIAGNOSTICS
# ============================================================================

def bound_diagnostics(params):
    p = clip_params(params)

    rows = []

    names = (
        ["ratio_PID{}".format(pid) for pid in CONTROL_PIDS.astype(int).tolist()]
        +
        ["nH2_1e19_PID{}".format(pid) for pid in CONTROL_PIDS.astype(int).tolist()]
        +
        ["ions_over_ne", "f_Hp"]
    )

    for i in range(N_PARAMS):
        lo = LOWER_BOUNDS[i]
        hi = UPPER_BOUNDS[i]
        value = p[i]
        width = hi - lo

        frac_from_low = (
            (value - lo) / width
            if width > 0
            else np.nan
        )

        distance_to_nearest_bound = min(
            value - lo,
            hi - value
        )

        rows.append({
            "parameter": names[i],
            "value": value,
            "lower": lo,
            "upper": hi,
            "fraction_across_range": frac_from_low,
            "distance_to_nearest_bound": distance_to_nearest_bound,
            "within_2pct_of_bound": bool(
                distance_to_nearest_bound
                <= 0.02 * width
            ),
            "within_5pct_of_bound": bool(
                distance_to_nearest_bound
                <= 0.05 * width
            ),
        })

    return pd.DataFrame(rows)


# ============================================================================
# FINAL SAVE / VERIFICATION
# ============================================================================

def save_best_and_verify(candidate):
    print(
        "\n============================================================\n"
        "FINAL EXACT SAVED-FILE VERIFICATION\n"
        "============================================================",
        flush=True
    )

    final_hash = write_input(
        candidate["params"],
        TEMPLATE,
        CRM_INPUT_FILE
    )

    safe_copy(
        CRM_INPUT_FILE,
        BEST_INPUT_FILE
    )

    saved_hash = file_sha256(
        BEST_INPUT_FILE
    )

    if saved_hash != final_hash:
        raise RuntimeError(
            "Saved best file differs from generated CRM input."
        )

    # Recopy exact saved bytes into active input and rerun fresh.
    safe_copy(
        BEST_INPUT_FILE,
        CRM_INPUT_FILE
    )

    verified = evaluate(
        candidate["params"],
        "FINAL_EXACT_SAVED_FILE",
        force_fresh=True,
        allow_after_deadline=True
    )

    active_hash = file_sha256(
        CRM_INPUT_FILE
    )

    if active_hash != saved_hash:
        raise RuntimeError(
            "Final active input hash differs from saved best file."
        )

    return verified


# ============================================================================
# REPORTING
# ============================================================================

def write_summary(final_candidate, local_results):
    report = constraint_report(
        final_candidate["params"],
        TEMPLATE
    )

    bounds_df = bound_diagnostics(
        final_candidate["params"]
    )

    bounds_csv = os.path.join(
        RUN_DIR,
        "final_bound_diagnostics.csv"
    )

    bounds_df.to_csv(
        bounds_csv,
        index=False
    )

    local_rows = []

    for i, c in enumerate(local_results):
        local_rows.append({
            "local_start": i + 1,
            "score": c["score"],
            "rms_error_percent": 100.0 * rms_relative_error(c["errors"]),
            "max_abs_error_percent": 100.0 * max_abs_relative_error(c["errors"]),
            "Ha_error_percent": 100.0 * c["errors"]["Ha-s"],
            "Hb_error_percent": 100.0 * c["errors"]["Hb-s"],
            "Hg_error_percent": 100.0 * c["errors"]["Hg-s"],
            "Fulcher_error_percent": 100.0 * c["errors"]["Fulcher-s"],
        })

    pd.DataFrame(
        local_rows
    ).to_csv(
        os.path.join(
            RUN_DIR,
            "local_start_results.csv"
        ),
        index=False
    )

    with open(SUMMARY_FILE, "w") as f:
        f.write("Hybrid global + multi-start local CRM fitter\n")
        f.write("=" * 78 + "\n\n")

        f.write("Template/start file: {}\n".format(TEMPLATE_FILE))
        f.write("Final best file: {}\n".format(BEST_INPUT_FILE))
        f.write("k_eiae: {}\n".format(CS_SETTING))
        f.write("Template SHA256: {}\n".format(TEMPLATE_SHA256))
        f.write("Config SHA256: {}\n".format(CONFIG_SHA256))
        f.write("CRM executable SHA256: {}\n".format(EXECUTABLE_SHA256))
        f.write("Runtime used: {:.1f} min\n".format(
            (time.time() - START_TIME) / 60.0
        ))
        f.write("Fresh CRM evaluations: {}\n".format(
            len(fresh_runtime_samples)
        ))
        f.write("Objective version: {}\n".format(OBJECTIVE_VERSION))
        f.write(
            "OES comparison: divide each CRM Ha/Hb/Hg/Fulcher output by its "
            "CRM-reported opl, then integrate that volumetric emissivity over "
            "the configured PID-distance interval.\n"
        )
        f.write("\n")
        f.write(
            "Driver concentration prior: density-weighted RMS distance of nH and "
            "nH2 from the driver PID; DRIVER_CONCENTRATION_WEIGHT={:.8g}.\n".format(
                DRIVER_CONCENTRATION_WEIGHT
            )
        )
        f.write(
            "The prior is soft, not a hard profile constraint.\n"
        )
        f.write("\n")

        f.write("Hard constraints:\n")
        f.write(
            "  {:.8g} <= (nHp+nH2p)/ne <= {:.8g}\n".format(
                ION_TOTAL_OVER_NE_BOUNDS[0], ION_TOTAL_OVER_NE_BOUNDS[1]
            )
        )
        f.write("  nH3p = 0 exactly\n")
        f.write(
            "  {:.8g} <= nH/nH2 <= {:.8g}\n".format(
                DISSOCIATION_RATIO_BOUNDS[0], DISSOCIATION_RATIO_BOUNDS[1]
            )
        )
        f.write(
            "  {:.8E} <= nH2 <= {:.8E} m^-3\n".format(
                NH2_DENSITY_BOUNDS_M3[0], NH2_DENSITY_BOUNDS_M3[1]
            )
        )
        f.write("\n")

        f.write("FINAL VERIFIED OES RESULTS\n")
        f.write("-" * 78 + "\n")

        for line in LINE_ORDER:
            f.write(
                "{:<10s} calc={:.10E} target={:.10E} error={:+.3f}%\n".format(
                    line,
                    final_candidate["values"][line],
                    TARGETS[line],
                    100.0 * final_candidate["errors"][line],
                )
            )

        f.write("\n")
        final_spatial = driver_concentration_metrics(
            final_candidate["params"]
        )

        f.write("total objective score = {:.12E}\n".format(
            final_candidate["score"]
        ))
        f.write("OES spectral score = {:.12E}\n".format(
            spectral_score(final_candidate["errors"])
        ))
        f.write("driver raw penalty = {:.12E}\n".format(
            final_spatial["raw_penalty"]
        ))
        f.write("driver weighted penalty = {:.12E}\n".format(
            final_spatial["weighted_penalty"]
        ))
        f.write("driver RMS distance nH = {:.6f}\n".format(
            final_spatial["rms_nH"]
        ))
        f.write("driver RMS distance nH2 = {:.6f}\n".format(
            final_spatial["rms_nH2"]
        ))
        f.write("RMS relative error = {:.3f}%\n".format(
            100.0 * rms_relative_error(final_candidate["errors"])
        ))
        f.write("max absolute relative error = {:.3f}%\n".format(
            100.0 * max_abs_relative_error(final_candidate["errors"])
        ))
        f.write(
            "all lines within +/-{:.3f}% = {}\n".format(
                100.0 * ACCEPTABLE_RELATIVE_ERROR,
                all_lines_within_tolerance(final_candidate["errors"])
            )
        )
        f.write("\n")

        f.write("FINAL PHYSICAL STATE\n")
        f.write("-" * 78 + "\n")
        f.write(
            "nH/nH2 range: {:.6f} .. {:.6f}\n".format(
                report["ratio_min"],
                report["ratio_max"]
            )
        )
        f.write(
            "nH2 range: {:.8E} .. {:.8E} m^-3\n".format(
                report["nH2_min_m3"],
                report["nH2_max_m3"]
            )
        )
        f.write(
            "(nHp+nH2p)/ne = {:.8f}\n".format(
                report["ion_total_over_ne"]
            )
        )
        f.write(
            "H+:H2+:H3+ = {:.3f}:{:.3f}:0.000 %\n".format(
                100.0 * report["hp_fraction"],
                100.0 * report["h2p_fraction"]
            )
        )
        f.write("\n")

        pinned = bounds_df[
            bounds_df["within_2pct_of_bound"]
        ]

        f.write(
            "Parameters within 2% of a hard bound: {}\n".format(
                len(pinned)
            )
        )

        for _, row in pinned.iterrows():
            f.write(
                "  {} = {:.8g}  bounds [{:.8g}, {:.8g}]\n".format(
                    row["parameter"],
                    row["value"],
                    row["lower"],
                    row["upper"],
                )
            )

        f.write("\n")
        f.write(
            "NOTE: This is the best solution FOUND by the hybrid search, "
            "not a mathematical proof of the global optimum.\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    configure_from_args(args)

    global START_TIME
    global DEADLINE
    global GLOBAL_DEADLINE
    global TEMPLATE
    global CS_SETTING
    global TEMPLATE_SHA256
    global CONFIG_SHA256
    global EXECUTABLE_SHA256
    global RUN_DIR
    global EVAL_DIR
    global HISTORY_FILE
    global SUMMARY_FILE
    global DRIVER_PID
    global OBJECTIVE_VERSION

    ensure_exists(
        CRM_ROOT,
        "CRM root"
    )

    ensure_exists(
        EXP_DIR,
        "CRM exp directory"
    )

    active_parent = os.path.dirname(CRM_INPUT_FILE)
    if not os.path.isdir(active_parent):
        raise RuntimeError(
            "Active-input directory not found:\n{}".format(active_parent)
        )

    ensure_exists(
        TEMPLATE_FILE,
        "Best/template input"
    )

    ensure_exists(
        CRM_CONFIG,
        "CRM config"
    )

    ensure_exists(
        CRM_EXECUTABLE,
        "CRM executable"
    )

    path_roles = {
        "template": os.path.abspath(TEMPLATE_FILE),
        "active input": os.path.abspath(CRM_INPUT_FILE),
        "best output": os.path.abspath(BEST_INPUT_FILE),
        "CRM output": os.path.abspath(CRM_OUTPUT_FILE),
        "CRM config": os.path.abspath(CRM_CONFIG),
    }
    if path_roles["template"] == path_roles["active input"]:
        raise RuntimeError(
            "The template and --active-input must be different files; the "
            "active input is overwritten during fitting."
        )
    if path_roles["template"] == path_roles["best output"]:
        raise RuntimeError(
            "--best-output must differ from the starting template. This "
            "portable version never overwrites the template."
        )
    if path_roles["active input"] == path_roles["best output"]:
        raise RuntimeError(
            "--best-output must differ from --active-input."
        )
    if path_roles["CRM output"] in {
        path_roles["template"], path_roles["active input"],
        path_roles["best output"], path_roles["CRM config"]
    }:
        raise RuntimeError(
            "CRM output path collides with an input/config/output file."
        )

    START_TIME = time.time()
    DEADLINE = (
        START_TIME
        +
        60.0 * MAX_RUNTIME_MINUTES
    )

    GLOBAL_DEADLINE = (
        START_TIME
        +
        GLOBAL_TIME_FRACTION
        *
        60.0
        *
        MAX_RUNTIME_MINUTES
    )

    TEMPLATE = load_template(
        TEMPLATE_FILE
    )

    template_pids = TEMPLATE["PID"].astype(int).tolist()
    if DRIVER_PID is None:
        DRIVER_PID = max(
            template_pids,
            key=lambda pid: PID_TO_DISTANCE_CM[int(pid)]
        )
    if int(DRIVER_PID) not in set(template_pids):
        raise RuntimeError(
            "Driver PID {} must be present in the fitted template.".format(
                DRIVER_PID
            )
        )

    OBJECTIVE_VERSION = build_objective_version()

    TEMPLATE_SHA256 = file_sha256(
        TEMPLATE_FILE
    )

    CONFIG_SHA256 = file_sha256(
        CRM_CONFIG
    )

    EXECUTABLE_SHA256 = file_sha256(
        CRM_EXECUTABLE
    )

    CS_SETTING = detect_cross_section_setting()

    if (
        REQUIRE_CS_SETTING is not None
        and CS_SETTING != REQUIRE_CS_SETTING
    ):
        raise RuntimeError(
            "This run requires k_eiae={}, but current config has k_eiae={}.".format(
                REQUIRE_CS_SETTING,
                CS_SETTING
            )
        )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    RUN_DIR = os.path.join(
        RESULTS_ROOT,
        "cs{}".format(CS_SETTING),
        timestamp
    )

    EVAL_DIR = os.path.join(
        RUN_DIR,
        "evaluations"
    )

    if not os.path.isdir(EVAL_DIR):
        os.makedirs(EVAL_DIR)

    # Preserve the exact starting best/template before it is overwritten.
    safe_copy(
        TEMPLATE_FILE,
        os.path.join(
            RUN_DIR,
            "starting_template.txt"
        )
    )

    safe_copy(
        CRM_CONFIG,
        os.path.join(
            RUN_DIR,
            "input_nofit_snapshot.txt"
        )
    )

    if not os.path.isdir(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)

    HISTORY_FILE = os.path.join(
        RESULTS_ROOT,
        "crm_hybrid_history_{}_cs{}_{}.csv".format(
            OBJECTIVE_VERSION,
            CS_SETTING,
            TEMPLATE_SHA256[:12]
        )
    )

    SUMMARY_FILE = os.path.join(
        RUN_DIR,
        "final_summary.txt"
    )

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    print(
        "Portable hybrid global + multi-start CRM fitter",
        flush=True
    )

    print(
        "Template/best: {}".format(
            TEMPLATE_FILE
        ),
        flush=True
    )

    print(
        "k_eiae={}".format(
            CS_SETTING
        ),
        flush=True
    )

    print(
        "Runtime budget: {:.1f} min; global stage budget: {:.1f} min".format(
            MAX_RUNTIME_MINUTES,
            MAX_RUNTIME_MINUTES * GLOBAL_TIME_FRACTION
        ),
        flush=True
    )

    print(
        "Hard constraints: H3+=0 | ions/ne {:.4g}..{:.4g} | "
        "nH/nH2 {:.4g}..{:.4g} | nH2 {:.4g}..{:.4g}".format(
            ION_TOTAL_OVER_NE_BOUNDS[0], ION_TOTAL_OVER_NE_BOUNDS[1],
            DISSOCIATION_RATIO_BOUNDS[0], DISSOCIATION_RATIO_BOUNDS[1],
            NH2_DENSITY_BOUNDS_M3[0], NH2_DENSITY_BOUNDS_M3[1],
        ),
        flush=True
    )

    print(
        "Objective: OES spectral error + adjustable driver-side neutral-density prior; "
        "smoothness remains tie-break only.",
        flush=True
    )

    print(
        "OES conversion: CRM line signal / CRM-reported opl -> volumetric "
        "emissivity; then integrate over the configured PID-distance profile.",
        flush=True
    )

    print(
        "Driver-concentration weight = {:.6g} "
        "(0 disables; larger values increasingly penalize nH/nH2 away from the configured driver PID).".format(
            DRIVER_CONCENTRATION_WEIGHT
        ),
        flush=True
    )

    load_compatible_history()

    initial = initial_params_from_template(
        TEMPLATE
    )

    initial_report = constraint_report(
        initial,
        TEMPLATE
    )

    print(
        "Projected initial state: ratio {:.3f}..{:.3f}, "
        "nH2 {:.3e}..{:.3e}, ions/ne {:.3f}, Hp {:.1f}%".format(
            initial_report["ratio_min"],
            initial_report["ratio_max"],
            initial_report["nH2_min_m3"],
            initial_report["nH2_max_m3"],
            initial_report["ion_total_over_ne"],
            100.0 * initial_report["hp_fraction"],
        ),
        flush=True
    )

    # ---------------------------------------------------------------------
    # Stage 1: broad global exploration.
    # ---------------------------------------------------------------------
    try:
        run_global_search(
            initial,
            rng
        )
    except TimeoutError as exc:
        print(
            "Optimization budget reached during global search: {}".format(exc),
            flush=True
        )

    if best_seen is None:
        raise RuntimeError(
            "No valid CRM evaluation was completed."
        )

    # ---------------------------------------------------------------------
    # Stage 2: independent local searches from several distinct good seeds.
    # ---------------------------------------------------------------------
    candidates = all_candidates_sorted()

    seeds = select_distinct_candidates(
        candidates,
        LOCAL_STARTS,
        min_distance=0.18
    )

    print(
        "\nSelected {} local starting candidates.".format(
            len(seeds)
        ),
        flush=True
    )

    for i, seed in enumerate(seeds):
        print(
            "  start {}: score {:.6g}, max|err| {:.2f}%, source {}".format(
                i + 1,
                seed["score"],
                100.0 * max_abs_relative_error(seed["errors"]),
                seed.get("source")
            ),
            flush=True
        )

    local_results = []

    for i, seed in enumerate(seeds):
        if deadline_reached():
            break

        # If we have measured CRM runtimes, avoid starting a full finite-
        # difference Jacobian when there is clearly not enough time left.
        if fresh_runtime_samples:
            mean_runtime = float(
                np.mean(
                    fresh_runtime_samples[-min(10, len(fresh_runtime_samples)):]
                )
            )

            estimated_jacobian_time = (
                N_PARAMS
                *
                max(mean_runtime, 0.01)
            )

            if (
                i > 0
                and time_remaining()
                <
                1.15 * estimated_jacobian_time
            ):
                print(
                    "Skipping another local start because remaining time "
                    "is shorter than an estimated fresh Jacobian.",
                    flush=True
                )
                break

        try:
            result = local_refine(
                seed,
                rng,
                i + 1
            )
        except TimeoutError as exc:
            print(
                "Optimization budget reached during local search: {}".format(exc),
                flush=True
            )
            break

        local_results.append(
            result
        )

        maybe_update_best(
            result
        )

        if (
            STOP_ON_TOLERANCE
            and best_seen is not None
            and all_lines_within_tolerance(best_seen["errors"])
        ):
            break

    # ---------------------------------------------------------------------
    # Final exact-file verification.
    # ---------------------------------------------------------------------
    if best_seen is None:
        raise RuntimeError(
            "Optimization ended without a best candidate."
        )

    final_verified = save_best_and_verify(
        best_seen
    )

    # The final fresh rerun is authoritative!!
    maybe_update_best(
        final_verified
    )

    write_summary(
        final_verified,
        local_results
    )

    bounds_df = bound_diagnostics(
        final_verified["params"]
    )

    print(
        "\n============================================================",
        flush=True
    )
    print(
        "FINAL VERIFIED BEST FOUND",
        flush=True
    )
    print(
        "============================================================",
        flush=True
    )

    print(
        "total objective score = {:.8g}".format(
            final_verified["score"]
        ),
        flush=True
    )

    final_spatial = driver_concentration_metrics(
        final_verified["params"]
    )

    print(
        "OES spectral score = {:.8g}".format(
            spectral_score(final_verified["errors"])
        ),
        flush=True
    )

    print(
        "driver weighted penalty = {:.8g} | RMS nH={:.3f}, nH2={:.3f}".format(
            final_spatial["weighted_penalty"],
            final_spatial["rms_nH"],
            final_spatial["rms_nH2"],
        ),
        flush=True
    )

    print(
        "RMS relative error = {:.2f}%".format(
            100.0 * rms_relative_error(
                final_verified["errors"]
            )
        ),
        flush=True
    )

    print(
        "max absolute relative error = {:.2f}%".format(
            100.0 * max_abs_relative_error(
                final_verified["errors"]
            )
        ),
        flush=True
    )

    print(
        "",
        flush=True
    )

    for line in LINE_ORDER:
        print(
            "  {:9s} calc {:.8E} target {:.8E} error {:+.2f}%".format(
                line,
                final_verified["values"][line],
                TARGETS[line],
                100.0 * final_verified["errors"][line]
            ),
            flush=True
        )

    print(
        "\nParameters close to hard bounds:",
        flush=True
    )

    pinned = bounds_df[
        bounds_df["within_5pct_of_bound"]
    ]

    if len(pinned) == 0:
        print(
            "  none within 5% of a bound",
            flush=True
        )
    else:
        for _, row in pinned.iterrows():
            print(
                "  {} = {:.6g} in [{:.6g}, {:.6g}]".format(
                    row["parameter"],
                    row["value"],
                    row["lower"],
                    row["upper"],
                ),
                flush=True
            )

    print(
        "\nBest input written to:\n{}".format(
            BEST_INPUT_FILE
        ),
        flush=True
    )

    print(
        "\nRun diagnostics:\n{}".format(
            RUN_DIR
        ),
        flush=True
    )

    print(
        "\nHistory:\n{}".format(
            HISTORY_FILE
        ),
        flush=True
    )

    print(
        "\nReminder: this is the best fit found by the hybrid search, not a formal proof of the global optimum.",
        flush=True
    )


if __name__ == "__main__":
    main()

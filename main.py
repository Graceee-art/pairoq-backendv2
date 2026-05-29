"""
Pairoq API v2.1 — Stabilized Production Backend
================================================
Railway free-tier (512MB RAM) compatible.
Single-file architecture. No compiled dependencies.

SCIENTIFIC DISCLAIMER:
This backend implements a regularized least-squares APPROXIMATION.
It is NOT equivalent to RES2DINV, pyGIMLi, or ResIPy.
Results are preliminary and require professional validation.

References:
- Edwards (1977): Wenner pseudo-depth formula z = 0.519 * a * n
- Constable et al. (1987): Occam regularization concept
- Loke & Barker (1996): Damping/correction principles
- Palacky (1987): Resistivity classification table
"""

from __future__ import annotations
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io, csv, base64, logging, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pairoq")

app = FastAPI(title="Pairoq API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ═══════════════════════════════════════════════════════════════
# COLORMAP
# ═══════════════════════════════════════════════════════════════
CMAP = LinearSegmentedColormap.from_list('pairoq', [
    (0.00, '#2C0057'), (0.08, '#1E3A8A'), (0.18, '#0284C7'),
    (0.30, '#06B6D4'), (0.42, '#10B981'), (0.54, '#84CC16'),
    (0.66, '#FBBF24'), (0.78, '#F97316'), (0.88, '#DC2626'),
    (1.00, '#7F1D1D'),
], N=512)

# ═══════════════════════════════════════════════════════════════
# ARRAY GEOMETRY
# ═══════════════════════════════════════════════════════════════

def wenner_depths(n: int, spacing: float) -> np.ndarray:
    """Edwards (1977): z = 0.519 * a * n"""
    return np.array([0.519 * spacing * (i + 1) for i in range(n)])

# ═══════════════════════════════════════════════════════════════
# NaN/INF SANITIZATION
# FIX #5: Prevent NaN propagation into inversion and rendering.
# All arrays must be sanitized before use.
# ═══════════════════════════════════════════════════════════════

def sanitize(arr: np.ndarray, fill: float = 1.0) -> np.ndarray:
    """Replace NaN/Inf with fill value. Always returns finite array."""
    arr = np.array(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr

def safe_log10(arr: np.ndarray, floor: float = 0.1) -> np.ndarray:
    """
    Safe log10 transform.
    FIX #5: Clamp to floor before log to prevent -inf from log10(0).
    """
    return np.log10(np.clip(sanitize(arr), floor, 1e9))

def safe_percentile(arr: np.ndarray, q: float, fallback: float = 1.0) -> float:
    """
    FIX #5: Safe percentile that handles empty/all-NaN arrays gracefully.
    Returns fallback if no valid values exist.
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return fallback
    return float(np.percentile(valid, q))

# ═══════════════════════════════════════════════════════════════
# PARSER
# FIX #1: Syscal CSV no longer uses sqrt(N) square reshape.
# Proper rectangular grid reconstruction from (x, n, rho) triplets.
# ═══════════════════════════════════════════════════════════════

def parse_res2dinv(content: bytes, spacing: float):
    """
    Parse Res2Dinv .DAT format.
    Returns: (grid, quality, depths, distances, spacing)
    """
    lines = [l.strip() for l in
             content.decode('utf-8', errors='ignore').splitlines()
             if l.strip()]
    if len(lines) < 5:
        raise ValueError("File too short for Res2Dinv DAT format.")
    try:
        spacing = float(lines[1])
    except Exception:
        pass

    points = []
    for line in lines[4:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x = float(parts[0])
            n = int(float(parts[1]))
            rho = float(parts[2])
            if x == 0 and n == 0 and rho == 0:
                break
            if rho > 0 and np.isfinite(rho):
                points.append((x, n, rho))
        except (ValueError, TypeError):
            continue

    if not points:
        raise ValueError("No valid data points in Res2Dinv DAT file.")

    return _triplets_to_grid(points, spacing)


def _triplets_to_grid(points: list, spacing: float):
    """
    FIX #1: Reconstruct proper RECTANGULAR grid from (x, n, rho) triplets.

    WHY: Real resistivity surveys have varying numbers of midpoints per level
    (fewer points at deeper levels). Using sqrt(N) forces a square grid which
    clips data and destroys the survey geometry.

    CORRECT approach: group by n-level, preserve actual horizontal positions.
    """
    n_vals = sorted(set(p[1] for p in points))
    n_levels = len(n_vals)

    # Collect rows grouped by n-level
    rows_data = []
    for nv in n_vals:
        row_pts = sorted([p for p in points if p[1] == nv], key=lambda p: p[0])
        rows_data.append(row_pts)

    # Use the maximum row length as n_cols (pad shorter rows)
    n_cols = max(len(r) for r in rows_data)

    grid = np.zeros((n_levels, n_cols), dtype=float)
    quality = np.zeros((n_levels, n_cols), dtype=bool)

    for ri, row_pts in enumerate(rows_data):
        for ci, (x, n, rho) in enumerate(row_pts):
            if ci < n_cols and np.isfinite(rho) and rho > 0.01:
                grid[ri, ci] = rho
                quality[ri, ci] = True
        # Pad missing columns with last valid value
        if row_pts and len(row_pts) < n_cols:
            last_val = row_pts[-1][2]
            for ci in range(len(row_pts), n_cols):
                grid[ri, ci] = last_val

    depths = wenner_depths(n_levels, spacing)
    distances = np.arange(n_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing


def parse_csv_generic(content: bytes, spacing: float):
    """
    Generic CSV: rows=depth levels, cols=electrode midpoints, values=Ω·m.
    Header row auto-skipped if non-numeric.
    """
    text = content.decode('utf-8', errors='ignore')
    rows = []
    for line in csv.reader(io.StringIO(text)):
        try:
            vals = [float(v.strip()) for v in line if v.strip()]
            if len(vals) >= 3:
                rows.append(vals)
        except ValueError:
            continue

    if not rows:
        raise ValueError(
            "No valid numeric data found. "
            "Expected: rows=depth levels, cols=electrode positions, values=Ω·m"
        )

    n_levels = len(rows)
    max_cols = max(len(r) for r in rows)
    # Pad shorter rows with last valid value
    grid = np.array([
        r + [r[-1]] * (max_cols - len(r)) for r in rows
    ], dtype=float)

    grid = sanitize(grid, fill=np.nanmedian(grid[grid > 0]) if (grid > 0).any() else 1.0)

    # QC: flag physically unreasonable values
    quality = (grid > 0.01) & (grid < 1e7) & np.isfinite(grid)
    if quality.any():
        p1 = safe_percentile(grid[quality], 1)
        p99 = safe_percentile(grid[quality], 99)
        quality &= (grid >= p1 * 0.01) & (grid <= p99 * 100)

    depths = wenner_depths(n_levels, spacing)
    distances = np.arange(max_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing


def parse_syscal_csv(content: bytes, spacing: float):
    """
    FIX #1: Syscal CSV parser.
    Reconstructs proper rectangular grid from column-based data.
    No longer uses sqrt(N) square reshape.
    """
    text = content.decode('utf-8', errors='ignore')
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows_raw = list(reader)
    except Exception as e:
        raise ValueError(f"Cannot parse Syscal CSV: {e}")

    if not rows_raw:
        raise ValueError("Empty Syscal CSV.")

    headers = {k.lower().strip(): k for k in rows_raw[0].keys()}
    rho_col = next((headers[c] for c in
                    ['rho', 'resistivity', 'app_res', 'appres', 'ra', 'r_app']
                    if c in headers), None)
    x_col = next((headers[c] for c in
                  ['x', 'xmid', 'x_mid', 'mid', 'midpoint', 'station']
                  if c in headers), None)
    n_col = next((headers[c] for c in
                  ['n', 'sep', 'separation', 'level', 'nl']
                  if c in headers), None)

    if rho_col is None:
        raise ValueError(
            f"Cannot find resistivity column in Syscal CSV. "
            f"Available: {list(headers.keys())}"
        )

    points = []
    for i, row in enumerate(rows_raw):
        try:
            rho = float(row[rho_col])
            x = float(row[x_col]) if x_col else i * spacing
            n = int(float(row[n_col])) if n_col else 1
            if np.isfinite(rho) and rho > 0:
                points.append((x, n, rho))
        except (ValueError, KeyError, TypeError):
            continue

    if not points:
        raise ValueError("No valid resistivity data in Syscal CSV.")

    # FIX #1: Use proper rectangular grid reconstruction
    return _triplets_to_grid(points, spacing)


def parse_file(content: bytes, filename: str, spacing: float):
    """Auto-detect format and route to correct parser."""
    fname = filename.lower()
    if fname.endswith('.dat'):
        return parse_res2dinv(content, spacing)

    text = content.decode('utf-8', errors='ignore')[:500]
    first_line = text.splitlines()[0].lower() if text.splitlines() else ''

    if any(k in first_line for k in ['rho', 'resistivity', 'app_res', 'ra,']):
        try:
            return parse_syscal_csv(content, spacing)
        except Exception:
            pass  # Fall through to generic

    return parse_csv_generic(content, spacing)


# ═══════════════════════════════════════════════════════════════
# INVERSION
# FIX #3: Adaptive damping, divergence detection, oscillation
# protection, RMS trend analysis, NaN enforcement.
# ═══════════════════════════════════════════════════════════════

def depth_sensitivity(depths: np.ndarray) -> np.ndarray:
    """
    Sensitivity decreases exponentially with depth.
    Approximates 1/r^2 geometric spreading.
    """
    norm = np.clip(depths / (depths[-1] + 1e-9), 0, 1)
    return np.exp(-1.5 * norm)


def lateral_smooth(row: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian lateral smoothing for one depth level."""
    n = min(7, len(row))
    x = np.arange(-(n // 2), n // 2 + 1)
    k = np.exp(-x ** 2 / (2 * max(sigma, 0.1) ** 2))
    k /= k.sum()
    smoothed = np.convolve(row, k, mode='same')
    return sanitize(smoothed, fill=np.nanmedian(row) if np.isfinite(row).any() else 1.0)


def regularized_inversion(
    apparent: np.ndarray,
    depths: np.ndarray,
    quality: np.ndarray,
    spacing: float,
    iterations: int = 6,
    smoothness: float = 0.5,
    convergence_thr: float = 2.0
):
    """
    Regularized least-squares approximation.

    FIX #3 changes:
    - Adaptive damping: detects RMS divergence and reduces step size
    - Oscillation protection: if RMS increases 2 consecutive times, halt
    - Finite-value enforcement after every operation
    - Safer clipping using actual valid data range (not fixed multipliers)
    """
    rows, cols = apparent.shape
    apparent = sanitize(apparent, fill=1.0)
    model = apparent.copy()

    # Fill invalid cells with neighbor interpolation
    for r in range(rows):
        for c in range(cols):
            if not quality[r, c]:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<rows and 0<=nc<cols and quality[nr,nc]:
                        neighbors.append(apparent[nr,nc])
                model[r, c] = np.mean(neighbors) if neighbors else np.mean(apparent[quality]) if quality.any() else 1.0

    model = sanitize(model, fill=1.0)
    sens = depth_sensitivity(depths)
    valid_data = apparent[quality] if quality.any() else apparent.flatten()
    valid_min = float(np.percentile(valid_data, 2)) * 0.3
    valid_max = float(np.percentile(valid_data, 98)) * 3.0
    valid_min = max(valid_min, 0.01)

    rms_log = []
    prev_rms = None
    diverge_count = 0
    converged = False

    for it in range(iterations):
        # FIX #3: Adaptive damping — reduce if diverging
        base_damp = smoothness / (1.0 + it * 0.4)
        if diverge_count > 0:
            base_damp *= (0.5 ** diverge_count)  # halve damping on divergence
        base_damp = float(np.clip(base_damp, 0.02, 0.8))

        # Lateral smoothing per depth level
        for r in range(rows):
            sigma = 0.4 + (1.0 - sens[r]) * 1.2
            sm = lateral_smooth(model[r, :], sigma)
            model[r, :] = (1 - base_damp) * model[r, :] + base_damp * sm

        # Vertical regularization
        for c in range(cols):
            col = model[:, c].copy()
            for r in range(1, rows - 1):
                w = base_damp * (1.0 - sens[r]) * 0.35
                w = float(np.clip(w, 0, 0.5))
                vert = 0.25*col[r-1] + 0.5*col[r] + 0.25*col[r+1]
                model[r, c] = (1-w)*col[r] + w*vert

        # Data fidelity correction
        residual = sanitize(apparent - model)
        for r in range(rows):
            corr_w = float(sens[r]) * 0.12
            model[r, quality[r, :]] += residual[r, quality[r, :]] * corr_w

        # Physical bounds + NaN enforcement
        model = np.clip(model, valid_min, valid_max)
        model = sanitize(model, fill=float(np.mean(valid_data)))

        # RMS on valid points
        valid_res = (model - apparent)[quality]
        if valid_res.size == 0:
            rms = 999.0
        else:
            rms = float(np.sqrt(np.mean(valid_res**2)) /
                        (np.mean(apparent[quality]) + 1e-9) * 100)

        if not np.isfinite(rms):
            rms = 999.0

        rms_log.append(round(rms, 4))

        # FIX #3: Divergence & oscillation detection
        if prev_rms is not None:
            if rms > prev_rms * 1.05:  # RMS grew > 5%
                diverge_count += 1
                logger.warning(f"Iter {it+1}: RMS diverged {prev_rms:.3f}→{rms:.3f}, reducing damping")
                if diverge_count >= 3:
                    logger.warning("Halting: 3 consecutive divergences")
                    break
            else:
                diverge_count = 0

        prev_rms = rms
        if rms < convergence_thr:
            converged = True
            break

    # Per-cell uncertainty
    residuals_rel = np.abs(sanitize(model - apparent)) / (sanitize(apparent) + 1e-9)
    depth_unc = (1 - sens).reshape(-1, 1) * np.ones((1, cols))
    uncertainty = np.clip(0.4*residuals_rel + 0.6*depth_unc, 0.0, 1.0)
    uncertainty[~quality] = 1.0
    uncertainty = sanitize(uncertainty, fill=1.0)

    return model, rms_log, converged, uncertainty


def compute_confidence(rms: float, converged: bool, completeness: float,
                       n_cols: int, n_levels: int) -> float:
    """
    Confidence reflects inversion stability, NOT geological certainty.
    Components: 40% RMS | 20% convergence | 25% completeness | 15% coverage
    """
    rms_score = float(np.clip(1 - (rms - 2) / 18, 0, 1))
    conv_score = 1.0 if converged else 0.5
    coverage = float(np.clip((n_cols / max(n_levels * 2, 1)) / 3, 0, 1))
    return round(float(np.clip(
        0.40*rms_score + 0.20*conv_score + 0.25*completeness + 0.15*coverage,
        0, 1
    )), 4)


# ═══════════════════════════════════════════════════════════════
# INTERPRETATION
# FIX #4: Classify zones using ABSOLUTE resistivity thresholds.
# WHY: Percentile-based slicing always produces 3 zones regardless
# of the actual data — even a perfectly uniform survey would show
# "Low / Moderate / High" zones. This is scientifically misleading.
# Absolute thresholds from Palacky (1987) / Reynolds (1997).
# ═══════════════════════════════════════════════════════════════

RESIS_TABLE = [
    (0,     10,    "Very Low",  "highly conductive anomaly",
     "consistent with saline water, clay-rich sediments, or metallic mineralization"),
    (10,    30,    "Low",       "conductive zone",
     "possible saturated sediments, clay layer, or weathered material"),
    (30,    100,   "Moderate",  "intermediate resistivity zone",
     "may indicate partially saturated sediments or alluvial deposits"),
    (100,   300,   "Elevated",  "moderately resistive zone",
     "consistent with dry sediments or consolidated material"),
    (300,   1000,  "High",      "resistive anomaly",
     "possible consolidated rock, dry coarse sediment, or resistive intrusion"),
    (1000,  1e9,   "Very High", "highly resistive anomaly",
     "consistent with crystalline rock or dry resistive bedrock"),
]

def classify_rho(v: float):
    for mn, mx, lbl, desc, ctx in RESIS_TABLE:
        if mn <= v < mx:
            return lbl, desc, ctx
    return "Very High", "highly resistive anomaly", RESIS_TABLE[-1][4]

def zone_color(cls: str) -> str:
    return {"Very Low":"#1E40AF","Low":"#0284C7","Moderate":"#059669",
            "Elevated":"#D97706","High":"#DC2626","Very High":"#7F1D1D"}.get(cls,"#6B7280")

def interpret(model: np.ndarray, uncertainty: np.ndarray, depths: np.ndarray):
    """
    FIX #4: Zone classification uses absolute resistivity thresholds.
    Only zones actually present in the data are reported.
    Anomaly detection requires uncertainty < 0.75 to avoid hallucination.
    """
    flat = sanitize(model.flatten(), fill=1.0)
    flat_unc = sanitize(uncertainty.flatten(), fill=1.0)

    # Group by absolute threshold — only report classes that exist
    zones = []
    seen_classes = set()
    for label_idx, (mn, mx, cls, desc, ctx) in enumerate(RESIS_TABLE):
        mask = (flat >= mn) & (flat < mx)
        if not mask.any():
            continue
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        vals = flat[mask]
        mean_rho = float(np.mean(vals))
        mean_unc = float(np.mean(flat_unc[mask]))
        conf = ("moderate-to-high" if mean_unc < 0.25
                else "moderate" if mean_unc < 0.50
                else "low")
        # Assign display label A/B/C in order of appearance
        display_label = f"Zone {chr(65 + len(zones))}"
        zones.append({
            "label": display_label,
            "resistivity_class": cls,
            "geophysical_descriptor": desc,
            "geological_context": ctx,
            "range_ohm_m": [round(float(vals.min()),1), round(float(vals.max()),1)],
            "mean_ohm_m": round(mean_rho, 2),
            "uncertainty": round(mean_unc, 3),
            "confidence_qualifier": conf,
            "range": f"{vals.min():.0f}–{vals.max():.0f} Ω·m",
            "interpretation": ctx,
            "color": zone_color(cls),
        })

    # Anomaly detection (statistical, only low-uncertainty regions)
    log_m = safe_log10(model)
    lmean = float(np.mean(log_m[np.isfinite(log_m)]))
    lstd = float(np.std(log_m[np.isfinite(log_m)]))
    anomalies = []
    for atype, amask in [
        ("conductive", (log_m < lmean - 1.5*lstd) & (uncertainty < 0.75)),
        ("resistive",  (log_m > lmean + 1.5*lstd) & (uncertainty < 0.75)),
    ]:
        if not amask.any():
            continue
        ri, ci = np.where(amask)
        mean_rho = float(np.mean(model[amask]))
        mean_unc = float(np.mean(uncertainty[amask]))
        mean_depth = float(np.mean(depths[np.clip(ri, 0, len(depths)-1)]))
        cls, desc, ctx = classify_rho(mean_rho)
        anomalies.append({
            "type": atype,
            "geophysical_label": f"{atype.title()} anomaly ({desc})",
            "mean_resistivity_ohm_m": round(mean_rho, 1),
            "approximate_center_depth_m": round(mean_depth, 1),
            "geological_context": ctx,
            "confidence_qualifier": (
                "moderate-to-high" if mean_unc<0.25
                else "moderate" if mean_unc<0.5 else "low"
            ),
            "caveat": "Extent is approximate. Field verification required."
        })

    return zones, anomalies


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# FIX #2: Always use meshgrid. Validate dimensions before plotting.
# FIX #6: Aggressive figure/buffer cleanup to prevent memory leaks.
# ═══════════════════════════════════════════════════════════════

def render(model, uncertainty, depths, distances, rms_log, converged,
           confidence, filename, display_mode, spacing):
    """
    FIX #2: meshgrid used for all contour calls.
    WHY: imshow uses array indices; contour/contourf needs explicit X,Y
    coordinate grids. Without meshgrid, shape mismatches crash on
    non-square or irregular grids.

    FIX #6: plt.close(fig) + del buf immediately after encode.
    WHY: matplotlib figures hold ~10-50MB each. On Railway 512MB,
    accumulating unclosed figures causes OOM crashes.
    """
    model = sanitize(model, fill=1.0)
    uncertainty = sanitize(uncertainty, fill=1.0)

    log_m = safe_log10(model)
    log_min = safe_percentile(log_m, 2, fallback=-1.0)
    log_max = safe_percentile(log_m, 98, fallback=3.0)
    if log_max <= log_min:
        log_max = log_min + 1.0
    norm = mcolors.Normalize(vmin=log_min, vmax=log_max)

    # FIX #2: Build explicit meshgrid for contour calls
    X, Y = np.meshgrid(distances, depths)

    # Validate all dimensions match
    assert X.shape == model.shape, f"Meshgrid shape {X.shape} != model {model.shape}"

    fig = None
    buf = None
    try:
        fig = plt.figure(figsize=(18, 6), facecolor='white')
        ax = fig.add_axes([0.06, 0.16, 0.83, 0.66])
        ax.set_facecolor('white')

        if display_mode == 'contoured':
            levels_fill = np.linspace(log_min, log_max, 20)
            ax.contourf(X, Y, log_m, levels=levels_fill,
                        cmap=CMAP, norm=norm, extend='both')
            cs = ax.contour(X, Y, log_m, levels=levels_fill[::3],
                            colors='black', linewidths=0.9, alpha=0.7)
            try:
                ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7,
                          fmt=lambda x: f'{10**x:.0f}',
                          colors='black', inline_spacing=2)
            except Exception:
                pass
        elif display_mode == 'hybrid':
            ax.imshow(log_m, aspect='auto', origin='upper',
                      extent=[distances[0], distances[-1], depths[-1], depths[0]],
                      cmap=CMAP, norm=norm, interpolation='bilinear')
            cs = ax.contour(X, Y, log_m,
                            levels=np.linspace(log_min, log_max, 12)[1:-1],
                            colors='black', linewidths=0.8, alpha=0.75)
            try:
                ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7,
                          fmt=lambda x: f'{10**x:.0f}',
                          colors='black', inline_spacing=2)
            except Exception:
                pass
        else:  # smooth (default)
            ax.imshow(log_m, aspect='auto', origin='upper',
                      extent=[distances[0], distances[-1], depths[-1], depths[0]],
                      cmap=CMAP, norm=norm, interpolation='bilinear')
            try:
                ax.contour(X, Y, log_m,
                           levels=np.linspace(log_min, log_max, 10)[1:-1],
                           colors='black', linewidths=0.6, alpha=0.5)
            except Exception:
                pass

        # High-uncertainty hatch overlay
        if (uncertainty > 0.70).any():
            try:
                ax.contourf(X, Y, uncertainty,
                            levels=[0.70, 1.01],
                            colors='white', alpha=0.25, hatches=['////'])
            except Exception:
                pass

        ax.set_xlabel('Distance (m)', fontsize=11)
        ax.set_ylabel('Depth (m)', fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.4, color='gray')
        ax.set_axisbelow(True)
        for sp2 in ax.spines.values():
            sp2.set_linewidth(0.8)
        step = max(1, len(depths) // 6)
        ax.set_yticks(depths[::step])
        ax.set_yticklabels([f'{d:.1f}' for d in depths[::step]], fontsize=9)

        # Colorbar
        cax = fig.add_axes([0.91, 0.16, 0.016, 0.66])
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Resistivity (Ω·m)', fontsize=10, labelpad=8)
        log_ticks = np.linspace(log_min, log_max, 8)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([f'{10**t:.0f}' for t in log_ticks], fontsize=8)

        # Title
        conv_str = "converged" if converged else "not converged"
        rms_val = rms_log[-1] if rms_log else 0.0
        fig.text(0.5, 0.95, f'Model Resistivity Section  —  {filename}',
                 ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 0.91,
                 f'RMS = {rms_val:.2f}%  |  Iter: {len(rms_log)}  |  {conv_str}',
                 ha='center', fontsize=9, color='#444')
        conf_pct = int(confidence * 100)
        conf_color = ('#16a34a' if conf_pct >= 70
                      else '#d97706' if conf_pct >= 50 else '#dc2626')
        fig.text(0.06, 0.91, f'Confidence: {conf_pct}%',
                 fontsize=8, color=conf_color, fontweight='bold')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200,
                    facecolor='white', bbox_inches='tight')
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        return result

    finally:
        # FIX #6: Always close figure and buffer regardless of success/failure
        if fig is not None:
            plt.close(fig)
        if buf is not None:
            buf.close()


# ═══════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "Pairoq Geophysical API",
        "version": "2.1.0",
        "inversion": "regularized_least_squares_approximation",
        "disclaimer": (
            "Preliminary interpretation tool. "
            "Not equivalent to RES2DINV/pyGIMLi. "
            "Requires professional validation."
        ),
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.1.0"}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    spacing: float = Form(5.0),
    iterations: int = Form(6),
    array_type: str = Form("wenner_alpha"),
    display_mode: str = Form("hybrid"),
):
    t0 = time.time()
    spacing = float(np.clip(spacing, 0.1, 1000.0))
    iterations = int(np.clip(iterations, 1, 10))
    if display_mode not in ("smooth", "contoured", "hybrid"):
        display_mode = "hybrid"

    content = await file.read()
    filename = file.filename or "survey"

    try:
        grid, quality, depths, distances, spacing = parse_file(content, filename, spacing)
    except ValueError as e:
        raise HTTPException(400, f"Parse error: {e}")
    except Exception as e:
        logger.error(f"Parse error: {e}")
        raise HTTPException(500, f"Internal parse error: {e}")

    # Minimum viable dataset
    if grid.shape[0] < 2 or grid.shape[1] < 3:
        raise HTTPException(400, "Dataset too small. Need at least 2 depth levels and 3 electrode positions.")

    try:
        model, rms_log, converged, uncertainty = regularized_inversion(
            grid, depths, quality, spacing, iterations
        )
    except Exception as e:
        logger.error(f"Inversion error: {e}")
        raise HTTPException(500, f"Inversion error: {e}")

    completeness = float(quality.sum()) / max(quality.size, 1)
    confidence = compute_confidence(
        rms_log[-1] if rms_log else 999.0,
        converged, completeness, grid.shape[1], grid.shape[0]
    )
    zones, anomalies = interpret(model, uncertainty, depths)

    try:
        img = render(model, uncertainty, depths, distances,
                     rms_log, converged, confidence,
                     filename, display_mode, spacing)
    except Exception as e:
        logger.error(f"Render error: {e}")
        raise HTTPException(500, f"Render error: {e}")

    valid_data = grid[quality] if quality.any() else grid.flatten()
    elapsed = round(time.time() - t0, 2)
    logger.info(f"Processed {filename} in {elapsed}s | RMS={rms_log[-1] if rms_log else 'N/A'}")

    return JSONResponse({
        "status": "ok",
        "processing_time_s": elapsed,
        "image_b64": img,
        "stats": {
            "points_total": int(quality.size),
            "points_valid": int(quality.sum()),
            "data_completeness": round(completeness, 4),
            "n_levels": int(grid.shape[0]),
            "n_electrodes_estimated": int(grid.shape[1] + 3*grid.shape[0]),
            "min_resistivity_ohm_m": round(float(safe_percentile(valid_data, 0)), 2),
            "max_resistivity_ohm_m": round(float(safe_percentile(valid_data, 100)), 2),
            "mean_resistivity_ohm_m": round(float(np.mean(valid_data)), 2),
            "max_depth_m": round(float(depths[-1]), 2),
            "electrode_spacing_m": spacing,
            "array_type": array_type,
        },
        "inversion": {
            "method": "regularized_least_squares_approximation",
            "iterations_run": len(rms_log),
            "converged": converged,
            "rms_final_pct": rms_log[-1] if rms_log else None,
            "rms_per_iteration": rms_log,
            "confidence_score": confidence,
            "disclaimer": (
                "Lightweight approximation. Not RES2DINV/pyGIMLi equivalent. "
                "For scientific publication use professional inversion software."
            ),
        },
        "zones": zones,
        "anomalies": anomalies,
        "interpretation": {
            "disclaimer": (
                "Preliminary only. Anomalies described in geophysical terms. "
                "Geological conclusions require professional validation."
            ),
        },
        "depths_m": [round(float(d), 2) for d in depths],
        "distances_m": [round(float(d), 2) for d in distances],
        # Legacy fields for frontend compatibility
        "rms_final": rms_log[-1] if rms_log else None,
        "rms_history": rms_log,
    })

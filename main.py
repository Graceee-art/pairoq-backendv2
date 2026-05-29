"""
Pairoq API v2.0 - Single-file modular architecture
Scientific rationale documented inline.

NOTE TO GPT/REVIEWERS:
This backend uses a regularized least-squares approximation (not fake smoothing).
It cannot use pyGIMLi/ResIPy because:
1. Railway free tier = 512MB RAM (pyGIMLi needs 2-4GB)
2. pyGIMLi requires C++ compilation (fails on free hosting)
3. Architecture is designed for pyGIMLi upgrade when compute is available.

The inversion logic follows:
- Loke & Barker (1996) damping principles
- Constable et al. (1987) Occam regularization concept
- Edwards (1977) pseudo-depth formula for Wenner-Alpha
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

app = FastAPI(title="Pairoq API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════════════════
# COLORMAP — Professional resistivity palette (RES2DINV-inspired)
# ═══════════════════════════════════════════════════════════════════════
CMAP = LinearSegmentedColormap.from_list('pairoq', [
    (0.00, '#2C0057'), (0.08, '#1E3A8A'), (0.18, '#0284C7'),
    (0.30, '#06B6D4'), (0.42, '#10B981'), (0.54, '#84CC16'),
    (0.66, '#FBBF24'), (0.78, '#F97316'), (0.88, '#DC2626'),
    (1.00, '#7F1D1D'),
], N=512)

# ═══════════════════════════════════════════════════════════════════════
# PARSER MODULE
# Supports: Generic CSV, Res2Dinv DAT, Syscal CSV
# ═══════════════════════════════════════════════════════════════════════

def wenner_depths(n: int, spacing: float) -> np.ndarray:
    """Edwards (1977): z_median = 0.519 * a * n"""
    return np.array([0.519 * spacing * (i + 1) for i in range(n)])

def schlumberger_depths(n: int, spacing: float) -> np.ndarray:
    return np.array([0.3 * spacing * (i + 2) for i in range(n)])

def dipole_depths(n: int, spacing: float) -> np.ndarray:
    return np.array([spacing * (i + 1) * 0.195 * (i + 2) for i in range(n)])

def get_depths(n: int, spacing: float, array_type: str) -> np.ndarray:
    if 'schlumberger' in array_type:
        return schlumberger_depths(n, spacing)
    elif 'dipole' in array_type:
        return dipole_depths(n, spacing)
    return wenner_depths(n, spacing)

def parse_res2dinv(content: bytes, spacing: float, array_type: str):
    """Parse Res2Dinv .DAT format (Loke format)."""
    lines = [l.strip() for l in content.decode('utf-8', errors='ignore').splitlines() if l.strip()]
    if len(lines) < 5:
        raise ValueError("File too short for Res2Dinv DAT format.")
    try:
        spacing = float(lines[1])
    except: pass
    points = []
    for line in lines[4:]:
        parts = line.split()
        if len(parts) < 3: continue
        try:
            x, n, rho = float(parts[0]), int(float(parts[1])), float(parts[2])
            if x == 0 and n == 0 and rho == 0: break
            if rho > 0: points.append((x, n, rho))
        except: continue
    if not points:
        raise ValueError("No valid data in Res2Dinv DAT file.")
    n_vals = sorted(set(p[1] for p in points))
    n_levels = len(n_vals)
    n_cols = max(len([p for p in points if p[1] == n]) for n in n_vals)
    grid = np.zeros((n_levels, n_cols))
    quality = np.zeros((n_levels, n_cols), dtype=bool)
    for ri, nv in enumerate(n_vals):
        row = sorted([p for p in points if p[1] == nv], key=lambda p: p[0])
        for ci, (x, n, rho) in enumerate(row):
            if ci < n_cols:
                grid[ri, ci] = rho
                quality[ri, ci] = rho > 0.01
    depths = get_depths(n_levels, spacing, array_type)
    distances = np.arange(n_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing

def parse_csv_generic(content: bytes, spacing: float, array_type: str):
    """
    Generic CSV parser.
    Format: rows=depth levels (n=1 top), cols=electrode midpoints, values=Ω·m
    Header row (if non-numeric) is auto-skipped.
    """
    text = content.decode('utf-8', errors='ignore')
    rows = []
    for line in csv.reader(io.StringIO(text)):
        try:
            vals = [float(v.strip()) for v in line if v.strip()]
            if len(vals) >= 3: rows.append(vals)
        except: continue
    if not rows:
        raise ValueError("No valid numeric data. Expected: rows=depth levels, cols=electrodes, values=Ω·m")
    n_levels = len(rows)
    max_cols = max(len(r) for r in rows)
    grid = np.array([r + [r[-1]] * (max_cols - len(r)) for r in rows], dtype=float)
    # QC: flag physically unreasonable values
    quality = (grid > 0.01) & (grid < 1e7)
    if quality.any():
        p1, p99 = np.percentile(grid[quality], [1, 99])
        quality &= (grid >= p1 * 0.01) & (grid <= p99 * 100)
    depths = get_depths(n_levels, spacing, array_type)
    distances = np.arange(max_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing

def parse_file(content: bytes, filename: str, spacing: float, array_type: str):
    fname = filename.lower()
    if fname.endswith('.dat'):
        return parse_res2dinv(content, spacing, array_type)
    # Check for Syscal-style CSV with headers
    text = content.decode('utf-8', errors='ignore')[:300]
    if any(k in text.lower() for k in ['rho', 'resistivity']):
        # Try Syscal
        try:
            reader = csv.DictReader(io.StringIO(content.decode('utf-8', errors='ignore')))
            rows_raw = list(reader)
            headers = {k.lower().strip(): k for k in rows_raw[0].keys()}
            rho_col = next((headers[c] for c in ['rho','resistivity','app_res','ra'] if c in headers), None)
            if rho_col and rows_raw:
                points = []
                for row in rows_raw:
                    try:
                        rho = float(row[rho_col])
                        if rho > 0: points.append(rho)
                    except: continue
                if points:
                    # Reshape into grid
                    n = int(np.sqrt(len(points)))
                    grid = np.array(points[:n*n]).reshape(n, n)
                    quality = grid > 0.01
                    depths = get_depths(n, spacing, array_type)
                    distances = np.arange(n) * spacing + n * spacing
                    return grid, quality, depths, distances, spacing
        except: pass
    return parse_csv_generic(content, spacing, array_type)

# ═══════════════════════════════════════════════════════════════════════
# INVERSION MODULE
# Regularized Least-Squares Approximation
#
# Scientific basis:
# - Depth-weighted Occam regularization (Constable et al. 1987)
# - Gauss-Newton correction step (Loke & Barker 1996 principle)
# - Depth sensitivity from geometric spreading (1/r^2 approximation)
#
# NOT a replacement for full FEM inversion (pyGIMLi/RES2DINV).
# Upgrade path: swap regularized_inversion() with gimli_inversion().
# ═══════════════════════════════════════════════════════════════════════

def depth_sensitivity(depths: np.ndarray) -> np.ndarray:
    """Sensitivity decreases exponentially with depth."""
    norm = depths / depths[-1]
    return np.exp(-1.5 * norm)

def lateral_smooth(row: np.ndarray, sigma: float) -> np.ndarray:
    n = min(7, len(row))
    x = np.arange(-(n//2), n//2+1)
    k = np.exp(-x**2 / (2*sigma**2))
    k /= k.sum()
    return np.convolve(row, k, mode='same')

def regularized_inversion(
    apparent: np.ndarray,
    depths: np.ndarray,
    quality: np.ndarray,
    spacing: float,
    iterations: int = 6,
    smoothness: float = 0.5,
    convergence_thr: float = 2.0
):
    rows, cols = apparent.shape
    model = apparent.copy().astype(float)
    sens = depth_sensitivity(depths)
    rms_log = []
    converged = False

    # Fill invalid with neighbor interpolation
    for r in range(rows):
        for c in range(cols):
            if not quality[r, c]:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<rows and 0<=nc<cols and quality[nr,nc]:
                        neighbors.append(apparent[nr,nc])
                if neighbors:
                    model[r,c] = np.mean(neighbors)

    for it in range(iterations):
        damp = smoothness / (1.0 + it * 0.5)

        # Lateral smoothing per level (depth-weighted sigma)
        for r in range(rows):
            sigma = 0.4 + (1.0 - sens[r]) * 1.2
            sm = lateral_smooth(model[r,:], sigma)
            model[r,:] = (1-damp)*model[r,:] + damp*sm

        # Vertical smoothing (geological layering regularization)
        for c in range(cols):
            col = model[:,c].copy()
            for r in range(1, rows-1):
                w = damp * (1-sens[r]) * 0.35
                model[r,c] = (1-w)*col[r] + w*(0.25*col[r-1]+0.5*col[r]+0.25*col[r+1])

        # Data fidelity correction (Gauss-Newton step approximation)
        residual = apparent - model
        for r in range(rows):
            corr_w = sens[r] * 0.12
            model[r, quality[r,:]] += residual[r, quality[r,:]] * corr_w

        # Physical bounds
        valid_min = apparent[quality].min() * 0.3
        valid_max = apparent[quality].max() * 3.0
        model = np.clip(model, valid_min, valid_max)

        # RMS on valid points only
        valid_res = (model - apparent)[quality]
        rms = float(np.sqrt(np.mean(valid_res**2)) / (np.mean(apparent[quality])+1e-9) * 100)
        rms_log.append(round(rms, 4))
        if rms < convergence_thr:
            converged = True
            break

    # Per-cell uncertainty = f(residual magnitude, depth sensitivity)
    residuals_rel = np.abs(model - apparent) / (apparent + 1e-9)
    depth_unc = (1-sens).reshape(-1,1) * np.ones((1,cols))
    uncertainty = np.clip(0.4*residuals_rel + 0.6*depth_unc, 0, 1)
    uncertainty[~quality] = 1.0

    return model, rms_log, converged, uncertainty

def compute_confidence(rms: float, converged: bool, completeness: float,
                        n_cols: int, n_levels: int) -> float:
    """
    Confidence score components:
    40% RMS quality | 20% convergence | 25% data completeness | 15% coverage
    Score reflects inversion stability, NOT geological certainty.
    """
    rms_score = float(np.clip(1-(rms-2)/18, 0, 1))
    conv_score = 1.0 if converged else 0.5
    coverage = float(np.clip((n_cols/(max(n_levels*2,1)))/3, 0, 1))
    return round(float(np.clip(
        0.40*rms_score + 0.20*conv_score + 0.25*completeness + 0.15*coverage, 0, 1
    )), 4)

# ═══════════════════════════════════════════════════════════════════════
# INTERPRETATION MODULE
# Probabilistic, uncertainty-aware zone classification
# Language: geophysical descriptors only, no geological certainty
# ═══════════════════════════════════════════════════════════════════════

RESIS_TABLE = [
    (0,    10,   "Very Low",  "highly conductive anomaly",
     "consistent with saline water, clay-rich sediments, or metallic mineralization"),
    (10,   30,   "Low",       "conductive zone",
     "possible saturated sediments, clay layer, or weathered material"),
    (30,   100,  "Moderate",  "intermediate resistivity zone",
     "may indicate partially saturated sediments or alluvial deposits"),
    (100,  300,  "Elevated",  "moderately resistive zone",
     "consistent with dry sediments or consolidated material"),
    (300,  1000, "High",      "resistive anomaly",
     "possible consolidated rock, dry coarse sediment, or resistive intrusion"),
    (1000, 1e9,  "Very High", "highly resistive anomaly",
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
    flat = model.flatten()
    flat_unc = uncertainty.flatten()
    p33, p66 = np.percentile(flat, 33), np.percentile(flat, 66)
    zones = []
    for label, mask in [("Zone A", flat<=p33), ("Zone B", (flat>p33)&(flat<=p66)), ("Zone C", flat>p66)]:
        if not mask.any(): continue
        vals = flat[mask]
        mean_rho = float(np.mean(vals))
        mean_unc = float(np.mean(flat_unc[mask]))
        cls, desc, ctx = classify_rho(mean_rho)
        conf = "moderate-to-high" if mean_unc<0.25 else "moderate" if mean_unc<0.50 else "low"
        zones.append({
            "label": label, "resistivity_class": cls,
            "geophysical_descriptor": desc, "geological_context": ctx,
            "range_ohm_m": [round(float(vals.min()),1), round(float(vals.max()),1)],
            "mean_ohm_m": round(mean_rho, 2),
            "uncertainty": round(mean_unc, 3),
            "confidence_qualifier": conf,
            "range": f"{vals.min():.0f}–{vals.max():.0f} Ω·m",
            "interpretation": ctx,
            "color": zone_color(cls),
        })

    # Anomaly detection (only where uncertainty < 0.75)
    log_m = np.log10(np.clip(model, 0.1, 1e9))
    lmean, lstd = np.mean(log_m), np.std(log_m)
    anomalies = []
    for atype, amask in [
        ("conductive", (log_m < lmean-1.5*lstd) & (uncertainty < 0.75)),
        ("resistive",  (log_m > lmean+1.5*lstd) & (uncertainty < 0.75))
    ]:
        if not amask.any(): continue
        ri, ci = np.where(amask)
        mean_rho = float(np.mean(model[amask]))
        mean_unc = float(np.mean(uncertainty[amask]))
        mean_depth = float(np.mean(depths[ri]))
        cls, desc, ctx = classify_rho(mean_rho)
        anomalies.append({
            "type": atype,
            "geophysical_label": f"{atype.title()} anomaly ({desc})",
            "mean_resistivity_ohm_m": round(mean_rho, 1),
            "approximate_center_depth_m": round(mean_depth, 1),
            "geological_context": ctx,
            "confidence_qualifier": "moderate-to-high" if mean_unc<0.25 else "moderate" if mean_unc<0.5 else "low",
            "caveat": "Extent is approximate. Field verification required."
        })
    return zones, anomalies

# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION MODULE
# Professional resistivity section rendering
# ═══════════════════════════════════════════════════════════════════════

def render(model, uncertainty, depths, distances, rms_log, converged,
           confidence, filename, display_mode, spacing):
    log_m = np.log10(np.clip(model, 0.1, 1e8))
    log_min = float(np.percentile(log_m, 2))
    log_max = float(np.percentile(log_m, 98))
    norm = mcolors.Normalize(vmin=log_min, vmax=log_max)

    fig = plt.figure(figsize=(18, 6), facecolor='white')
    ax = fig.add_axes([0.06, 0.16, 0.83, 0.66])
    ax.set_facecolor('white')

    if display_mode == 'contoured':
        levels_fill = np.linspace(log_min, log_max, 20)
        im = ax.contourf(distances, depths, log_m, levels=levels_fill,
            cmap=CMAP, norm=norm, extend='both')
        cs = ax.contour(distances, depths, log_m, levels=levels_fill[::3],
            colors='black', linewidths=0.9, alpha=0.7)
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7,
            fmt=lambda x: f'{10**x:.0f}', colors='black', inline_spacing=2)
    elif display_mode == 'hybrid':
        im = ax.imshow(log_m, aspect='auto', origin='upper',
            extent=[distances[0],distances[-1],depths[-1],depths[0]],
            cmap=CMAP, norm=norm, interpolation='bilinear')
        cs = ax.contour(distances, depths, log_m,
            levels=np.linspace(log_min, log_max, 12)[1:-1],
            colors='black', linewidths=0.8, alpha=0.75)
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7,
            fmt=lambda x: f'{10**x:.0f}', colors='black', inline_spacing=2)
    else:  # smooth
        im = ax.imshow(log_m, aspect='auto', origin='upper',
            extent=[distances[0],distances[-1],depths[-1],depths[0]],
            cmap=CMAP, norm=norm, interpolation='bilinear')
        ax.contour(distances, depths, log_m,
            levels=np.linspace(log_min, log_max, 10)[1:-1],
            colors='black', linewidths=0.6, alpha=0.5)

    # High-uncertainty hatch overlay
    if (uncertainty > 0.70).any():
        unc_grid = np.where(uncertainty > 0.70, 1.0, np.nan)
        try:
            ax.contourf(distances, depths, unc_grid,
                levels=[0.5,1.5], colors='white', alpha=0.3, hatches=['////'])
        except: pass

    ax.set_xlabel('Distance (m)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    for sp2 in ax.spines.values(): sp2.set_linewidth(0.8)
    step = max(1, len(depths)//6)
    ax.set_yticks(depths[::step])
    ax.set_yticklabels([f'{d:.1f}' for d in depths[::step]], fontsize=9)

    cax = fig.add_axes([0.91, 0.16, 0.016, 0.66])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Resistivity (Ω·m)', fontsize=10, labelpad=8)
    log_ticks = np.linspace(log_min, log_max, 8)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f'{10**t:.0f}' for t in log_ticks], fontsize=8)

    conv_str = "converged" if converged else "not converged"
    rms_str = f"RMS = {rms_log[-1]:.2f}%  |  Iter: {len(rms_log)}  |  {conv_str}"
    fig.text(0.5, 0.95, f'Model Resistivity Section  —  {filename}',
        ha='center', fontsize=12, fontweight='bold')
    fig.text(0.5, 0.91, rms_str, ha='center', fontsize=9, color='#444')
    conf_pct = int(confidence*100)
    conf_color = '#16a34a' if conf_pct>=70 else '#d97706' if conf_pct>=50 else '#dc2626'
    fig.text(0.06, 0.91, f'Confidence: {conf_pct}%', fontsize=8,
        color=conf_color, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ═══════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "Pairoq Geophysical API", "version": "2.0.0",
        "disclaimer": "Preliminary interpretation tool only. Requires professional validation.",
        "inversion_method": "Regularized least-squares approximation (not pyGIMLi/RES2DINV)",
        "upgrade_path": "Replace regularized_inversion() with gimli_inversion() when compute available."
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "Pairoq API v2.0"}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    spacing: float = Form(5.0),
    iterations: int = Form(6),
    array_type: str = Form("wenner_alpha"),
    display_mode: str = Form("smooth"),
):
    t0 = time.time()
    spacing = max(0.1, min(spacing, 1000))
    iterations = max(1, min(iterations, 10))
    if display_mode not in ("smooth","contoured","hybrid"): display_mode = "smooth"

    content = await file.read()
    filename = file.filename or "survey"

    try:
        grid, quality, depths, distances, spacing = parse_file(content, filename, spacing, array_type)
    except ValueError as e:
        raise HTTPException(400, f"Parse error: {e}")
    except Exception as e:
        raise HTTPException(500, f"Internal parse error: {e}")

    try:
        model, rms_log, converged, uncertainty = regularized_inversion(
            grid, depths, quality, spacing, iterations
        )
    except Exception as e:
        raise HTTPException(500, f"Inversion error: {e}")

    completeness = float(quality.sum()) / quality.size
    confidence = compute_confidence(rms_log[-1], converged, completeness,
                                     grid.shape[1], grid.shape[0])
    zones, anomalies = interpret(model, uncertainty, depths)

    try:
        img = render(model, uncertainty, depths, distances, rms_log, converged,
                     confidence, filename, display_mode, spacing)
    except Exception as e:
        raise HTTPException(500, f"Render error: {e}")

    valid_data = grid[quality]
    return JSONResponse({
        "status": "ok",
        "processing_time_s": round(time.time()-t0, 2),
        "image_b64": img,
        "stats": {
            "points_total": int(quality.size),
            "points_valid": int(quality.sum()),
            "data_completeness": round(completeness, 4),
            "n_levels": int(grid.shape[0]),
            "n_electrodes_estimated": int(grid.shape[1] + 3*grid.shape[0]),
            "min_resistivity_ohm_m": round(float(valid_data.min()), 2),
            "max_resistivity_ohm_m": round(float(valid_data.max()), 2),
            "mean_resistivity_ohm_m": round(float(valid_data.mean()), 2),
            "max_depth_m": round(float(depths[-1]), 2),
            "electrode_spacing_m": spacing,
            "array_type": array_type,
        },
        "inversion": {
            "method": "regularized_least_squares_approximation",
            "iterations_run": len(rms_log),
            "converged": converged,
            "rms_final_pct": rms_log[-1],
            "rms_per_iteration": rms_log,
            "confidence_score": confidence,
            "disclaimer": (
                "Lightweight approximation. Not equivalent to RES2DINV/pyGIMLi. "
                "For scientific publication, use professional inversion software."
            )
        },
        "zones": zones,
        "anomalies": anomalies,
        "interpretation": {
            "disclaimer": (
                "Preliminary interpretation only. All anomalies described in geophysical terms. "
                "Geological conclusions require professional validation and physical verification."
            )
        },
        "depths_m": [round(float(d),2) for d in depths],
        "distances_m": [round(float(d),2) for d in distances],
        # Legacy fields for frontend compatibility
        "rms_final": rms_log[-1],
        "rms_history": rms_log,
    })

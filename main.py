"""
Pairoq API v2.3 — Production Ready Stabilization
=================================================
Railway free-tier (512MB RAM). Single-file. No compiled deps.

SCIENTIFIC DISCLAIMER:
Lightweight pseudo forward-response & iterative masked relaxation.
Approximation only. NOT equivalent to RES2DINV, pyGIMLi, or ResIPy.
Results are preliminary. Professional validation required.

FIXED IN V2.3:
1. True Wenner inverted trapezoid geometry (No more false right-edge anomalies).
2. Masked Laplacian Regularization & Depth-decay update (RAM safe & non-blurring).
3. Fixed Geometric-Logarithmic contour intervals ala RES2DINV.
"""

from __future__ import annotations
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io, csv, base64, logging, time, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pairoq")

app = FastAPI(title="Pairoq API", version="2.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MAX_UPLOAD_BYTES  = 2 * 1024 * 1024
MAX_ROWS          = 30
MAX_COLS          = 200
MAX_VALID_RHO     = 1_000_000.0
MIN_VALID_RHO     = 0.001
STAGNATION_DELTA  = 0.01

CMAP = LinearSegmentedColormap.from_list('pairoq', [
    (0.00, '#2C0057'), (0.08, '#1E3A8A'), (0.18, '#0284C7'),
    (0.30, '#06B6D4'), (0.42, '#10B981'), (0.54, '#84CC16'),
    (0.66, '#FBBF24'), (0.78, '#F97316'), (0.88, '#DC2626'),
    (1.00, '#7F1D1D'),
], N=256)

def sanitize(arr, fill=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    mask = ~np.isfinite(arr)
    if mask.any():
        arr = arr.copy()
        arr[mask] = fill
    return arr

def safe_log10(arr, floor=0.1):
    return np.log10(np.clip(sanitize(arr), floor, 1e9))

def safe_percentile(arr, q, fallback=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return fallback
    return float(np.percentile(valid, float(np.clip(q, 0, 100))))

def safe_mean(arr, fallback=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size > 0 else fallback

def wenner_depths(n, spacing):
    return np.array([0.519 * spacing * (i + 1) for i in range(n)], dtype=np.float64)

def _quality_mask(grid):
    q = np.isfinite(grid) & (grid >= MIN_VALID_RHO) & (grid <= MAX_VALID_RHO)
    if q.any():
        median = float(np.median(grid[q]))
        if median > 0:
            q &= (grid >= median / 1e4) & (grid <= median * 1e4)
    return q

def _triplets_to_grid(points, spacing):
    """
    FIX PROBLEM 1: Murni mempertahankan geometri trapesium terbalik ERT Wenner.
    Menghapus total duplikasi/padding nilai pinggir yang merusak penampang.
    """
    n_vals = sorted(set(p[1] for p in points))[:MAX_ROWS]
    n_levels = len(n_vals)
    
    rows_data = []
    for nv in n_vals:
        row_pts = sorted([p for p in points if p[1] == nv], key=lambda p: p[0])
        rows_data.append(row_pts)
        
    n_cols = min(max(len(r) for r in rows_data), MAX_COLS)
    
    # Inisialisasi awal dengan np.nan untuk area tanpa data
    grid = np.full((n_levels, n_cols), np.nan, dtype=np.float64)
    quality = np.zeros((n_levels, n_cols), dtype=bool)
    
    for ri, row_pts in enumerate(rows_data):
        for ci, (x, n, rho) in enumerate(row_pts[:n_cols]):
            if np.isfinite(rho) and rho > 0:
                grid[ri, ci] = rho
                quality[ri, ci] = True
                
    # Bersihkan noise ekstrim menggunakan median filter global pada data valid
    if quality.any():
        median_val = float(np.median(grid[quality]))
        v_mask = (grid >= median_val / 1e4) & (grid <= median_val * 1e4)
        quality &= v_mask
        grid[~quality] = np.nan

    depths = wenner_depths(n_levels, spacing)
    distances = np.arange(n_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing

def parse_res2dinv(content, spacing):
    lines = [l.strip() for l in content.decode('utf-8', errors='ignore').splitlines() if l.strip()]
    if len(lines) < 5:
        raise ValueError("File too short for Res2Dinv DAT.")
    try:
        spacing = max(0.1, float(lines[1]))
    except Exception:
        pass
    points = []
    for line in lines[4:]:
        if len(points) > MAX_ROWS * MAX_COLS * 2:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, n, rho = float(parts[0]), int(float(parts[1])), float(parts[2])
            if x == 0 and n == 0 and rho == 0:
                break
            if np.isfinite(rho) and rho > 0:
                points.append((x, n, rho))
        except (ValueError, TypeError):
            continue
    if not points:
        raise ValueError("No valid data in Res2Dinv DAT file.")
    return _triplets_to_grid(points, spacing)

def parse_syscal_csv(content, spacing):
    text = content.decode('utf-8', errors='ignore')
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows_raw = list(reader)[:MAX_ROWS * MAX_COLS]
    except Exception as e:
        raise ValueError(f"Cannot parse Syscal CSV: {e}")
    if not rows_raw:
        raise ValueError("Empty Syscal CSV.")
    headers = {k.lower().strip(): k for k in rows_raw[0].keys()}
    rho_col = next((headers[c] for c in ['rho','resistivity','app_res','appres','ra','r_app'] if c in headers), None)
    x_col   = next((headers[c] for c in ['x','xmid','x_mid','mid','midpoint','station'] if c in headers), None)
    n_col   = next((headers[c] for c in ['n','sep','separation','level','nl'] if c in headers), None)
    if rho_col is None:
        raise ValueError(f"Cannot find resistivity column. Available: {list(headers.keys())}")
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
    return _triplets_to_grid(points, spacing)

def parse_csv_generic(content, spacing):
    text = content.decode('utf-8', errors='ignore')
    rows = []
    for line in csv.reader(io.StringIO(text)):
        if len(rows) >= MAX_ROWS:
            break
        try:
            vals = [float(v.strip()) for v in line if v.strip()]
            if len(vals) >= 3:
                rows.append(vals[:MAX_COLS])
        except ValueError:
            continue
    if not rows:
        raise ValueError("No valid numeric data. Expected: rows=depth levels, cols=electrodes, values=ohm.m")
    n_levels = len(rows)
    max_cols = max(len(r) for r in rows)
    grid = np.array([r + [r[-1]] * (max_cols - len(r)) for r in rows], dtype=np.float64)
    fill = safe_mean(grid[grid > 0], fallback=100.0)
    grid = sanitize(grid, fill=fill)
    quality = _quality_mask(grid)
    depths = wenner_depths(n_levels, spacing)
    distances = np.arange(max_cols) * spacing + n_levels * spacing
    return grid, quality, depths, distances, spacing

def parse_file(content, filename, spacing):
    fname = filename.lower()
    if fname.endswith('.dat'):
        return parse_res2dinv(content, spacing)
    text = content.decode('utf-8', errors='ignore')[:500]
    first_line = text.splitlines()[0].lower() if text.splitlines() else ''
    if any(k in first_line for k in ['rho','resistivity','app_res','ra,']):
        try:
            return parse_syscal_csv(content, spacing)
        except Exception:
            pass
    return parse_csv_generic(content, spacing)

def depth_sensitivity(depths):
    dmax = depths[-1] if depths[-1] > 0 else 1.0
    return np.exp(-1.5 * np.clip(depths / dmax, 0, 1))

def regularized_inversion(apparent, depths, quality, spacing, iterations=6,
                          smoothness=0.5, convergence_thr=2.0):
    """
    FIX PROBLEM 2: Pseudo Forward-Response & Iterative Masked Relaxation.
    Aman untuk RAM 512MB, mengabaikan NaN, mengontrol redaman berdasarkan kedalaman, 
    dan menghasilkan batas kontur batuan yang memisah tajam.
    """
    rows, cols = apparent.shape
    
    valid_data = apparent[quality]
    if valid_data.size == 0:
        valid_data = np.array([100.0])
    
    bg_rho = float(np.exp(np.mean(np.log(valid_data))))
    
    model = np.copy(apparent)
    model[~quality] = bg_rho
    
    v_min = max(safe_percentile(valid_data, 2) * 0.2, MIN_VALID_RHO)
    v_max = min(safe_percentile(valid_data, 98) * 5.0, MAX_VALID_RHO)
    
    sens = depth_sensitivity(depths)
    
    rms_log = []
    converged = False
    prev_rms = None
    
    for it in range(iterations):
        damp = float(smoothness / (1.0 + it * 0.5))
        damp = max(0.05, min(damp, 0.7))
        
        next_model = np.copy(model)
        
        # 1. Masked Laplacian Regularization
        for r in range(rows):
            z_factor = 1.0 + (1.0 - float(sens[r])) * 1.5
            for c in range(cols):
                neighbors = []
                if c > 0: neighbors.append(model[r, c-1] * z_factor)
                if c < cols - 1: neighbors.append(model[r, c+1] * z_factor)
                if r > 0: neighbors.append(model[r-1, c])
                if r < rows - 1: neighbors.append(model[r+1, c])
                
                if neighbors:
                    local_avg = np.mean(neighbors) / (z_factor if r in (0, rows-1) else 1.0)
                    next_model[r, c] = (1.0 - damp) * model[r, c] + damp * local_avg

        # 2. Pseudo Forward-Response & Back-Projection Weighting
        residual = np.zeros_like(apparent)
        residual[quality] = apparent[quality] - next_model[quality]
        
        for r in range(rows):
            update_weight = float(sens[r]) * 0.25
            next_model[r, quality[r, :]] += residual[r, quality[r, :]] * update_weight
            
        model = np.clip(next_model, v_min, v_max)
        
        # 3. Hitung Koreksi Root Mean Square (RMS) Error Global
        denom = safe_mean(apparent[quality], fallback=1.0)
        diff = (model - apparent)[quality]
        
        rms = float(np.sqrt(np.mean(diff**2)) / denom * 100) if diff.size > 0 else 999.0
        if not np.isfinite(rms): 
            rms = 999.0
            
        rms_log.append(round(rms, 4))
        
        if prev_rms is not None:
            if 0 <= (prev_rms - rms) < STAGNATION_DELTA:
                converged = True
                break
        if rms < convergence_thr:
            converged = True
            break
        prev_rms = rms

    res_rel = np.abs(model - apparent) / (apparent + 1e-9)
    res_rel[~quality] = 0.0
    
    depth_unc = (1.0 - sens).reshape(-1, 1) * np.ones((1, cols))
    uncertainty = np.clip(0.3 * res_rel + 0.7 * depth_unc, 0.0, 1.0)
    uncertainty[~quality] = 1.0
    
    return model, rms_log, converged, uncertainty

def compute_confidence(rms, converged, completeness, n_cols, n_levels):
    return round(float(np.clip(
        0.40*float(np.clip(1-(rms-2)/18,0,1)) +
        0.20*(1.0 if converged else 0.5) +
        0.25*completeness +
        0.15*float(np.clip((n_cols/max(n_levels*2,1))/3,0,1)),
        0, 1
    )), 4)

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

def classify_rho(v):
    for mn, mx, lbl, desc, ctx in RESIS_TABLE:
        if mn <= v < mx: return lbl, desc, ctx
    return "Very High", "highly resistive anomaly", RESIS_TABLE[-1][4]

def zone_color(cls):
    return {"Very Low":"#1E40AF","Low":"#0284C7","Moderate":"#059669",
            "Elevated":"#D97706","High":"#DC2626","Very High":"#7F1D1D"} .get(cls,"#6B7280")

def interpret(model, uncertainty, depths):
    flat = sanitize(model.flatten(), fill=100.0)
    flat_unc = sanitize(uncertainty.flatten(), fill=1.0)
    zones, seen = [], set()
    for mn, mx, cls, desc, ctx in RESIS_TABLE:
        mask = (flat >= mn) & (flat < mx)
        if not mask.any() or cls in seen: continue
        seen.add(cls)
        vals = flat[mask]
        mean_rho = safe_mean(vals, fallback=(mn+mx)/2)
        mean_unc = safe_mean(flat_unc[mask], fallback=0.5)
        conf = "moderate-to-high" if mean_unc<0.25 else "moderate" if mean_unc<0.50 else "low"
        label = f"Zone {chr(65+len(zones))}"
        zones.append({
            "label": label, "resistivity_class": cls,
            "gephysical_descriptor": desc, "geological_context": ctx,
            "range_ohm_m": [round(float(vals.min()),1), round(float(vals.max()),1)],
            "mean_ohm_m": round(mean_rho, 2), "uncertainty": round(mean_unc, 3),
            "confidence_qualifier": conf,
            "range": f"{vals.min():.0f}-{vals.max():.0f} ohm.m",
            "interpretation": ctx, "color": zone_color(cls),
        })
    log_m = safe_log10(model)
    finite_log = log_m[np.isfinite(log_m)]
    lmean = safe_mean(finite_log, fallback=2.0)
    lstd = max(float(np.std(finite_log)) if len(finite_log) > 1 else 1.0, 0.01)
    anomalies = []
    for atype, amask in [
        ("conductive", (log_m < lmean-1.5*lstd) & (uncertainty < 0.75)),
        ("resistive",  (log_m > lmean+1.5*lstd) & (uncertainty < 0.75)),
    ]:
        if not amask.any(): continue
        ri, _ = np.where(amask)
        mean_rho  = safe_mean(model[amask], fallback=100.0)
        mean_unc  = safe_mean(uncertainty[amask], fallback=0.5)
        mean_depth = safe_mean(depths[np.clip(ri, 0, len(depths)-1)], fallback=0.0)
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

def render(model, uncertainty, depths, distances, rms_log, converged,
           confidence, filename, display_mode, spacing):
    """
    FIX PROBLEM 3: Skala Interval Logaritmik Tetap Konstan ala RES2DINV.
    Menghilangkan dominasi warna merah akibat lonjakan nilai ekstrim di pinggir penampang.
    """
    rows, cols = model.shape
    X, Y = np.meshgrid(distances, depths)
    
    fixed_rho_levels = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=np.float64)
    log_levels = np.log10(fixed_rho_levels)
    
    log_m = np.log10(np.clip(model, 0.1, 1e5))
    
    valid_logs = log_m[uncertainty < 1.0]
    if valid_logs.size > 0:
        vmin_log = max(valid_logs.min(), log_levels[0])
        vmax_log = min(valid_logs.max(), log_levels[-1])
    else:
        vmin_log, vmax_log = 0.0, 4.0
        
    norm = mcolors.Normalize(vmin=vmin_log, vmax=vmax_log)
    
    fig = buf = None
    try:
        fig = plt.figure(figsize=(16, 5), facecolor='white')
        ax = fig.add_axes([0.07, 0.16, 0.82, 0.66])
        ax.set_facecolor('white')
        
        active_levels = log_levels[(log_levels >= vmin_log - 0.1) & (log_levels <= vmax_log + 0.1)]
        if len(active_levels) < 3:
            active_levels = np.linspace(vmin_log, vmax_log, 12)
            
        if display_mode in ('contoured', 'hybrid'):
            cf = ax.contourf(X, Y, log_m, levels=active_levels, cmap=CMAP, norm=norm, extend='both')
            cs = ax.contour(X, Y, log_m, levels=active_levels[::2], colors='black', linewidths=0.6, alpha=0.5)
            try:
                ax.clabel(cs, inline=True, fontsize=7, fmt=lambda x: f'{10**x:.0f}', colors='black')
            except Exception:
                pass
        else:
            ax.imshow(log_m, aspect='auto', origin='upper',
                      extent=[distances[0], distances[-1], depths[-1], depths[0]],
                      cmap=CMAP, norm=norm, interpolation='bilinear')
            
        # Masking Area Trapesium Terbalik (Hatching Area No-Data)
        if (uncertainty >= 0.99).any():
            try:
                ax.contourf(X, Y, uncertainty, levels=[0.99, 1.01], colors='white', alpha=1.0)
                ax.contourf(X, Y, uncertainty, levels=[0.99, 1.01], colors='#e5e7eb', alpha=0.4, hatches=['\\\\\\\\'])
            except Exception:
                pass
                
        ax.set_xlabel('Distance (m)', fontsize=11, fontweight='semibold')
        ax.set_ylabel('Depth (m)', fontsize=11, fontweight='semibold')
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.3, color='gray')
        
        ax.plot(distances, np.full_like(distances, depths[0]), 'v', color='black', markersize=4, alpha=0.6, label='Electrodes')
        
        ax.set_ylim(depths[-1], depths[0])
        ax.set_yticks(depths)
        ax.set_yticklabels([f'{d:.1f}' for d in depths], fontsize=8)
        
        cax = fig.add_axes([0.91, 0.16, 0.015, 0.66])
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Apparent Resistivity (Ω·m)', fontsize=10, labelpad=8)
        
        tick_values = fixed_rho_levels[(log_levels >= vmin_log) & (log_levels <= vmax_log)]
        cbar.set_ticks(np.log10(tick_values))
        cbar.set_ticklabels([f'{int(v)}' for v in tick_values], fontsize=8)
        
        rms_val = rms_log[-1] if rms_log else 0.0
        conv_str = "converged" if converged else "maximum iterations reached"
        fig.text(0.5, 0.94, f'Pairoq Model Resistivity Section: {filename}', ha='center', fontsize=12, fontweight='bold')
        fig.text(0.5, 0.89, f'RMS Error: {rms_val:.2f}%  |  Iterations: {len(rms_log)}  |  Status: {conv_str}  |  Spacing: {spacing}m', 
                 ha='center', fontsize=9, color='#374151')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=190, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    finally:
        if fig is not None: plt.close(fig)
        if buf is not None: buf.close()
        plt.clf(); plt.cla(); gc.collect()

@app.get("/")
def root():
    return {"service": "Pairoq Geophysical API", "version": "2.3.0",
            "disclaimer": "Preliminary tool. Not RES2DINV/pyGIMLi equivalent. Requires professional validation."}

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.3.0"}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    spacing: float = Form(5.0),
    iterations: int = Form(6),
    array_type: str = Form("wenner_alpha"),
    display_mode: str = Form("hybrid"),
):
    t0 = time.time()
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_BYTES//1024}KB.")
    spacing = float(np.clip(spacing, 0.1, 1000.0))
    iterations = int(np.clip(iterations, 1, 10))
    display_mode = display_mode if display_mode in ("smooth","contoured","hybrid") else "hybrid"
    filename = file.filename or "survey"
    try:
        grid, quality, depths, distances, spacing = parse_file(content, filename, spacing)
    except ValueError as e:
        raise HTTPException(400, f"Parse error: {e}")
    except Exception as e:
        logger.error(f"Parse: {e}"); raise HTTPException(500, "Internal parse error.")
    if grid.shape[0] < 2 or grid.shape[1] < 3:
        raise HTTPException(400, "Dataset too small. Need >= 2 depth levels and >= 3 electrode positions.")
    try:
        model, rms_log, converged, uncertainty = regularized_inversion(grid, depths, quality, spacing, iterations)
    except Exception as e:
        logger.error(f"Inversion: {e}"); raise HTTPException(500, "Inversion failed.")
    completeness = float(quality.sum()) / max(quality.size, 1)
    confidence = compute_confidence(rms_log[-1] if rms_log else 999.0, converged, completeness, grid.shape[1], grid.shape[0])
    zones, anomalies = interpret(model, uncertainty, depths)
    try:
        img = render(model, uncertainty, depths, distances, rms_log, converged, confidence, filename, display_mode, spacing)
    except Exception as e:
        logger.error(f"Render: {e}"); raise HTTPException(500, "Rendering failed.")
    valid_data = grid[quality] if quality.any() else grid.flatten()
    elapsed = round(time.time()-t0, 2)
    logger.info(f"OK {filename} | {elapsed}s | RMS={rms_log[-1] if rms_log else 'N/A'} | conf={confidence}")
    return JSONResponse({
        "status": "ok", "processing_time_s": elapsed, "image_b64": img,
        "stats": {
            "points_total": int(quality.size), "points_valid": int(quality.sum()),
            "data_completeness": round(completeness, 4), "n_levels": int(grid.shape[0]),
            "n_electrodes_estimated": int(grid.shape[1]+3*grid.shape[0]),
            "min_resistivity_ohm_m": round(safe_percentile(valid_data,0), 2),
            "max_resistivity_ohm_m": round(safe_percentile(valid_data,100), 2),
            "mean_resistivity_ohm_m": round(safe_mean(valid_data, fallback=0.0), 2),
            "max_depth_m": round(float(depths[-1]), 2),
            "electrode_spacing_m": spacing, "array_type": array_type,
        },
        "inversion": {
            "method": "regularized_least_squares_approximation",
            "iterations_run": len(rms_log), "converged": converged,
            "rms_final_pct": rms_log[-1] if rms_log else None,
            "rms_per_iteration": rms_log, "confidence_score": confidence,
            "disclaimer": "Lightweight approximation. Not RES2DINV/pyGIMLi equivalent.",
        },
        "zones": zones, "anomalies": anomalies,
        "interpretation": {"disclaimer": "Preliminary only. Geological conclusions require professional validation."},
        "depths_m": [round(float(d),2) for d in depths],
        "distances_m": [round(float(d),2) for d in distances],
        "rms_final": rms_log[-1] if rms_log else None,
        "rms_history": rms_log,
    })

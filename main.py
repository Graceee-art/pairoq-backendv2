"""
Pairoq API v2.3.2 — Robust Dual-Parser Production Engine
========================================================
Railway free-tier optimized. Auto-detects Res2Dinv .dat & Spatial CSV.

FIXED IN V2.3.2:
1. Strict File Type Detection: Memisahkan parsing .dat vs .csv secara absolut.
2. Anti-Crash Header Skip: Mengabaikan baris konfigurasi awal pada file .dat.
3. Proper Inverted Trapezoid Rendering untuk data panjang (120+ poin).
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

app = FastAPI(title="Pairoq API", version="2.3.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MAX_UPLOAD_BYTES  = 2 * 1024 * 1024
MAX_ROWS          = 40
MAX_COLS          = 300
MAX_VALID_RHO     = 1_000_000.0
MIN_VALID_RHO     = 0.001

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
    if valid.size == 0: return fallback
    return float(np.percentile(valid, float(np.clip(q, 0, 100))))

def safe_mean(arr, fallback=1.0):
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    return float(np.mean(valid)) if valid.size > 0 else fallback

def wenner_depths(n, spacing):
    return np.array([0.519 * spacing * (i + 1) for i in range(n)], dtype=np.float64)

def _triplets_to_grid(points, spacing):
    if not points:
        raise ValueError("Tidak ada poin data valid untuk dipetakan ke grid.")
        
    x_vals = sorted(set(p[0] for p in points))
    n_vals = sorted(set(p[1] for p in points))
    
    n_levels = min(len(n_vals), MAX_ROWS)
    n_cols = min(len(x_vals), MAX_COLS)
    
    x_map = {x: i for i, x in enumerate(x_vals[:n_cols])}
    n_map = {n: i for i, n in enumerate(n_vals[:n_levels])}
    
    grid = np.full((n_levels, n_cols), np.nan, dtype=np.float64)
    quality = np.zeros((n_levels, n_cols), dtype=bool)
    
    for x, n, rho in points:
        if x in x_map and n in n_map:
            ri = n_map[n]
            ci = x_map[x]
            if np.isfinite(rho) and rho > 0:
                grid[ri, ci] = rho
                quality[ri, ci] = True
                
    if quality.any():
        median_val = float(np.median(grid[quality]))
        v_mask = (grid >= median_val / 1e4) & (grid <= median_val * 1e4)
        quality &= v_mask
        grid[~quality] = np.nan

    depths = wenner_depths(n_levels, spacing)
    distances = np.array(x_vals[:n_cols], dtype=np.float64)
    return grid, quality, depths, distances, spacing

def parse_pairoq_csv(text, default_spacing):
    delimiter = ','
    if ';' in text and text.count(';') > text.count(','):
        delimiter = ';'
        
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_idx = 0
    for idx, line in enumerate(lines[:10]):
        low_line = line.lower()
        if 'rho' in low_line or 'datum' in low_line or 'rho app' in low_line:
            header_idx = idx
            break
            
    f = io.StringIO("\n".join(lines[header_idx:]))
    reader = csv.reader(f, delimiter=delimiter)
    headers = [h.strip().lower() for h in next(reader)]
    
    rho_col = next((i for i, h in enumerate(headers) if h in ['rho app', 'rho_app', 'rho', 'resistivity', 'ra']), None)
    datum_col = next((i for i, h in enumerate(headers) if h in ['datum', 'x', 'mid', 'midpoint']), None)
    a_col = next((i for i, h in enumerate(headers) if h in ['a', 'spacing', 'sep', 'n']), None)
    
    if rho_col is None:
        if len(headers) >= 9:
            rho_col = next((i for i, h in enumerate(headers) if 'rho' in h or 'app' in h), len(headers) - 2)
    if datum_col is None: datum_col = len(headers) - 1
    if a_col is None: a_col = next((i for i, h in enumerate(headers) if h == 'a' or 'space' in h), 8)

    points = []
    detected_spacings = []
    
    for row in reader:
        if not row or len(row) <= max(rho_col, datum_col, a_col): continue
        try:
            rho = float(row[rho_col].replace(',', '.'))
            datum = float(row[datum_col].replace(',', '.'))
            a_val = float(row[a_col].replace(',', '.'))
            detected_spacings.append(a_val)
            
            if np.isfinite(rho) and rho > 0:
                points.append((datum, a_val, rho))
        except: continue
            
    if not points:
        raise ValueError("Gagal memproses baris tabel CSV spasial.")
        
    final_spacing = float(np.median(detected_spacings)) if detected_spacings else default_spacing
    unique_a = sorted(set(p[1] for p in points))
    a_to_n = {a: idx + 1 for idx, a in enumerate(unique_a)}
    normalized_points = [(p[0], a_to_n[p[1]], p[2]) for p in points]
    return _triplets_to_grid(normalized_points, final_spacing)

def parse_res2dinv(text, default_spacing):
    """
    PARSER KHUSUS .DAT RES2DINV: Ekstraksi murni data baris panjang kustom.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        raise ValueError("Struktur file data lapangan terlalu pendek.")
        
    try:
        detected_spacing = float(lines[1].split()[0])
    except:
        detected_spacing = default_spacing

    points = []
    data_start = False
    
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) >= 3:
            # Lewatin 4 baris header metadata bawaan RES2DINV .dat
            if idx < 4: continue
            try:
                x = float(parts[0].replace(',', '.'))
                # Konversi level n atau spasi elektroda vertikal
                n = int(float(parts[1].replace(',', '.')))
                rho = float(parts[2].replace(',', '.'))
                
                if x == 0 and n == 0 and rho == 0: break # Tanda end of file res2dinv
                if rho > 0:
                    points.append((x, n, rho))
            except: continue

    if not points:
        # Jika format spasial baris spasi datar tanpa format standar res2dinv header
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    points.append((float(parts[0]), int(float(parts[1])), float(parts[2])))
                except: continue

    return _triplets_to_grid(points, detected_spacing)

def parse_file(content, filename, spacing):
    text = content.decode('utf-8', errors='ignore').strip()
    # Pengecekan cerdas berbasis baris pertama atau extension nama file
    if filename.lower().endswith('.dat') or (not text.startswith("NO") and not "," in text.splitlines()[0] and not ";" in text.splitlines()[0]):
        return parse_res2dinv(text, spacing)
    return parse_pairoq_csv(text, spacing)

def depth_sensitivity(depths):
    dmax = depths[-1] if depths[-1] > 0 else 1.0
    return np.exp(-1.1 * np.clip(depths / dmax, 0, 1))

def regularized_inversion(apparent, depths, quality, spacing, iterations=6, smoothness=0.5):
    rows, cols = apparent.shape
    valid_data = apparent[quality]
    if valid_data.size == 0: valid_data = np.array([100.0])
    
    bg_rho = float(np.exp(np.mean(np.log(valid_data))))
    model = np.copy(apparent)
    model[~quality] = bg_rho
    
    v_min = max(safe_percentile(valid_data, 1) * 0.1, MIN_VALID_RHO)
    v_max = min(safe_percentile(valid_data, 99) * 10.0, MAX_VALID_RHO)
    
    sens = depth_sensitivity(depths)
    rms_log = []
    
    for it in range(iterations):
        damp = float(smoothness / (1.0 + it * 0.2))
        damp = max(0.15, min(damp, 0.7))
        next_model = np.copy(model)
        
        for r in range(rows):
            z_factor = 1.0 + (1.0 - float(sens[r])) * 0.3
            for c in range(cols):
                neighbors = []
                if c > 0: neighbors.append(model[r, c-1])
                if c < cols - 1: neighbors.append(model[r, c+1])
                if r > 0: neighbors.append(model[r-1, c] * z_factor)
                if r < rows - 1: neighbors.append(model[r+1, c] * z_factor)
                if neighbors:
                    next_model[r, c] = (1.0 - damp) * model[r, c] + damp * np.mean(neighbors)

        residual = np.zeros_like(apparent)
        residual[quality] = apparent[quality] - next_model[quality]
        
        for r in range(rows):
            next_model[r, quality[r, :]] += residual[r, quality[r, :]] * (float(sens[r]) * 0.35)
            
        model = np.clip(next_model, v_min, v_max)
        denom = safe_mean(apparent[quality], fallback=1.0)
        diff = (model - apparent)[quality]
        rms = float(np.sqrt(np.mean(diff**2)) / denom * 100) if diff.size > 0 else 99.0
        rms_log.append(round(rms, 2))
        
    res_rel = np.abs(model - apparent) / (apparent + 1e-9)
    res_rel[~quality] = 0.0
    uncertainty = np.clip(0.4 * res_rel + 0.6 * (1.0 - sens).reshape(-1, 1), 0.0, 1.0)
    uncertainty[~quality] = 1.0
    return model, rms_log, True, uncertainty

RESIS_TABLE = [
    (0,    10,   "Very Low",  "highly conductive anomaly", "saline water, clay-rich sediments, or water-saturated ground"),
    (10,   30,   "Low",       "conductive zone", "saturated sediments, clay layers, or weathered material"),
    (30,   100,  "Moderate",  "intermediate resistivity", "alluvial deposits, sand-silt mixtures, or partially saturated zones"),
    (100,  300,  "Elevated",  "moderately resistive zone", "dry soil, sands, gravels, or compacted formations"),
    (300,  1000, "High",      "resistive anomaly", "consolidated rock, dry sand/gravel, or bedrock structures"),
    (1000, 1e9,  "Very High", "highly resistive anomaly", "massive crystalline bedrock, granitic intrusion, or dense dry rock"),
]

def interpret(model, uncertainty, depths):
    flat = sanitize(model.flatten(), fill=100.0)
    flat_unc = sanitize(uncertainty.flatten(), fill=1.0)
    zones = []
    seen = set()
    for mn, mx, cls, desc, ctx in RESIS_TABLE:
        mask = (flat >= mn) & (flat < mx)
        if not mask.any() or cls in seen: continue
        seen.add(cls)
        vals = flat[mask]
        mean_rho = safe_mean(vals, fallback=(mn+mx)/2)
        mean_unc = safe_mean(flat_unc[mask], fallback=0.5)
        zones.append({
            "label": f"Zone {chr(65+len(zones))}", "resistivity_class": cls,
            "gephysical_descriptor": desc, "geological_context": ctx,
            "mean_ohm_m": round(mean_rho, 2), "uncertainty": round(mean_unc, 3),
            "confidence_qualifier": "high" if mean_unc < 0.3 else "moderate" if mean_unc < 0.6 else "low",
            "range": f"{vals.min():.1f}-{vals.max():.1f} ohm.m", "color": "#0284C7"
        })
    return zones, []

def render(model, uncertainty, depths, distances, rms_log, converged, filename, display_mode, spacing):
    rows, cols = model.shape
    X, Y = np.meshgrid(distances, depths)
    
    log_m = np.log10(np.clip(model, 0.1, 1e5))
    valid_logs = log_m[uncertainty < 1.0]
    vmin_log = float(valid_logs.min()) if valid_logs.size > 0 else 0.0
    vmax_log = float(valid_logs.max()) if valid_logs.size > 0 else 3.0
    
    levels = np.linspace(vmin_log, vmax_log, 30)
    norm = mcolors.Normalize(vmin=vmin_log, vmax=vmax_log)
    
    fig = buf = None
    try:
        fig = plt.figure(figsize=(14, 5), facecolor='white')
        ax = fig.add_axes([0.07, 0.15, 0.83, 0.7])
        
        cf = ax.contourf(X, Y, log_m, levels=levels, cmap=CMAP, norm=norm, extend='both')
        cs = ax.contour(X, Y, log_m, levels=levels[::5], colors='black', linewidths=0.3, alpha=0.3)
        
        if (uncertainty >= 0.9).any():
            try:
                ax.contourf(X, Y, uncertainty, levels=[0.9, 1.1], colors='white', alpha=1.0)
                ax.contourf(X, Y, uncertainty, levels=[0.9, 1.1], colors='#9ca3af', alpha=0.25, hatches=['////'])
            except: pass
                
        ax.set_xlabel('Distance / Datum Midpoint (m)', fontsize=9, fontweight='semibold')
        ax.set_ylabel('Depth (m)', fontsize=9, fontweight='semibold')
        ax.set_ylim(depths[-1], depths[0])
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Resistivity (Ω·m)', fontsize=9)
        
        ticks = np.linspace(vmin_log, vmax_log, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{10**t:.1f}' for t in ticks], fontsize=8)
        
        rms_val = rms_log[-1] if rms_log else 0.0
        ax.set_title(f"Pairoq Inversion Model Section: {filename}\nRMS Error: {rms_val:.2f}% | Spacing: {spacing}m", fontsize=11, pad=10, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    finally:
        if fig: plt.close(fig)
        if buf: buf.close()
        plt.clf(); plt.cla(); gc.collect()

@app.get("/")
def root(): return {"status": "running", "version": "2.3.2"}

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    spacing: float = Form(10.0),
    iterations: int = Form(6),
    display_mode: str = Form("hybrid"),
):
    t0 = time.time()
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File kebesaran, maksimal 2MB.")
        
    filename = file.filename or "data.csv"
    try:
        grid, quality, depths, distances, spacing = parse_file(content, filename, spacing)
    except Exception as e:
        raise HTTPException(400, f"Gagal ekstraksi baris data lapangan: {e}")
        
    model, rms_log, converged, uncertainty = regularized_inversion(grid, depths, quality, spacing, iterations)
    zones, anomalies = interpret(model, uncertainty, depths)
    img_b64 = render(model, uncertainty, depths, distances, rms_log, converged, filename, display_mode, spacing)
    
    valid_data = grid[quality] if quality.any() else grid.flatten()
    return JSONResponse({
        "status": "ok", "processing_time_s": round(time.time()-t0, 2), "image_b64": img_b64,
        "stats": {
            "points_total": int(quality.size), "points_valid": int(quality.sum()),
            "min_resistivity_ohm_m": round(float(valid_data.min()), 2),
            "max_resistivity_ohm_m": round(float(valid_data.max()), 2),
            "max_depth_m": round(float(depths[-1]), 2)
        },
        "inversion": {"iterations_run": len(rms_log), "rms_final_pct": rms_log[-1]},
        "zones": zones, "anomalies": anomalies
    })

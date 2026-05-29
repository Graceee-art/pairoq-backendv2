"""
Pairoq API v2.3.1 — Production Ready Fix (Coordinate-Aware Parser)
==================================================================
Railway free-tier (512MB RAM). Single-file. No compiled deps.

FIXED IN V2.3.1:
1. Intelligent Coordinate Mapping: Membaca kolom 'Rho app'/'Datum'/'a' secara spasial.
2. Menghentikan penumpukan baris homogen (Efek Lapis Legit hancur total!).
3. True Wenner inverted trapezoid geometry dengan arsiran no-data.
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

app = FastAPI(title="Pairoq API", version="2.3.1")
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

def _triplets_to_grid(points, spacing):
    """
    Mengubah koordinat (X, Depth_Level, Rho) menjadi matriks 2D spasial sejati.
    """
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

def parse_pairoq_csv(content, default_spacing):
    """
    PARSER FIX: Membaca tabel kolom spasial seperti gambar tabel milik user.
    """
    text = content.decode('utf-8', errors='ignore')
    
    # Deteksi delimiter (koma atau titik koma atau tab)
    delimiter = ','
    if ';' in text and text.count(';') > text.count(','):
        delimiter = ';'
        
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("File kosong.")
        
    # Cari baris header
    header_idx = 0
    for idx, line in enumerate(lines[:10]):
        low_line = line.lower()
        if 'rho' in low_line or 'datum' in low_line or 'rho app' in low_line:
            header_idx = idx
            break
            
    f = io.StringIO("\n".join(lines[header_idx:]))
    reader = csv.reader(f, delimiter=delimiter)
    headers = [h.strip().lower() for h in next(reader)]
    
    # Mapping indeks kolom
    rho_col = next((i for i, h in enumerate(headers) if h in ['rho app', 'rho_app', 'rho', 'resistivity', 'ra']), None)
    datum_col = next((i for i, h in enumerate(headers) if h in ['datum', 'x', 'mid', 'midpoint']), None)
    a_col = next((i for i, h in enumerate(headers) if h in ['a', 'spacing', 'sep', 'n']), None)
    
    # Fallback gila jika header berantakan tapi kolomnya cocok dengan gambar user
    if rho_col is None:
        if len(headers) >= 9: # Sesuai struktur gambar (NO, A, B, M, N, V, I, R, a, k, Rho app, Datum)
            rho_col = next((i for i, h in enumerate(headers) if 'rho' in h or 'app' in h), len(headers) - 2)
    if datum_col is None:
        datum_col = len(headers) - 1
    if a_col is None:
        a_col = next((i for i, h in enumerate(headers) if h == 'a' or 'space' in h), 8)

    points = []
    detected_spacings = []
    
    for row in reader:
        if not row or len(row) <= max(rho_col, datum_col, a_col):
            continue
        try:
            rho = float(row[rho_col].replace(',', '.'))
            datum = float(row[datum_col].replace(',', '.'))
            
            # Hitung tingkat kedalaman pseudo-level (N) dari spasi elektroda 'a'
            a_val = float(row[a_col].replace(',', '.'))
            detected_spacings.append(a_val)
            
            if np.isfinite(rho) and rho > 0:
                points.append((datum, a_val, rho))
        except (ValueError, IndexError, TypeError):
            continue
            
    if not points:
        raise ValueError("Gagal mengekstrak koordinat spasial dari tabel CSV.")
        
    final_spacing = float(np.median(detected_spacings)) if detected_spacings else default_spacing
    
    # Normalisasi pseudo-level spasi menjadi indeks urutan kedalaman 1, 2, 3...
    unique_a = sorted(set(p[1] for p in points))
    a_to_n = {a: idx + 1 for idx, a in enumerate(unique_a)}
    
    normalized_points = [(p[0], a_to_n[p[1]], p[2]) for p in points]
    return _triplets_to_grid(normalized_points, final_spacing)

def parse_res2dinv(content, spacing):
    lines = [l.strip() for l in content.decode('utf-8', errors='ignore').splitlines() if l.strip()]
    if len(lines) < 5:
        raise ValueError("File DAT terlalu pendek.")
    try:
        spacing = max(0.1, float(lines[1]))
    except Exception:
        pass
    points = []
    for line in lines[4:]:
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
    return _triplets_to_grid(points, spacing)

def parse_file(content, filename, spacing):
    fname = filename.lower()
    if fname.endswith('.dat'):
        return parse_res2dinv(content, spacing)
    # Gunakan parser cerdas spasial untuk CSV data lapangan user
    return parse_pairoq_csv(content, spacing)

def depth_sensitivity(depths):
    dmax = depths[-1] if depths[-1] > 0 else 1.0
    return np.exp(-1.2 * np.clip(depths / dmax, 0, 1))

def regularized_inversion(apparent, depths, quality, spacing, iterations=6, smoothness=0.6):
    """
    Inversion Engine: Menggunakan relaksasi iteratif menyamping (anti-lapis legit).
    """
    rows, cols = apparent.shape
    valid_data = apparent[quality]
    if valid_data.size == 0:
        valid_data = np.array([100.0])
    
    bg_rho = float(np.exp(np.mean(np.log(valid_data))))
    model = np.copy(apparent)
    model[~quality] = bg_rho
    
    v_min = max(safe_percentile(valid_data, 1) * 0.1, MIN_VALID_RHO)
    v_max = min(safe_percentile(valid_data, 99) * 10.0, MAX_VALID_RHO)
    
    sens = depth_sensitivity(depths)
    rms_log = []
    converged = False
    
    for it in range(iterations):
        damp = float(smoothness / (1.0 + it * 0.2))
        damp = max(0.15, min(damp, 0.8))
        
        next_model = np.copy(model)
        
        # Laplacian smoothing yang seimbang antara horizontal dan vertikal
        for r in range(rows):
            z_factor = 1.0 + (1.0 - float(sens[r])) * 0.4
            for c in range(cols):
                neighbors = []
                if c > 0: neighbors.append(model[r, c-1])
                if c < cols - 1: neighbors.append(model[r, c+1])
                if r > 0: neighbors.append(model[r-1, c] * z_factor)
                if r < rows - 1: neighbors.append(model[r+1, c] * z_factor)
                
                if neighbors:
                    next_model[r, c] = (1.0 - damp) * model[r, c] + damp * np.mean(neighbors)

        # Back-projection sisa nilai residual lapangan
        residual = np.zeros_like(apparent)
        residual[quality] = apparent[quality] - next_model[quality]
        
        for r in range(rows):
            update_weight = float(sens[r]) * 0.35
            next_model[r, quality[r, :]] += residual[r, quality[r, :]] * update_weight
            
        model = np.clip(next_model, v_min, v_max)
        
        # Hitung Error RMS Global
        denom = safe_mean(apparent[quality], fallback=1.0)
        diff = (model - apparent)[quality]
        rms = float(np.sqrt(np.mean(diff**2)) / denom * 100) if diff.size > 0 else 999.0
        rms_log.append(round(rms, 4))
        
        if rms < 3.0:
            converged = True
            break
            
    res_rel = np.abs(model - apparent) / (apparent + 1e-9)
    res_rel[~quality] = 0.0
    uncertainty = np.clip(0.4 * res_rel + 0.6 * (1.0 - sens).reshape(-1, 1), 0.0, 1.0)
    uncertainty[~quality] = 1.0
    
    return model, rms_log, converged, uncertainty

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
    zones, seen = [], set()
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
            "range": f"{vals.min():.1f}-{vals.max():.1f} ohm.m", "color": {"Very Low":"#1E40AF","Low":"#0284C7","Moderate":"#059669","Elevated":"#D97706","High":"#DC2626","Very High":"#7F1D1D"}.get(cls,"#6B7280")
        })
    return zones, []

def render(model, uncertainty, depths, distances, rms_log, converged, filename, display_mode, spacing):
    """
    Rendering Engine: Membuat grafik kontur halus dan mengarsir area kosong (Inverted Trapezoid).
    """
    rows, cols = model.shape
    X, Y = np.meshgrid(distances, depths)
    
    # Ambil sebaran data riil log10
    log_m = np.log10(np.clip(model, 0.1, 1e5))
    valid_logs = log_m[uncertainty < 1.0]
    
    vmin_log = float(valid_logs.min()) if valid_logs.size > 0 else 0.0
    vmax_log = float(valid_logs.max()) if valid_logs.size > 0 else 3.0
    
    # Jamin minimal ada 20 level kontur halus untuk menghindari patahan blok warna kaku
    levels = np.linspace(vmin_log, vmax_log, 25)
    if len(levels) < 5: levels = 25
    
    norm = mcolors.Normalize(vmin=vmin_log, vmax=vmax_log)
    
    fig = buf = None
    try:
        fig = plt.figure(figsize=(15, 5), facecolor='white')
        ax = fig.add_axes([0.06, 0.15, 0.84, 0.68])
        
        # Plot kontur utama dengan gradasi warna melengkung yang rapat
        cf = ax.contourf(X, Y, log_m, levels=levels, cmap=CMAP, norm=norm, extend='both')
        cs = ax.contour(X, Y, log_m, levels=levels[::4], colors='black', linewidths=0.3, alpha=0.3)
        try:
            ax.clabel(cs, inline=True, fontsize=7, fmt=lambda x: f'{10**x:.0f}')
        except: pass
            
        # MASKING TRApesium TERBALIK: Mengarsir area tepi bawah yang tidak dilewati arus listrik
        if (uncertainty >= 0.95).any():
            try:
                ax.contourf(X, Y, uncertainty, levels=[0.95, 1.05], colors='white', alpha=1.0)
                ax.contourf(X, Y, uncertainty, levels=[0.95, 1.05], colors='#9ca3af', alpha=0.3, hatches=['////'])
            except: pass
                
        ax.set_xlabel('Distance / Datum (m)', fontsize=10, fontweight='semibold')
        ax.set_ylabel('Depth (m)', fontsize=10, fontweight='semibold')
        ax.set_ylim(depths[-1], depths[0])
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        
        # Tarik Colorbar di sisi kanan
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.68])
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Resistivity (Ω·m)', fontsize=9)
        
        # Ubah label ticks eksponen log10 menjadi angka normal (10, 100, 1000)
        ticks = np.linspace(vmin_log, vmax_log, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{10**t:.1f}' for t in ticks], fontsize=8)
        
        rms_val = rms_log[-1] if rms_log else 0.0
        ax.set_title(f"Pairoq Model Resistivity Section: {filename}\nRMS Error: {rms_val:.2f}% | Spacing: {spacing}m", fontsize=11, pad=12, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    finally:
        if fig: plt.close(fig)
        if buf: buf.close()
        plt.clf(); plt.cla(); gc.collect()

@app.get("/")
def root(): return {"status": "running", "version": "2.3.1"}

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
        raise HTTPException(413, "File terlalu besar.")
        
    filename = file.filename or "data_survey.csv"
    try:
        grid, quality, depths, distances, spacing = parse_file(content, filename, spacing)
    except Exception as e:
        raise HTTPException(400, f"Error membaca data lintasan: {e}")
        
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

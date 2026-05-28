"""
Pairoq API - main.py
=====================
Production-grade FastAPI backend for geoelectrical data processing.

Architecture:
  parser/      → multi-format data ingestion
  inversion/   → regularized least-squares inversion
  interpretation/ → probabilistic zone & anomaly analysis
  report/      → structured output
  api/         → FastAPI routes (this file)

Deployment: Railway, Render, or any Python hosting
Requirements: fastapi, uvicorn, numpy, matplotlib, python-multipart
"""

from __future__ import annotations
import logging
import time
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Internal modules
from parser.base import parse_file, ArrayType
from inversion.rls import regularized_inversion
from inversion.visualize import render_section
from interpretation.analyze import interpret

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pairoq")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pairoq Geophysical API",
    description=(
        "AI-assisted geoelectrical interpretation platform. "
        "Provides preliminary resistivity inversion and probabilistic interpretation. "
        "NOT a replacement for professional geophysical software."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Pairoq Geophysical API",
        "version": "2.0.0",
        "status": "operational",
        "disclaimer": (
            "Preliminary interpretation tool only. "
            "Results require validation by a licensed geophysicist."
        ),
        "endpoints": {
            "process": "POST /process — main processing pipeline",
            "health": "GET /health",
            "docs": "GET /docs",
        }
    }


@app.get("/health", tags=["Info"])
def health():
    return {"status": "ok", "service": "Pairoq API v2.0"}


@app.post("/process", tags=["Processing"])
async def process(
    file: UploadFile = File(...),
    spacing: float = Form(5.0),
    iterations: int = Form(6),
    array_type: str = Form("wenner_alpha"),
    display_mode: str = Form("smooth"),
):
    """
    Main processing endpoint.

    Accepts a resistivity data file and returns:
    - Inverted resistivity section (base64 PNG)
    - Inversion quality metrics (RMS, convergence)
    - Probabilistic zone interpretation
    - Anomaly detection results
    - Confidence score and uncertainty statement

    Supported file formats:
    - Generic CSV (rows=depth levels, cols=electrode positions)
    - Res2Dinv DAT
    - Syscal CSV export

    Args:
        file: Survey data file
        spacing: Electrode spacing in meters (default 5.0)
        iterations: Inversion iterations (default 6, max 10)
        array_type: wenner_alpha | wenner_beta | schlumberger | dipole_dipole
        display_mode: smooth | contoured | hybrid

    Returns:
        JSON with image_b64, stats, zones, anomalies, interpretation
    """
    t0 = time.time()

    # ── Validation ──
    if spacing <= 0 or spacing > 1000:
        raise HTTPException(400, "Electrode spacing must be between 0 and 1000 meters.")
    iterations = min(max(iterations, 1), 10)

    if display_mode not in ("smooth", "contoured", "hybrid"):
        display_mode = "smooth"

    valid_arrays = [a.value for a in ArrayType]
    if array_type not in valid_arrays:
        array_type = "wenner_alpha"

    ALLOWED_EXTENSIONS = {'.csv', '.txt', '.dat', '.DAT'}
    filename = file.filename or "upload"
    ext = '.' + filename.rsplit('.', 1)[-1] if '.' in filename else ''
    if ext.lower() not in {e.lower() for e in ALLOWED_EXTENSIONS}:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # ── Step 1: Parse ──
    try:
        content = await file.read()
        dataset = parse_file(content, filename, spacing, array_type)
        logger.info(
            f"Parsed {filename}: {dataset.n_levels} levels × "
            f"{dataset.apparent_resistivity.shape[1]} cols, "
            f"completeness={dataset.data_completeness:.2f}"
        )
    except ValueError as e:
        raise HTTPException(400, f"Data parsing error: {e}")
    except Exception as e:
        logger.error(f"Unexpected parse error: {e}")
        raise HTTPException(500, f"Internal parse error: {e}")

    # ── Step 2: Inversion ──
    try:
        inv_result = regularized_inversion(
            apparent=dataset.apparent_resistivity,
            depths=dataset.depths_pseudo,
            electrode_spacing=spacing,
            iterations=iterations,
            smoothness_constraint=0.5,
            convergence_threshold=2.0,
            quality_mask=dataset.data_quality,
        )
        logger.info(
            f"Inversion complete: RMS={inv_result.rms_final:.3f}%, "
            f"converged={inv_result.converged}, "
            f"confidence={inv_result.confidence_score:.3f}"
        )
    except Exception as e:
        logger.error(f"Inversion error: {e}")
        raise HTTPException(500, f"Inversion error: {e}")

    # ── Step 3: Interpretation ──
    try:
        interp = interpret(inv_result)
    except Exception as e:
        logger.error(f"Interpretation error: {e}")
        raise HTTPException(500, f"Interpretation error: {e}")

    # ── Step 4: Render ──
    try:
        image_b64 = render_section(
            result=inv_result,
            filename=filename,
            display_mode=display_mode,
            electrode_spacing=spacing,
        )
    except Exception as e:
        logger.error(f"Render error: {e}")
        raise HTTPException(500, f"Render error: {e}")

    elapsed = round(time.time() - t0, 2)
    logger.info(f"Total processing time: {elapsed}s")

    # ── Response ──
    return JSONResponse({
        "status": "ok",
        "processing_time_s": elapsed,

        # Image
        "image_b64": image_b64,

        # Survey stats
        "stats": {
            "points_total": dataset.total_point_count,
            "points_valid": dataset.valid_point_count,
            "data_completeness": round(dataset.data_completeness, 4),
            "n_levels": dataset.n_levels,
            "n_electrodes_estimated": dataset.n_electrodes,
            "min_resistivity_ohm_m": round(float(dataset.apparent_resistivity[dataset.data_quality].min()), 2),
            "max_resistivity_ohm_m": round(float(dataset.apparent_resistivity[dataset.data_quality].max()), 2),
            "mean_resistivity_ohm_m": round(float(dataset.apparent_resistivity[dataset.data_quality].mean()), 2),
            "max_depth_m": round(dataset.max_depth_m, 2),
            "electrode_spacing_m": spacing,
            "array_type": array_type,
            "source_format": dataset.source_format.value,
        },

        # Inversion quality
        "inversion": {
            "method": inv_result.inversion_method,
            "iterations_run": inv_result.iterations_run,
            "converged": inv_result.converged,
            "rms_final_pct": inv_result.rms_final,
            "rms_per_iteration": inv_result.rms_per_iteration,
            "confidence_score": inv_result.confidence_score,
            "scientific_disclaimer": inv_result.scientific_disclaimer,
        },

        # Interpretation
        "zones": [
            {
                "label": z.label,
                "resistivity_class": z.resistivity_class,
                "geophysical_descriptor": z.geophysical_descriptor,
                "geological_context": z.geological_context,
                "range_ohm_m": list(z.range_ohm_m),
                "mean_ohm_m": z.mean_ohm_m,
                "uncertainty": z.uncertainty_mean,
                "confidence_qualifier": z.confidence_qualifier,
                "area_fraction": z.area_fraction,
                # Legacy fields for frontend compatibility
                "range": f"{z.range_ohm_m[0]:.0f}–{z.range_ohm_m[1]:.0f} Ω·m",
                "interpretation": z.geological_context,
                "color": _zone_color(z.resistivity_class),
            }
            for z in interp.zones
        ],

        "anomalies": interp.anomalies,

        "interpretation": {
            "summary": interp.summary_text,
            "uncertainty_statement": interp.uncertainty_statement,
            "recommended_followup": interp.recommended_followup,
            "disclaimer": interp.disclaimer,
        },

        # Geometry
        "depths_m": [round(float(d), 2) for d in inv_result.depths_m],
        "distances_m": [round(float(d), 2) for d in inv_result.distances_m],
    })


def _zone_color(resistivity_class: str) -> str:
    """Map resistivity class to display color for frontend."""
    colors = {
        "Very Low": "#1E40AF",
        "Low": "#0284C7",
        "Moderate": "#059669",
        "Elevated": "#D97706",
        "High": "#DC2626",
        "Very High": "#7F1D1D",
    }
    return colors.get(resistivity_class, "#6B7280")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Pairoq Backend v2.0
## AI-Assisted Geophysical Interpretation Platform

---

## Architecture

```
pairoq-v2/
├── main.py                    # FastAPI app + routes
├── requirements.txt
├── render.yaml
├── parser/
│   └── base.py               # Multi-format data ingestion
├── inversion/
│   ├── rls.py                # Regularized least-squares inversion
│   └── visualize.py          # Professional section rendering
└── interpretation/
    └── analyze.py            # Probabilistic zone & anomaly analysis
```

---

## Scientific Design Decisions

### Why NOT pyGIMLi or ResIPy?

Both are excellent libraries for real inversion. However:

| Constraint | Issue |
|---|---|
| RAM | pyGIMLi needs 2-4GB minimum. Railway free = 512MB |
| Build time | pyGIMLi requires C++ compilation. Fails on most free hosting |
| Cold start | ResIPy is GUI-first, not API-friendly |

**Upgrade path:** The inversion module (`inversion/rls.py`) is designed to be
swapped with a pyGIMLi wrapper when proper compute is available.
Replace `regularized_inversion()` with `gimli_inversion()` and the rest stays.

### What the inversion actually does

This is a **regularized least-squares approximation**, not true finite-element inversion.

It performs:
1. Depth-weighted lateral smoothing (Occam-style regularization)
2. Vertical gradient constraint (geological layering prior)
3. Iterative data fidelity correction (Gauss-Newton approximation)
4. Per-cell uncertainty estimation from residuals + depth sensitivity

What it does NOT do:
- Full forward modeling
- Jacobian/sensitivity matrix computation
- True model space exploration

**RMS values are relative to the approximation, not to field data acquisition errors.**

### Confidence Score

The confidence score (0-1) reflects:
- 40%: RMS quality (lower = better fit to apparent data)
- 20%: Convergence (did iterations stabilize?)
- 25%: Data completeness (fraction of valid measurements)
- 15%: Electrode coverage (lateral profile quality)

**A high confidence score means the inversion is internally consistent,
NOT that the geological interpretation is correct.**

### Interpretation Language

All interpretations use probabilistic language:
- "conductive anomaly" not "groundwater"
- "consistent with clay-rich sediments" not "this is clay"
- "possible saturated layer" not "aquifer confirmed"

This is deliberate. Geological conclusions require field validation.

---

## Supported Formats

| Format | Extension | Notes |
|---|---|---|
| Generic CSV | .csv, .txt | rows=depth levels, cols=electrodes |
| Res2Dinv DAT | .dat | Standard Loke format |
| Syscal CSV | .csv | Auto-detected by column headers |

## Supported Arrays

- Wenner-Alpha (default)
- Wenner-Beta
- Schlumberger
- Dipole-Dipole

---

## API Reference

### POST /process

| Parameter | Type | Default | Description |
|---|---|---|---|
| file | File | required | Survey data file |
| spacing | float | 5.0 | Electrode spacing (m) |
| iterations | int | 6 | Inversion iterations (1-10) |
| array_type | string | wenner_alpha | Array configuration |
| display_mode | string | smooth | smooth / contoured / hybrid |

### Response includes:
- `image_b64`: Base64 PNG section
- `stats`: Survey statistics
- `inversion`: RMS, convergence, confidence
- `zones`: Resistivity zone classification
- `anomalies`: Detected anomalies with uncertainty
- `interpretation`: Probabilistic summary + recommendations

---

## Deploy to Railway

```bash
# Push to GitHub, connect Railway, auto-deploys via render.yaml
railway up
```

---

## Upgrade to Real Inversion (Future)

Replace `inversion/rls.py:regularized_inversion()` with:

```python
import pygimli as pg
import pygimli.physics.ert as ert

def gimli_inversion(dataset, spacing, iterations):
    # pygimli ERT inversion
    mgr = ert.ERTManager(dataset)
    mgr.invert(lam=20, maxIter=iterations)
    return mgr.model, mgr.inv.chi2History
```

The rest of the architecture (parser, interpretation, visualization, API) remains unchanged.

# Geo-INQUIRE | EFEHR Workshop

**Messina, April 2026**

<img src="geo-inquire.png" alt="Geo-INQUIRE" height="40"> <img src="efehr.png" alt="EFEHR" height="40">

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Hands-on notebooks for querying the EFEHR/SHARE seismic hazard web service. Run them in order - each notebook builds on the config file written by the first.

---

## Repository layout

```
geoinquire-efehr-workshop/
|
|- notebooks/                        # participant notebooks (run in order)
|   |- 01_parameter_discovery.ipynb
|   |- 02_interactive_hazard_plotter.ipynb
|   `- 03_hazard_maps.ipynb
|
|- solutions/                        # instructor copies with cells filled in
|   |- 01_parameter_discovery_SOLUTIONS.ipynb
|   |- 02_interactive_hazard_plotter_SOLUTIONS.ipynb
|   `- 03_hazard_maps_SOLUTIONS.ipynb
|
|- reference/                        # original Basel notebooks (read-only)
|   `- source_model_statistics_explorer.ipynb
|
|- data/
|   `- hazard_config_messina.yaml    # pre-fetched config so NB 02/03 survive API outages
|
|- instructor/
|   `- Workshop_Guide_EFEHR_Messina2026.docx
|
|- environment.yml                   # conda environment (recommended)
|- requirements.txt                  # pip fallback
`- .github/workflows/test-notebooks.yml
```

---

## Quick start

### Option A - conda (recommended)

```bash
conda env create -f environment.yml
conda activate geoinquire-efehr
jupyter lab
```

### Option B - venv + pip (Debian/Ubuntu)

If you get an `externally-managed-environment` error, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```

To reuse the environment in future sessions:

```bash
source .venv/bin/activate
jupyter lab
```

### Option C - pip (other systems)

```bash
pip install requests pyyaml scipy matplotlib numpy
pip install cartopy folium          # optional - notebooks degrade gracefully without these
jupyter lab
```

Run notebooks **in order**: `01` then `02` then `03`. Notebook 01 writes `hazard_config.yaml`; notebooks 02 and 03 depend on it.

---

## Notebooks

### 01 - Parameter Discovery

Queries the SHARE API for a given location and establishes what is available: which hazard models cover the site, which IMTs and soil types are supported, and which return periods are defined. Saves everything to `hazard_config.yaml`.

**Key outputs:** `hazard_config.yaml`, printed parameter tables, site comparison table for five Italian cities.

### 02 - Interactive Hazard Plotter

Fetches single-site results (hazard curves and Uniform Hazard Spectra) and plots them interactively. Five exercises cover a single curve, epistemic uncertainty bands, ESHM13 vs ESHM20 model comparison, a 475-yr UHS, and a five-city comparison.

**Key outputs:** hazard curve plots, UHS plots.

### 03 - Hazard Maps

Downloads pre-computed spatial grids from the `/map` endpoint and plots them geographically. Five exercises cover a basic PGA map, three return periods on a shared colour scale, an ESHM20 - ESHM13 difference map, a SA[0.2s]/PGA spectral ratio map, and an interactive folium HTML map.

**Key outputs:** static hazard maps (cartopy or matplotlib fallback), `hazard_map_interactive.html`.

### Bonus - Source Model Statistics Explorer

Located in `reference/`, this notebook lets you browse and compare EFEHR seismic source model files directly. It is self-contained and independent of the main three notebooks.

---

## API architecture

The SHARE service at `http://appsrvr.share-eu.org:8080/share` exposes three main endpoints:

| Endpoint | Used in | Notes |
|----------|---------|-------|
| `/hazardcurve` | NB 01, 02 | Single-site hazard curves |
| `/spectra` | NB 02 | Single-site UHS |
| `/map` | NB 03 | Spatial grid - 4-level cascade (see below) |

The `/map` endpoint is a **4-level cascade**, not a direct data endpoint:

1. `GET /map?id=M&imt=I` - list of available PoEs
2. `GET /map?id=M&imt=I&hmapexceedprob=P&hmapexceedyears=Y` - list of aggregation types
3. Same URL + `&soiltype=S&aggregationtype=T&aggregationlevel=L` - map reference (`hmapid`, `hmapwms`, `hazardmaplocation`)
4. `GET <hazardmaplocation>` - actual grid data (CSV or XML)

If Level 3 returns an `hmapid` but no `hazardmaplocation`, the code probes a set of candidate data-download URLs derived from both the `hmapid` integer and the `hmapwms` layer name.

---

## Known issues

- **`/map` Level 4 data download is unresolved.** Level 3 returns `hmapid` and `hmapwms` but `hazardmaplocation` is not always present. The probe in `fetch_map()` tries 14 candidate URL patterns and prints HTTP status for each.
- **ESHM20 PoEs are annual rates** (years=1); ESHM13 uses cumulative 50-year probabilities. `resolve_poe()` handles the conversion automatically.
- **cartopy is optional.** All map exercises fall back to plain matplotlib if cartopy is not installed.
- The `hazard_config.yaml` written by NB 01 for Basel (the original notebooks default) will not work for Messina exercises - always run NB 01 first if the YAML is missing or stale. A pre-fetched Messina config is available in `data/hazard_config_messina.yaml`.

---

## Files

| File | Description |
|------|-------------|
| `data/hazard_config_messina.yaml` | Pre-fetched config for Messina (lat 38.19, lon 15.55) |
| `instructor/Workshop_Guide_EFEHR_Messina2026.docx` | Instructor guide: API architecture, session timeline, exercise goals, common errors |
| `hazard_map_interactive.html` | Generated by NB 03 Exercise E (folium) - open in a browser |

---

## Acknowledgements

This workshop is part of the **Geo-INQUIRE** project, funded by the European Commission under the HORIZON-INFRA-2021-SERV-01 call, Grant Agreement No. 101058518.

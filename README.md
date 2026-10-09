# International Training Course on Seismology, Seismic Data Analysis, Hazard Assessment and Risk Mitigation

## ESHM20 Seismic Hazard — Hands-on Notebooks

**15 - 30 June 2026 · Potsdam, Germany**

<img src="geo-inquire.png" alt="Geo-INQUIRE" height="40"> <img src="efehr.png" alt="EFEHR" height="40">

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Three hands-on notebooks for querying the EFEHR seismic hazard web services. Run them in order — each notebook builds on the config file written by the first.

---

## Repository layout

```
notebooks/
|
|- notebooks/                        # participant notebooks (run in order)
|   |- 01_parameter_discovery.ipynb
|   |- 02_interactive_hazard_plotter.ipynb
|   `- 03_hazard_maps.ipynb
|
|- environment.yml                   # conda environment (recommended)
|- requirements.txt                  # pip fallback
`- README.md
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

Queries the EFEHR REST API for a given location to find which hazard models, IMTs, soil types, aggregations, and return periods are available. Saves everything to `hazard_config.yaml`.

**Key outputs:** `hazard_config.yaml`, printed parameter tables.

### 02 - Interactive Hazard Plotter

Fetches single-site results (hazard curves and Uniform Hazard Spectra) and plots them interactively. Five exercises cover a single curve, epistemic uncertainty bands, IMT comparison, a 475-yr UHS, and a five-city comparison.

**Key outputs:** hazard curve plots, UHS plots.

### 03 - Hazard Maps

Downloads spatial hazard grids via `/v1/maps/area`, plots them geographically, and builds an interactive web map combining WMS hazard tiles and OGC API source-model overlays. Five exercises cover a basic PGA map, three return periods, an epistemic uncertainty map, a spectral ratio map, and a folium HTML map.

**Key outputs:** static hazard maps (cartopy or matplotlib fallback), `hazard_map_interactive.html`.

---

## Services used

| Service | Base URL | Used in |
|---------|----------|---------|
| **REST JSON API** | `efehr-services.ethz.ch/hazard/api` | NB 01, 02, 03 |
| **WMS** | `efehr-services.ethz.ch/hazard/ows/eshm20-output` | NB 03 Exercise E |
| **OGC API Features** | `efehr-services.ethz.ch/hazard/ows/eshm20-input/ogcapi` | NB 03 Exercise E |

Key REST endpoints used by the notebooks:

| Endpoint | Purpose |
|----------|---------|
| `GET /v1/models?lat=&lon=` | Which models cover this site |
| `GET /v1/models/{id}/imts` | Available IMTs |
| `GET /v1/curves/soiltypes?modelid=&imt=&lat=&lon=` | Available soil types |
| `GET /v1/curves/aggregations?modelid=&imt=&lat=&lon=` | Available aggregation levels |
| `GET /v1/models/{id}/poe?imt=` | Available return periods |
| `GET /v1/curves?modelid=&imt=&…` | Single-site hazard curve |
| `GET /v1/maps/area?modelid=&imt=&poe=&…&bbox=` | Spatial hazard grid |
| `GET /v1/maps/reference?modelid=&imt=&poe=&timespan=` | WMS layer name for a map |

---

## Notes

- **ESHM20 PoEs are annual rates** (years=1). The 475-yr RP corresponds to `prob=0.002103, years=1`.
- **cartopy is optional.** NB 03 map exercises fall back to plain matplotlib if cartopy is not installed.
- Always run NB 01 first when you change the site location — NB 02 and 03 read `hazard_config.yaml`.

---

## Acknowledgements

These materials were prepared for the **International Training Course on Seismology, Seismic Data Analysis, Hazard Assessment and Risk Mitigation**, 15 - 30 June 2026, Potsdam, Germany.

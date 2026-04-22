from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

GITLAB_API_BASE = "https://gitlab.seismo.ethz.ch/api/v4"
GITLAB_PROJECT_PATH = "efehr/eshm20"
DEFAULT_REPO_REF = "master"
OQ_COMPUTATIONAL_ROOT = "oq_computational"
REPO_RAW_BASE = "https://gitlab.seismo.ethz.ch/efehr/eshm20/-/raw/master"
DEFAULT_VERSION_ROOT = "oq_computational/oq_configuration_eshm20_v12e_region_main"
ASM_V12E_ROOT = f"{DEFAULT_VERSION_ROOT}/source_models/asm_v12e"
ASM_V12E_DEFAULT_FILES = [
    "asm_ver12e_winGT_fs017_hi_abgrs_maxmag_low.xml",
    "asm_ver12e_winGT_fs017_hi_abgrs_maxmag_mid.xml",
]
ASM_V12E_EXCLUDED_FILES = {"asm_ver12e_winGT_fs017_twingr.xml"}
VERSION_ROOT_RE = re.compile(r"^oq_configuration_eshm20_(?P<version>[^_]+)_region_(?P<region>.+)$")
SOURCE_KEY_RE = re.compile(r"([A-Z]{2,}\d{3,})$")
FAMILY_KIND_PREFIXES = {
    "asm": "asm",
    "deep": "deep",
    "fsm": "fsm",
    "interface": "interface",
    "ssm": "ssm",
    "volcanic": "volcanic",
}


def _join_repo_path(base_path: str, subpath: Optional[str]) -> str:
    base = str(base_path).strip().strip("/")
    if not base:
        raise ValueError("base_path cannot be empty.")
    child = str(subpath or "").strip().strip("/")
    return f"{base}/{child}" if child else base


def _repo_rel_to_asm_v12e_dir(repo_rel: str) -> bool:
    return "/source_models/asm_v12e/" in f"/{repo_rel.strip('/')}/"


def _family_kind_from_dir(family_dir: str, version_relative: str) -> str:
    if not family_dir:
        return "context" if not version_relative.startswith("source_models/") else "unknown"
    for prefix, family_kind in FAMILY_KIND_PREFIXES.items():
        if family_dir.startswith(f"{prefix}_"):
            return family_kind
    return "unknown"


def repo_rel_path(
    filename: str,
    dataset_family: str = "asm_v12e",
    version_root: str = DEFAULT_VERSION_ROOT,
) -> str:
    if dataset_family != "asm_v12e":
        raise NotImplementedError("v1 currently supports dataset_family='asm_v12e' only.")
    asm_root = _join_repo_path(version_root, "source_models/asm_v12e")
    return f"{asm_root}/{filename}"


def default_asm_v12e_selection(version_root: str = DEFAULT_VERSION_ROOT) -> List[str]:
    return [repo_rel_path(name, version_root=version_root) for name in ASM_V12E_DEFAULT_FILES]


def validate_selected_files(
    selected_files: Sequence[str],
    dataset_family: str = "asm_v12e",
    excluded_files: Optional[Sequence[str]] = None,
) -> List[str]:
    if dataset_family != "asm_v12e":
        raise NotImplementedError("v1 currently supports dataset_family='asm_v12e' only.")

    cleaned = [str(path).strip() for path in selected_files if str(path).strip()]
    if not cleaned:
        raise ValueError("selected_files is empty. Choose at least one asm_v12e XML path.")

    invalid_family = [path for path in cleaned if not _repo_rel_to_asm_v12e_dir(path)]
    if invalid_family:
        raise ValueError(
            "selected_files must stay under source_models/asm_v12e in v1:\n - " + "\n - ".join(invalid_family)
        )

    invalid_extension = [path for path in cleaned if PurePosixPath(path).suffix.lower() != ".xml"]
    if invalid_extension:
        raise ValueError("selected_files must be XML files in v1:\n - " + "\n - ".join(invalid_extension))

    invalid = [path for path in cleaned if not path.startswith("oq_computational/")]
    if invalid:
        raise ValueError("selected_files must be repository-relative paths under oq_computational/:\n - " + "\n - ".join(invalid))

    excluded = set(excluded_files or ASM_V12E_EXCLUDED_FILES)
    blocked = [path for path in cleaned if path.split("/")[-1] in excluded]
    if blocked:
        raise ValueError("v1 excludes these files:\n - " + "\n - ".join(blocked))

    return list(dict.fromkeys(cleaned))


def describe_selected_files(selected_files: Sequence[str], dataset_family: str = "asm_v12e") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": [repo_rel.split("/")[-1] for repo_rel in selected_files],
            "repo_rel": list(selected_files),
            "dataset_family": dataset_family,
            "parser_path": "asm_v12e_area_source_xml_v1",
        }
    )


def _gitlab_tree_url(project_path: str = GITLAB_PROJECT_PATH, api_base: str = GITLAB_API_BASE) -> str:
    encoded_project = quote(project_path, safe="")
    return f"{api_base}/projects/{encoded_project}/repository/tree"


def list_repository_tree(
    path: str,
    repo_ref: str = DEFAULT_REPO_REF,
    recursive: bool = False,
    per_page: int = 100,
    timeout_s: int = 60,
    project_path: str = GITLAB_PROJECT_PATH,
    api_base: str = GITLAB_API_BASE,
) -> List[Dict[str, Any]]:
    """
    List repository entries under one path using GitLab repository/tree with pagination.
    """
    if per_page <= 0:
        raise ValueError("per_page must be positive.")

    tree_url = _gitlab_tree_url(project_path=project_path, api_base=api_base)
    normalized_path = str(path).strip().strip("/")
    if not normalized_path:
        raise ValueError("path cannot be empty.")

    page = 1
    max_pages = 1000
    all_entries: List[Dict[str, Any]] = []
    while True:
        response = requests.get(
            tree_url,
            params={
                "path": normalized_path,
                "ref": repo_ref,
                "recursive": str(bool(recursive)).lower(),
                "per_page": int(per_page),
                "page": page,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected GitLab tree payload type for path='{normalized_path}'.")
        all_entries.extend(payload)

        next_page = response.headers.get("X-Next-Page", "").strip()
        if next_page:
            page = int(next_page)
            if page > max_pages:
                raise RuntimeError("GitLab tree pagination exceeded safety limit.")
            continue

        # Fallback in case pagination headers are omitted.
        if len(payload) < per_page:
            break
        page += 1
        if page > max_pages:
            raise RuntimeError("GitLab tree pagination exceeded safety limit.")

    return all_entries


def discover_version_roots(
    repo_ref: str = DEFAULT_REPO_REF,
    computational_root: str = OQ_COMPUTATIONAL_ROOT,
    timeout_s: int = 60,
) -> pd.DataFrame:
    """
    Discover candidate dataset version roots directly under oq_computational/.
    """
    entries = list_repository_tree(
        path=computational_root,
        repo_ref=repo_ref,
        recursive=False,
        timeout_s=timeout_s,
    )
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("type") != "tree":
            continue
        name = str(entry.get("name", ""))
        match = VERSION_ROOT_RE.match(name)
        if match is None:
            continue
        rows.append(
            {
                "name": name,
                "repo_rel": str(entry.get("path", "")),
                "version_tag": match.group("version"),
                "region": match.group("region"),
                "entry_type": "directory",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["version_tag", "region", "name"]).reset_index(drop=True)


def _classify_inventory_row(
    *,
    repo_rel: str,
    entry_type: str,
    version_root: str,
    scope_root: str,
    excluded_files: Sequence[str],
) -> Dict[str, Any]:
    version_root_clean = version_root.strip("/")
    scope_root_clean = scope_root.strip("/")
    file_name = PurePosixPath(repo_rel).name

    extension = PurePosixPath(file_name).suffix.lower() if entry_type == "file" else ""
    relative_path = repo_rel
    if repo_rel == scope_root_clean:
        relative_path = "."
    elif repo_rel.startswith(f"{scope_root_clean}/"):
        relative_path = repo_rel[len(scope_root_clean) + 1 :]

    version_relative = repo_rel
    if repo_rel == version_root_clean:
        version_relative = "."
    elif repo_rel.startswith(f"{version_root_clean}/"):
        version_relative = repo_rel[len(version_root_clean) + 1 :]

    family_dir = ""
    if version_relative.startswith("source_models/"):
        parts = version_relative.split("/")
        if len(parts) >= 2:
            family_dir = parts[1]
    family_kind = _family_kind_from_dir(family_dir=family_dir, version_relative=version_relative)

    excluded = set(excluded_files)
    if entry_type == "directory":
        analysis_status = "directory"
        analysis_reason = "Directory entry (browsable, not analyzable)."
    elif extension != ".xml":
        analysis_status = "unsupported_type"
        analysis_reason = "Current parser path supports XML files only."
    elif family_dir != "asm_v12e":
        analysis_status = "unsupported_family"
        analysis_reason = "Current parser path supports asm_v12e XML files only."
    elif file_name in excluded:
        analysis_status = "unsupported_variant"
        analysis_reason = "Explicitly excluded by current asm_v12e v1 contract."
    else:
        analysis_status = "supported"
        analysis_reason = "Supported by the current asm_v12e XML analysis path."

    return {
        "repo_rel": repo_rel,
        "relative_path": relative_path,
        "name": file_name,
        "entry_type": entry_type,
        "extension": extension,
        "family_dir": family_dir,
        "family_kind": family_kind,
        "analysis_status": analysis_status,
        "analysis_reason": analysis_reason,
    }


def _count_series(series: pd.Series, column_name: str) -> pd.DataFrame:
    counts = series.fillna("").replace("", "(none)").value_counts(dropna=False)
    return counts.rename_axis(column_name).reset_index(name="count")


def discover_scoped_inventory(
    version_root: str,
    discovery_subpath: str = "source_models",
    repo_ref: str = DEFAULT_REPO_REF,
    timeout_s: int = 60,
    excluded_files: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Discover + classify every entry under version_root/discovery_subpath.
    """
    scope_root = _join_repo_path(version_root, discovery_subpath)
    excluded = list(excluded_files or ASM_V12E_EXCLUDED_FILES)

    entries = list_repository_tree(
        path=scope_root,
        repo_ref=repo_ref,
        recursive=True,
        timeout_s=timeout_s,
    )

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        gitlab_type = str(entry.get("type", ""))
        entry_type = "directory" if gitlab_type == "tree" else "file"
        repo_rel = str(entry.get("path", "")).strip().strip("/")
        if not repo_rel:
            continue
        rows.append(
            _classify_inventory_row(
                repo_rel=repo_rel,
                entry_type=entry_type,
                version_root=version_root,
                scope_root=scope_root,
                excluded_files=excluded,
            )
        )

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "repo_rel",
                "relative_path",
                "name",
                "entry_type",
                "extension",
                "family_dir",
                "family_kind",
                "analysis_status",
                "analysis_reason",
            ]
        )
    else:
        inventory = inventory.sort_values(["entry_type", "relative_path", "name"]).reset_index(drop=True)

    supported_inventory = inventory[inventory["analysis_status"] == "supported"].copy().reset_index(drop=True)
    warnings: List[str] = []
    unknown_families = sorted(
        set(inventory.loc[inventory["family_kind"] == "unknown", "family_dir"].dropna().astype(str)) - {""}
    )
    if unknown_families:
        warnings.append("Unknown family_dir patterns observed: " + ", ".join(unknown_families))
    if inventory.empty:
        warnings.append("No entries found under the chosen scope.")

    summary = {
        "n_inventory_rows": int(len(inventory)),
        "n_supported_rows": int(len(supported_inventory)),
        "counts_by_entry_type": _count_series(inventory["entry_type"], "entry_type"),
        "counts_by_family_dir": _count_series(inventory["family_dir"], "family_dir"),
        "counts_by_analysis_status": _count_series(inventory["analysis_status"], "analysis_status"),
    }

    return {
        "scope": {
            "repo_ref": repo_ref,
            "version_root": version_root.strip("/"),
            "discovery_subpath": str(discovery_subpath or "").strip("/"),
            "scope_root": scope_root,
        },
        "inventory": inventory,
        "supported_inventory": supported_inventory,
        "summary": summary,
        "warnings": warnings,
    }


def choose_default_supported_files(supported_inventory: pd.DataFrame, default_count: int = 2) -> List[str]:
    if supported_inventory.empty:
        return []
    unique_repo_rels = (
        supported_inventory["repo_rel"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    )
    return unique_repo_rels[: max(0, int(default_count))]


def raw_url(repo_rel: str, repo_raw_base: str = REPO_RAW_BASE) -> str:
    return f"{repo_raw_base}/{repo_rel}"


@lru_cache(maxsize=64)
def fetch_remote_xml(repo_rel: str, repo_raw_base: str = REPO_RAW_BASE, timeout_s: int = 60) -> str:
    """Download one XML file from the ESHM20 raw endpoint."""
    url = raw_url(repo_rel=repo_rel, repo_raw_base=repo_raw_base)
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    text = response.text
    if text.lstrip().startswith("<!DOCTYPE html"):
        raise ValueError(f"Remote response for {repo_rel} was HTML, not XML. URL: {url}")
    if "<nrml" not in text:
        raise ValueError(f"Remote response for {repo_rel} does not look like NRML XML. URL: {url}")
    return text


def _lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> Optional[ET.Element]:
    for child in list(node):
        if _lname(child.tag) == name:
            return child
    return None


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    stripped = value.strip()
    return float(stripped) if stripped else None


def _parse_pos_list(text: str) -> List[Tuple[float, float]]:
    values = [float(part) for part in text.strip().split()]
    return list(zip(values[0::2], values[1::2])) if len(values) >= 4 else []


def _extract_source_key(source_id: str, source_name: str) -> str:
    for candidate in (source_id or "", source_name or ""):
        match = SOURCE_KEY_RE.search(candidate)
        if match:
            return match.group(1)
    return source_id or source_name


def _parse_mfd(area_source: ET.Element) -> Dict[str, Any]:
    trunc_gr = _child(area_source, "truncGutenbergRichterMFD")
    if trunc_gr is not None:
        return {
            "type": "truncGutenbergRichterMFD",
            "aValue": _as_float(trunc_gr.attrib.get("aValue")),
            "bValue": _as_float(trunc_gr.attrib.get("bValue")),
            "minMag": _as_float(trunc_gr.attrib.get("minMag")),
            "maxMag": _as_float(trunc_gr.attrib.get("maxMag")),
        }

    incremental = _child(area_source, "incrementalMFD")
    if incremental is not None:
        rates_node = _child(incremental, "occurRates")
        rates = [float(piece) for piece in rates_node.text.split()] if (rates_node is not None and rates_node.text) else []
        return {
            "type": "incrementalMFD",
            "minMag": _as_float(incremental.attrib.get("minMag")),
            "binWidth": _as_float(incremental.attrib.get("binWidth")),
            "occurRates": rates,
        }

    return {"type": "unknown"}


def parse_asm_v12e_area_sources(xml_text: str, repo_rel: str) -> Dict[str, Any]:
    """Parse area-source geometry + MFD fields used by the v1 notebook."""
    root = ET.fromstring(xml_text)
    area_sources = [element for element in root.iter() if _lname(element.tag) == "areaSource"]

    sources: List[Dict[str, Any]] = []
    for source in area_sources:
        source_id = source.attrib.get("id", "")
        source_name = source.attrib.get("name", "")

        area_geometry = _child(source, "areaGeometry")
        polygon: List[Tuple[float, float]] = []
        upper_depth = None
        lower_depth = None

        if area_geometry is not None:
            pos_list = next((n for n in area_geometry.iter() if _lname(n.tag) == "posList" and n.text), None)
            if pos_list is not None and pos_list.text:
                polygon = _parse_pos_list(pos_list.text)

            upper = _child(area_geometry, "upperSeismoDepth")
            lower = _child(area_geometry, "lowerSeismoDepth")
            upper_depth = _as_float(upper.text if upper is not None else None)
            lower_depth = _as_float(lower.text if lower is not None else None)

        nodal_dist = _child(source, "nodalPlaneDist")
        nodal = _child(nodal_dist, "nodalPlane") if nodal_dist is not None else None
        hypo_dist = _child(source, "hypoDepthDist")
        hypo = _child(hypo_dist, "hypoDepth") if hypo_dist is not None else None

        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]

        sources.append(
            {
                "repo_rel": repo_rel,
                "source_id": source_id,
                "source_name": source_name,
                "source_key": _extract_source_key(source_id, source_name),
                "tectonic_region": source.attrib.get("tectonicRegion", ""),
                "polygon_lonlat": polygon,
                "centroid_lon": float(np.mean(lons)) if lons else None,
                "centroid_lat": float(np.mean(lats)) if lats else None,
                "upper_seismo_depth": upper_depth,
                "lower_seismo_depth": lower_depth,
                "strike": _as_float(nodal.attrib.get("strike")) if nodal is not None else None,
                "dip": _as_float(nodal.attrib.get("dip")) if nodal is not None else None,
                "rake": _as_float(nodal.attrib.get("rake")) if nodal is not None else None,
                "hypo_depth": _as_float(hypo.attrib.get("depth")) if hypo is not None else None,
                "mfd": _parse_mfd(source),
            }
        )

    mfd_types = pd.Series([item["mfd"].get("type", "unknown") for item in sources]).value_counts().to_dict()
    return {
        "repo_rel": repo_rel,
        "n_area_sources": len(sources),
        "mfd_type_counts": mfd_types,
        "sources": sources,
    }


def load_selected_source_model_files(
    selected_files: Sequence[str],
    dataset_family: str = "asm_v12e",
    repo_raw_base: str = REPO_RAW_BASE,
    timeout_s: int = 60,
) -> Dict[str, Dict[str, Any]]:
    selected = validate_selected_files(selected_files=selected_files, dataset_family=dataset_family)
    if dataset_family != "asm_v12e":
        raise NotImplementedError("v1 currently supports dataset_family='asm_v12e' only.")

    docs: Dict[str, Dict[str, Any]] = {}
    for repo_rel in selected:
        xml_text = fetch_remote_xml(repo_rel=repo_rel, repo_raw_base=repo_raw_base, timeout_s=timeout_s)
        docs[repo_rel] = parse_asm_v12e_area_sources(xml_text=xml_text, repo_rel=repo_rel)
    return docs


def summarize_source_inventory(docs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for repo_rel, doc in docs.items():
        rows.append(
            {
                "file": repo_rel.split("/")[-1],
                "repo_rel": repo_rel,
                "n_area_sources": doc["n_area_sources"],
                "n_trunc_gr": doc["mfd_type_counts"].get("truncGutenbergRichterMFD", 0),
                "n_incremental": doc["mfd_type_counts"].get("incrementalMFD", 0),
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def closed_polygon(polygon: Sequence[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    if len(polygon) < 3:
        return None
    pts = list(polygon)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def compute_bounds_from_docs(docs: Dict[str, Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    min_lon, max_lon = np.inf, -np.inf
    min_lat, max_lat = np.inf, -np.inf

    for doc in docs.values():
        for source in doc["sources"]:
            polygon = source["polygon_lonlat"]
            if len(polygon) < 3:
                continue
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            min_lon, max_lon = min(min_lon, min(xs)), max(max_lon, max(xs))
            min_lat, max_lat = min(min_lat, min(ys)), max(max_lat, max(ys))

    if not (np.isfinite(min_lon) and np.isfinite(max_lon) and np.isfinite(min_lat) and np.isfinite(max_lat)):
        return None
    return (min_lon, max_lon, min_lat, max_lat)


def parse_polygon_text(text: str) -> List[Tuple[float, float]]:
    coords = []
    for part in text.split(";"):
        cleaned = part.strip()
        if cleaned:
            lon_str, lat_str = [piece.strip() for piece in cleaned.split(",")]
            coords.append((float(lon_str), float(lat_str)))
    return coords


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _point_in_polygon(lon: float, lat: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        if (y1 > lat) != (y2 > lat):
            x_cross = (x2 - x1) * (lat - y1) / ((y2 - y1) + 1e-12) + x1
            if lon < x_cross:
                inside = not inside
    return inside


def _point_to_segment_distance_km(
    point_lon: float,
    point_lat: float,
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(point_lat))
    x1, y1 = (lon1 - point_lon) * km_per_deg_lon, (lat1 - point_lat) * km_per_deg_lat
    x2, y2 = (lon2 - point_lon) * km_per_deg_lon, (lat2 - point_lat) * km_per_deg_lat

    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(x1, y1)

    t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg_len_sq))
    return math.hypot(x1 + t * dx, y1 + t * dy)


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


def _segments_intersect(
    p1: Tuple[float, float],
    q1: Tuple[float, float],
    p2: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    o1, o2 = _orientation(p1, q1, p2), _orientation(p1, q1, q2)
    o3, o4 = _orientation(p2, q2, p1), _orientation(p2, q2, q1)

    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True
    if abs(o1) < 1e-12 and _on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-12 and _on_segment(p1, q2, q1):
        return True
    if abs(o3) < 1e-12 and _on_segment(p2, p1, q2):
        return True
    if abs(o4) < 1e-12 and _on_segment(p2, q1, q2):
        return True
    return False


def _polygon_intersects_polygon(a: Sequence[Tuple[float, float]], b: Sequence[Tuple[float, float]]) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False

    if any(_point_in_polygon(lon=lon, lat=lat, polygon=b) for lon, lat in a):
        return True
    if any(_point_in_polygon(lon=lon, lat=lat, polygon=a) for lon, lat in b):
        return True

    for i in range(len(a)):
        a1, a2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            if _segments_intersect(a1, a2, b[j], b[(j + 1) % len(b)]):
                return True
    return False


def _polygon_intersects_circle(
    polygon: Sequence[Tuple[float, float]],
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> bool:
    if len(polygon) < 3:
        return False

    if any(_haversine_km(center_lat, center_lon, lat, lon) <= radius_km for lon, lat in polygon):
        return True
    if _point_in_polygon(lon=center_lon, lat=center_lat, polygon=polygon):
        return True

    for idx in range(len(polygon)):
        lon1, lat1 = polygon[idx]
        lon2, lat2 = polygon[(idx + 1) % len(polygon)]
        if _point_to_segment_distance_km(center_lon, center_lat, lon1, lat1, lon2, lat2) <= radius_km:
            return True

    return False


def normalize_region_selection(selection: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "region_mode": str(selection.get("region_mode", "whole")).strip().lower(),
        "circle_lat": float(selection.get("circle_lat", 45.0)),
        "circle_lon": float(selection.get("circle_lon", 12.0)),
        "circle_radius_km": float(selection.get("circle_radius_km", 300.0)),
        "polygon_text": str(selection.get("polygon_text", "")).strip(),
    }
    if normalized["region_mode"] not in {"whole", "circle", "polygon"}:
        raise ValueError("region_mode must be one of: 'whole', 'circle', 'polygon'.")
    if normalized["region_mode"] == "circle" and normalized["circle_radius_km"] <= 0.0:
        raise ValueError("circle_radius_km must be positive when region_mode='circle'.")
    if normalized["region_mode"] == "polygon" and not normalized["polygon_text"]:
        raise ValueError("polygon_text is required when region_mode='polygon'.")
    return normalized


def _source_in_region(source: Dict[str, Any], selection: Dict[str, Any]) -> bool:
    mode = selection["region_mode"]
    polygon = source["polygon_lonlat"]

    if mode == "whole":
        return True
    if mode == "circle":
        return _polygon_intersects_circle(
            polygon=polygon,
            center_lat=selection["circle_lat"],
            center_lon=selection["circle_lon"],
            radius_km=selection["circle_radius_km"],
        )

    region_poly = parse_polygon_text(selection["polygon_text"])
    if not region_poly:
        raise ValueError("polygon_text did not produce valid coordinates.")
    return _polygon_intersects_polygon(polygon, region_poly)


def filter_sources_by_region(
    docs: Dict[str, Dict[str, Any]],
    region_selection: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """Apply source-membership filtering (rates are unchanged for retained sources)."""
    filtered_docs: Dict[str, Dict[str, Any]] = {}
    for repo_rel, doc in docs.items():
        kept = [source for source in doc["sources"] if _source_in_region(source, region_selection)]
        filtered_docs[repo_rel] = {**doc, "sources": kept, "n_area_sources": len(kept)}

    summary = pd.DataFrame(
        [
            {
                "file": repo_rel.split("/")[-1],
                "repo_rel": repo_rel,
                "sources_before": docs[repo_rel]["n_area_sources"],
                "sources_after_filter": filtered_docs[repo_rel]["n_area_sources"],
                "region_mode": region_selection["region_mode"],
            }
            for repo_rel in docs.keys()
        ]
    )
    summary["sources_removed"] = summary["sources_before"] - summary["sources_after_filter"]
    summary["pct_kept"] = np.where(
        summary["sources_before"] > 0,
        100.0 * summary["sources_after_filter"] / summary["sources_before"],
        np.nan,
    )
    return filtered_docs, summary.sort_values("file").reset_index(drop=True)


def build_truncgr_source_table(docs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for repo_rel, doc in docs.items():
        for source in doc["sources"]:
            mfd = source["mfd"]
            if mfd.get("type") != "truncGutenbergRichterMFD":
                continue
            rows.append(
                {
                    "file": repo_rel.split("/")[-1],
                    "repo_rel": repo_rel,
                    "source_key": source["source_key"],
                    "aValue": mfd.get("aValue"),
                    "bValue": mfd.get("bValue"),
                    "minMag": mfd.get("minMag"),
                    "maxMag": mfd.get("maxMag"),
                }
            )
    return pd.DataFrame(rows)


def _trunc_gr_cumulative_rate(a_value: float, b_value: float, mag: float, max_mag: float) -> float:
    if mag >= max_mag:
        return 0.0
    return max(0.0, 10.0 ** (a_value - b_value * mag) - 10.0 ** (a_value - b_value * max_mag))


def _source_incremental_rates(
    a_value: float,
    b_value: float,
    min_mag: float,
    max_mag: float,
    mag_bin_starts: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    rates = np.zeros_like(mag_bin_starts, dtype=float)
    if any(value is None for value in [a_value, b_value, min_mag, max_mag]) or max_mag <= min_mag:
        return rates

    for idx, mag in enumerate(mag_bin_starts):
        if mag < min_mag or mag >= max_mag:
            continue
        mag_next = min(mag + bin_width, max_mag)
        n_mag = _trunc_gr_cumulative_rate(a_value, b_value, mag, max_mag)
        n_next = _trunc_gr_cumulative_rate(a_value, b_value, mag_next, max_mag)
        rates[idx] = max(0.0, n_mag - n_next)

    return rates


def build_truncgr_mfd(
    docs: Dict[str, Dict[str, Any]],
    bin_width: float = 0.1,
    mag_edges: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build per-file incremental/cumulative bundles from truncated-GR source parameters."""
    table = build_truncgr_source_table(docs)
    if table.empty:
        return {
            "trunc_table": table,
            "bin_width": bin_width,
            "mag_edges": np.array([]),
            "mag_bin_starts": np.array([]),
            "per_file": {},
        }

    if mag_edges is None:
        min_mag = float(np.floor(table["minMag"].min() / bin_width) * bin_width)
        max_mag = float(np.ceil(table["maxMag"].max() / bin_width) * bin_width)
        mag_edges = np.arange(min_mag, max_mag + bin_width, bin_width)

    mag_bin_starts = mag_edges[:-1]
    per_file: Dict[str, Dict[str, Any]] = {
        repo_rel: {
            "incremental": np.zeros_like(mag_bin_starts, dtype=float),
            "cumulative": np.zeros_like(mag_bin_starts, dtype=float),
            "n_sources": 0,
        }
        for repo_rel in docs.keys()
    }

    for repo_rel, group in table.groupby("repo_rel"):
        inc = np.zeros_like(mag_bin_starts, dtype=float)
        for _, row in group.iterrows():
            inc += _source_incremental_rates(
                row["aValue"],
                row["bValue"],
                row["minMag"],
                row["maxMag"],
                mag_bin_starts,
                bin_width,
            )
        per_file[repo_rel]["incremental"] = inc
        per_file[repo_rel]["cumulative"] = np.cumsum(inc[::-1])[::-1]
        per_file[repo_rel]["n_sources"] = len(group)

    return {
        "trunc_table": table,
        "bin_width": bin_width,
        "mag_edges": mag_edges,
        "mag_bin_starts": mag_bin_starts,
        "per_file": per_file,
    }


def compare_matched_sources(docs: Dict[str, Dict[str, Any]], file_a: str, file_b: str) -> pd.DataFrame:
    if file_a not in docs or file_b not in docs:
        raise KeyError("Both file_a and file_b must be present in docs.")

    table_a = build_truncgr_source_table({file_a: docs[file_a]})
    table_b = build_truncgr_source_table({file_b: docs[file_b]})
    if table_a.empty or table_b.empty:
        return pd.DataFrame()

    left = table_a[["source_key", "aValue", "bValue", "maxMag"]].rename(
        columns={"aValue": "aValue_a", "bValue": "bValue_a", "maxMag": "maxMag_a"}
    )
    right = table_b[["source_key", "aValue", "bValue", "maxMag"]].rename(
        columns={"aValue": "aValue_b", "bValue": "bValue_b", "maxMag": "maxMag_b"}
    )
    cmp = left.merge(right, on="source_key", how="inner")
    cmp["delta_maxMag"] = cmp["maxMag_b"] - cmp["maxMag_a"]
    cmp["delta_bValue"] = cmp["bValue_b"] - cmp["bValue_a"]
    return cmp


__all__ = [
    "ASM_V12E_ROOT",
    "DEFAULT_REPO_REF",
    "DEFAULT_VERSION_ROOT",
    "GITLAB_API_BASE",
    "GITLAB_PROJECT_PATH",
    "OQ_COMPUTATIONAL_ROOT",
    "REPO_RAW_BASE",
    "build_truncgr_mfd",
    "build_truncgr_source_table",
    "choose_default_supported_files",
    "closed_polygon",
    "compare_matched_sources",
    "compute_bounds_from_docs",
    "discover_scoped_inventory",
    "discover_version_roots",
    "default_asm_v12e_selection",
    "describe_selected_files",
    "filter_sources_by_region",
    "list_repository_tree",
    "load_selected_source_model_files",
    "normalize_region_selection",
    "parse_polygon_text",
    "repo_rel_path",
    "summarize_source_inventory",
    "validate_selected_files",
]

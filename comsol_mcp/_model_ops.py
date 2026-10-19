#!/usr/bin/env python3
"""Pure model operation helpers: parameters, expressions, metrics, tree, geometry."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def _parameter_rows(model: Any) -> list[dict[str, str]]:
    names = list(model.java.param().varnames())
    rows = []
    for name in names:
        key = str(name)
        rows.append(
            {
                "name": key,
                "expression": str(model.java.param().get(key)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Expression evaluation helpers
# ---------------------------------------------------------------------------
def _coerce_eval_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_eval_value(item) for item in value]
    return str(value)


def _last_scalar(value: Any) -> Any:
    current = value
    while isinstance(current, list) and current:
        current = current[-1]
    return current


def _evaluate_named_expressions(
    model: Any, items: list[dict[str, Any]], max_result_size: int = 0
) -> list[dict[str, Any]]:
    """Evaluate expression items of the form used by evaluate_expressions."""
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each expression entry must be an object.")
        name = str(item.get("name", "")).strip()
        expression = str(item.get("expression", "")).strip()
        aggregate = str(item.get("aggregate", "none")).strip().lower()
        domains = item.get("domains")
        boundaries = item.get("boundaries")
        time_point = str(item.get("time_point", "last")).strip()
        geometry = str(item.get("geometry", "") or "").strip()
        if not name:
            raise ValueError("Expression name is required.")
        if not expression:
            raise ValueError(f'Expression is required for "{name}".')
        row: dict[str, Any] = {"name": name, "expression": expression}
        int_domains = [int(d) for d in domains] if domains else None
        int_bounds = [int(b) for b in boundaries] if boundaries else None
        try:
            if aggregate and aggregate != "none":
                value = _evaluate_aggregate(
                    model, expression, aggregate, int_domains, int_bounds, time_point, geometry
                )
                row["value"] = value
                row["last_value"] = value
                row["aggregate"] = aggregate
            else:
                raw = model.evaluate(expression)
                value = _coerce_eval_value(raw)
                row["value"] = value
                row["last_value"] = _coerce_eval_value(_last_scalar(value))

            # Truncation
            if max_result_size > 0:
                serialized = json.dumps(row["value"], default=str)
                if len(serialized) > max_result_size:
                    row["value"] = serialized[:max_result_size] + "...(truncated)"
                    row["_truncated"] = True
                    row["original_size"] = len(serialized)
            row["ok"] = True
        except Exception as exc:
            row["ok"] = False
            row["error"] = str(exc)
        results.append(row)
    return results


def _parse_expression_items(raw_json: str, field: str) -> list[dict[str, Any]]:
    parsed = json.loads(raw_json)
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON array.")
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each expression entry must be an object.")
        if not str(item.get("name", "")).strip():
            raise ValueError("Expression name is required.")
        if not str(item.get("expression", "")).strip():
            raise ValueError(f'Expression is required for "{item.get("name")}".')
    return parsed


@contextmanager
def _temporary_numerical(model: Any, feature_type: str) -> Iterator[Any]:
    """Create a uniquely tagged Derived Values feature and remove only it afterwards."""
    numerical = model.java.result().numerical()
    tag = str(numerical.uniquetag("mcpeval"))
    numerical.create(tag, feature_type)
    try:
        yield model.java.result().numerical(tag)
    finally:
        try:
            numerical.remove(tag)
        except Exception:
            logging.debug("Failed to remove temporary numerical feature %s", tag, exc_info=True)


def _default_geometry(model: Any) -> tuple[str, int]:
    """Return (tag, spatial dimension) of the first geometry in the first component that has one."""
    try:
        for comp_tag in list(model.java.component().tags()):
            comp = model.java.component(str(comp_tag))
            geom_tags = list(comp.geom().tags())
            if geom_tags:
                tag = str(geom_tags[0])
                try:
                    sdim = int(comp.geom(tag).getSDim())
                except Exception:
                    sdim = 0
                return tag, sdim
    except Exception:
        pass
    return "", 0


def _get_model_dimension(model: Any) -> int:
    """Get the configured or auto-detected spatial dimension of the model.

    Priority:
    1. Workflow state `model_dimension` (set by configure_single_main_workflow)
    2. Spatial dimension of the model's first geometry
    3. Auto-detect from existing physics interfaces
    4. Default to 0 (unknown)
    """
    # Check workflow state first
    try:
        from comsol_mcp._state import _read_workflow_state
        state = _read_workflow_state()
        dim = int(state.get("model_dimension", 0))
        if dim in (1, 2, 3):
            return dim
    except Exception:
        pass

    _, sdim = _default_geometry(model)
    if sdim in (1, 2, 3):
        return sdim

    # Auto-detect from existing physics interfaces
    try:
        comps = list(model.java.component().tags())
        if comps:
            comp = model.java.component(comps[0])
            phys_tags = list(comp.physics().tags())
            if phys_tags:
                phys = comp.physics(phys_tags[0])
                # Try common API methods for getting spatial dimension from physics
                for method_name in ("getNDim", "getSpatialDim", "sdim", "getDim"):
                    try:
                        val = getattr(phys, method_name)
                        if callable(val):
                            d = int(val())
                            if d in (1, 2, 3):
                                return d
                        else:
                            d = int(val)
                            if d in (1, 2, 3):
                                return d
                    except Exception:
                        continue
                # Try reading the physics selection dimension
                try:
                    sel = phys.feature().tags()
                    if sel:
                        feat = phys.feature(sel[0])
                        # The selection geom dimension reveals spatial dim
                        for attr_name in ("_geomDim", "dim"):
                            try:
                                d = int(getattr(feat.selection(), attr_name))
                                if d in (1, 2, 3):
                                    return d
                            except Exception:
                                continue
                except Exception:
                    pass
    except Exception:
        pass

    return 0


def _evaluate_aggregate(
    model: Any,
    expression: str,
    aggregate: str,
    domains: list[int] | None = None,
    boundaries: list[int] | None = None,
    time_point: str = "last",
    geometry: str = "",
) -> Any:
    """Evaluate an expression with aggregation. Returns scalar for aggregated, raw for 'none'.

    Selections use `geometry`, or the model's first geometry when it is empty.

    Automatically selects correct COMSOL result feature type based on model dimension:
    - 3D: domains→Volume, boundaries→Surface
    - 2D: domains→Surface, boundaries→Line
    - 1D: domains→Line, boundaries→Point
    """
    aggregate = (aggregate or "none").lower().strip()
    if aggregate == "none":
        raw = model.evaluate(expression)
        return _coerce_eval_value(raw)

    # Get spatial dimension from workflow config or auto-detect
    dim = _get_model_dimension(model)
    if dim == 0:
        dim = 2  # Default to 2D if unknown (common for structural mechanics)

    # Map: (dim, entity) → suffix for max/min/int
    # 3D: domains=Volume, boundaries=Surface
    # 2D: domains=Surface, boundaries=Line
    # 1D: domains=Line, boundaries=Point (rare)
    if boundaries:
        geom_dim = max(dim - 1, 1)  # boundary is (dim-1) entity
    elif domains:
        geom_dim = dim
    else:
        geom_dim = dim

    dim_suffix = {3: "Volume", 2: "Surface", 1: "Line"}.get(geom_dim, "Surface")

    # Choose correct COMSOL result feature type
    if aggregate == "max":
        ftype = f"Max{dim_suffix}"
    elif aggregate == "min":
        ftype = f"Min{dim_suffix}"
    elif aggregate == "avg":
        ftype = f"Int{dim_suffix}" if (domains or boundaries) else "EvalGlobal"
    elif aggregate == "integral":
        ftype = f"Int{dim_suffix}"
    else:
        raise ValueError(f'Unknown aggregate "{aggregate}". Use max, min, avg, integral, or none.')

    geom_tag = geometry or _default_geometry(model)[0] or "geom1"
    with _temporary_numerical(model, ftype) as feature:
        # Set selection with correct geometric dimension
        if domains:
            feature.selection().geom(geom_tag, dim)
            feature.selection().set(domains)
        elif boundaries:
            feature.selection().geom(geom_tag, max(dim - 1, 1))
            feature.selection().set(boundaries)

        # Set expression(s) based on aggregate type
        if aggregate == "avg" and (domains or boundaries):
            feature.set("expr", [expression, "1"])
        else:
            feature.set("expr", [expression])

        if time_point and time_point != "all":
            feature.setIndex("looplevelinput", time_point, 0)

        feature.run()
        values = feature.getReal()

    if aggregate == "avg" and (domains or boundaries):
        denom = values[1][0]
        if denom == 0:
            return float("nan")
        return float(values[0][0] / denom)
    return float(values[0][0])


# ---------------------------------------------------------------------------
# Model tree
# ---------------------------------------------------------------------------
def _model_tree_data(model: Any) -> dict[str, Any]:
    from comsol_mcp._state import _safe_model_label, _safe_model_path
    java = model.java
    components = [str(tag) for tag in java.component().tags()]
    component_details = []
    for component in components:
        comp = java.component(component)
        component_details.append(
            {
                "tag": component,
                "geometries": [str(tag) for tag in comp.geom().tags()],
                "meshes": [str(tag) for tag in comp.mesh().tags()],
                "physics": [str(tag) for tag in comp.physics().tags()],
                "materials": [str(tag) for tag in comp.material().tags()],
            }
        )
    return {
        "label": _safe_model_label(model),
        "file_path": _safe_model_path(model),
        "components": components,
        "component_details": component_details,
        "parameters": [row["name"] for row in _parameter_rows(model)],
        "studies": [str(tag) for tag in java.study().tags()],
        "solutions": [str(tag) for tag in java.sol().tags()],
        "datasets": [str(tag) for tag in java.result().dataset().tags()],
        "results": [str(tag) for tag in java.result().tags()],
    }


# ---------------------------------------------------------------------------
# Geometry feature helpers
# ---------------------------------------------------------------------------
def _normalize_properties(properties_json: str) -> list[tuple[str, list[str]]]:
    raw = str(properties_json or "").strip()
    if not raw:
        return []
    parsed = json.loads(raw)
    items: list[dict[str, Any]]
    if isinstance(parsed, dict):
        items = []
        for name, value in parsed.items():
            if isinstance(value, list):
                items.append({"name": name, "values": [str(item) for item in value]})
            else:
                items.append({"name": name, "value": str(value)})
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("properties_json must be a JSON object or array.")

    normalized: list[tuple[str, list[str]]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each properties_json entry must be an object.")
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if "values" in item:
            values = item.get("values")
            if not isinstance(values, list):
                raise ValueError(f'Property "{name}" values must be an array.')
            normalized.append((name, [str(value) for value in values]))
        elif "value" in item:
            normalized.append((name, [str(item.get("value", ""))]))
        else:
            raise ValueError(f'Property "{name}" must include value or values.')
    return normalized


def _ensure_component_java(model: Any, component: str, dimension: int) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    return {"component": component, "dimension": dimension}


def _ensure_geometry_java(model: Any, component: str, geometry: str, dimension: int) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    if dimension <= 0:
        dimension = 2
    if geometry not in list(java.component(component).geom().tags()):
        java.component(component).geom().create(geometry, dimension)
    return {"component": component, "geometry": geometry, "dimension": dimension}


def _ensure_mesh_java(model: Any, component: str, mesh: str) -> dict[str, Any]:
    java = model.java
    if component not in list(java.component().tags()):
        java.component().create(component, True)
    if mesh not in list(java.component(component).mesh().tags()):
        java.component(component).mesh().create(mesh)
    return {"component": component, "mesh": mesh}


def _apply_feature_properties(feature: Any, properties: list[tuple[str, list[str]]]) -> list[dict[str, Any]]:
    applied = []
    for name, values in properties:
        if len(values) <= 1:
            feature.set(name, values[0] if values else "")
            applied.append({"name": name, "value": values[0] if values else ""})
        else:
            feature.set(name, values)
            applied.append({"name": name, "values": values})
    return applied

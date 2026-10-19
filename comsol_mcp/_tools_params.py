#!/usr/bin/env python3
"""MCP tools: get_parameters, set_parameters, evaluate_expressions, get_core_metrics."""

from __future__ import annotations

import json
from typing import Any

from comsol_mcp._state import (
    _run_tool, _run_tool_readonly, _read_workflow_state, _safe_model_label,
)
from comsol_mcp._model import _require_visible_main
from comsol_mcp._model_ops import (
    _parameter_rows, _evaluate_named_expressions, _parse_expression_items,
)


def get_parameters() -> str:
    """Return current global parameters from the selected server-side model."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_parameters")
        rows = _parameter_rows(model)
        return {"label": _safe_model_label(model), "parameters": rows, "count": len(rows)}

    return _run_tool_readonly("get_parameters", _impl)


def evaluate_expressions(expressions_json: str = "[]", max_result_size: int = 0) -> str:
    """Evaluate one or more expressions on the current server-side model.

    Each expression item supports: {"name": "...", "expression": "...",
      "aggregate": "max"|"min"|"avg"|"integral"|"none",
      "domains": [1,2], "boundaries": [5,6], "time_point": "last"|"all"|"N",
      "geometry": "geom1"}. geometry defaults to the model's first geometry.

    Backward compatible: [{"name": "...", "expression": "..."}] still works.
    max_result_size: when > 0, truncate arrays exceeding this size (chars).
    """

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("evaluate_expressions")
        items = _parse_expression_items(expressions_json, "expressions_json")
        results = _evaluate_named_expressions(model, items, max_result_size)
        return {
            "label": _safe_model_label(model),
            "results": results,
            "count": len(results),
        }

    return _run_tool_readonly("evaluate_expressions", _impl)


def get_core_metrics() -> str:
    """Evaluate the core metrics configured with configure_single_main_workflow(core_metrics_json=...).

    Each metric uses the same item format as evaluate_expressions.
    """

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("get_core_metrics")
        metrics = _read_workflow_state().get("core_metrics") or []
        if not metrics:
            return {
                "label": _safe_model_label(model),
                "solve_status": "not_configured",
                "results": [],
                "message": "No core metrics configured. Set them with configure_single_main_workflow(core_metrics_json=...).",
            }
        results = _evaluate_named_expressions(model, metrics)
        return {
            "label": _safe_model_label(model),
            "solve_status": "success" if all(row.get("ok") for row in results) else "partial",
            "results": results,
        }

    return _run_tool_readonly("get_core_metrics", _impl)


def set_parameters(parameters_json: str) -> str:
    """Set multiple global parameters on the selected server-side model."""

    def _impl() -> dict[str, Any]:
        model = _require_visible_main("set_parameters")
        parsed = json.loads(parameters_json)
        if not isinstance(parsed, list):
            raise ValueError("parameters_json must be a JSON array.")
        updated = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Each parameter entry must be an object.")
            name = str(item.get("name", "")).strip()
            expression = str(item.get("expression", "")).strip()
            if not name:
                raise ValueError("Parameter name is required.")
            model.java.param().set(name, expression)
            updated.append({"name": name, "expression": expression})
        return {"updated": updated, "count": len(updated)}

    return _run_tool("set_parameters", _impl)


def register(mcp_instance) -> None:
    mcp_instance.add_tool(get_parameters)
    mcp_instance.add_tool(evaluate_expressions)
    mcp_instance.add_tool(get_core_metrics)
    mcp_instance.add_tool(set_parameters)

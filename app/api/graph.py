"""Graph inspection endpoints (Phase 6).

Phase 6 exposes two helpers that are cheap wins for interviews:

* ``GET /api/graph/structure`` — the static node / edge map of the graph.
* ``GET /api/graph/trace?conversation_id=...`` — replay the most recent run
  for a thread and return its node order.

Phase 9 SSE will stream the trace live; this is the static snapshot.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.agent.graph import get_compiled_graph, node_trace_from_state

router = APIRouter()


# ---------- /structure ----------------------------------------------------


@router.get("/graph/structure")
async def structure() -> dict[str, Any]:
    """Return the nodes + edges of the compiled LangGraph.

    Useful for ``GET /api/graph/structure`` demo: the frontend can render
    the agent pipeline visually.
    """
    app = get_compiled_graph()
    try:
        # langgraph exposes a graph representation on the compiled object.
        graph_obj = app.get_graph()
        nodes = [
            {"id": nid, "type": "node"}
            for nid in graph_obj.nodes.keys()
        ]
        edges = [
            {"source": e.source, "target": e.target}
            for e in graph_obj.edges
        ]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"cannot introspect graph: {type(e).__name__}: {e}",
        ) from e

    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


# ---------- /trace --------------------------------------------------------


@router.get("/graph/trace")
async def trace(conversation_id: str = Query("default")) -> dict[str, Any]:
    """Return the most recent run's node path for a conversation thread."""
    app = get_compiled_graph()
    try:
        cfg = {"configurable": {"thread_id": conversation_id}}
        state = app.get_state(cfg)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"cannot read state: {type(e).__name__}: {e}",
        ) from e

    values = getattr(state, "values", {}) or {}
    path = node_trace_from_state(values)
    return {
        "conversation_id": conversation_id,
        "node_path": path,
        "node_count": len(path),
        "intent": values.get("intent"),
        "has_sql": bool(values.get("generated_sql")),
        "row_count": values.get("row_count"),
        "error": values.get("error"),
    }
"""
Tools the agent can call. The docstring and type hints ARE the tool description the LLM reads,
so write them as carefully as a prompt.

Security note: the SQL tool does NOT let the LLM write SQL. It exposes a fixed, parameterised query
over an allow-list of metrics. Never pass LLM output straight into a database.
"""
import sqlite3
from datetime import date

from langchain_core.tools import tool

from app import rag
from app.config import get_settings

ALLOWED_METRICS = {
    "biogas_nm3": "Biogas produced (Nm3/day)",
    "ch4_nm3": "Methane produced (Nm3/day)",
    "ch4_pct_prom": "Average CH4 concentration (%)",
    "electricidad_potencial_kwh": "Potential electricity with CHP (kWh/day)",
    "temp_prom_c": "Average digester temperature (C)",
    "temp_min_c": "Minimum digester temperature (C)",
    "ph_prom": "Average pH",
    "h2s_max_ppm": "Max H2S (ppm)",
    "horas_temp_fuera_rango": "Hours with temperature out of range",
    "disponibilidad_datos_pct": "Data availability (%)",
    "n_alarmas": "Number of alarms",
    "estado_operacion": "Daily status (NORMAL / ALERTA / CRITICO)",
}


@tool
def query_kpis(start_date: str, end_date: str, metrics: list[str]) -> str:
    """Get daily plant KPIs between two dates (inclusive, format YYYY-MM-DD).
    Available data: 2026-10-01 to 2026-10-30.
    metrics: any of biogas_nm3, ch4_nm3, ch4_pct_prom, electricidad_potencial_kwh, temp_prom_c, temp_min_c,
    ph_prom, h2s_max_ppm, horas_temp_fuera_rango, disponibilidad_datos_pct, n_alarmas, estado_operacion.
    Returns a Markdown table plus totals/averages."""
    try:
        d0, d1 = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return "Error: dates must be YYYY-MM-DD."
    if d0 > d1:
        return "Error: start_date must be <= end_date."
    bad = [m for m in metrics if m not in ALLOWED_METRICS]
    if bad or not metrics:
        return f"Error: unknown metrics {bad}. Allowed: {list(ALLOWED_METRICS)}"

    cols = ", ".join(["fecha", *metrics])  # safe: every name was checked against the allow-list
    with sqlite3.connect(get_settings().warehouse_db) as con:
        rows = con.execute(f"SELECT {cols} FROM gold_kpi_diario WHERE fecha BETWEEN ? AND ? ORDER BY fecha",
                           (d0.isoformat(), d1.isoformat())).fetchall()
    if not rows:
        return "No data for that period (available: 2026-10-01 to 2026-10-30)."
    header = "| date | " + " | ".join(metrics) + " |\n|" + "---|" * (len(metrics) + 1)
    lines = ["| " + " | ".join(str(v) for v in r) + " |" for r in rows[:31]]
    summary = []
    for i, m in enumerate(metrics, start=1):
        vals = [r[i] for r in rows if isinstance(r[i], (int, float))]
        if vals:
            summary.append(f"{m}: total={sum(vals):,.2f}, avg={sum(vals) / len(vals):,.2f}, "
                           f"min={min(vals):,.2f}, max={max(vals):,.2f}")
    return "\n".join([header, *lines, "", "Summary:", *summary])


@tool
def search_manuals(query: str) -> str:
    """Search plant operating procedures (SOPs), alarm responses and troubleshooting guides.
    Use it for 'how to', 'what to do if', safety and procedure questions. Query in English works best."""
    hits = rag.search(query, k=3)
    if not hits:
        return "No relevant procedure found."
    return "\n\n---\n\n".join(f"[{d.metadata['source']}] (similarity {score})\n{d.page_content}" for d, score in hits)


ALL_TOOLS = [query_kpis, search_manuals]

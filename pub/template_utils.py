from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .constants import LATEX_TABLE_STYLE


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=1)
def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )


def render_template(template_name: str, /, **context: object) -> str:
    return _template_environment().get_template(template_name).render(**context)


def _normalize_column_spec(column_spec: str, style: str) -> str:
    if style == "booktabs":
        return column_spec.replace("|", "")
    return column_spec


def render_latex_table(
    *,
    column_spec: str,
    header_rows: Sequence[str],
    body_rows: Sequence[str],
    footer_rows: Sequence[str] = (),
    opening_lines: Sequence[str] = (),
    closing_lines: Sequence[str] = (),
    style: str | None = None,
) -> str:
    resolved_style = LATEX_TABLE_STYLE if style is None else style
    resolved_column_spec = _normalize_column_spec(column_spec, resolved_style)
    top_rule = r"\toprule" if resolved_style == "booktabs" else r"\hline"
    mid_rule = r"\midrule" if resolved_style == "booktabs" else r"\hline"
    bottom_rule = r"\bottomrule" if resolved_style == "booktabs" else r"\hline"

    lines = [*opening_lines, rf"\begin{{tabular}}{{{resolved_column_spec}}}", top_rule]
    lines.extend(header_rows)
    if body_rows:
        lines.append(mid_rule)
    lines.extend(body_rows)
    if footer_rows:
        lines.append(mid_rule)
        lines.extend(footer_rows)
    lines.extend((bottom_rule, r"\end{tabular}", *closing_lines))
    return "\n".join(lines) + "\n"

__all__ = ["render_template", "render_latex_table"]

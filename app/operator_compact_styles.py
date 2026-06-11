"""Compact Streamlit CSS for the MedAI operator workflow."""
from __future__ import annotations


COMPACT_OPERATOR_CSS = """
<style>
section.main > div.block-container {
    padding-top: .65rem;
    padding-bottom: 1rem;
}
div[data-testid="stVerticalBlock"] {
    gap: .45rem;
}
div[data-testid="stHorizontalBlock"] {
    gap: .6rem;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d8dee5;
    border-radius: 8px;
    padding: .35rem .55rem;
}
div[data-testid="stMetric"] label {
    font-size: .72rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.18rem;
}
div[data-testid="stCaptionContainer"] {
    margin-top: -.15rem;
}
.medai-card {
    padding: .65rem .75rem;
    margin-bottom: .45rem;
}
.medai-header {
    padding: .62rem .78rem;
    margin-bottom: .45rem;
}
.compact-session-header {
    display: flex;
    justify-content: space-between;
    gap: .75rem;
    align-items: center;
    flex-wrap: wrap;
}
.compact-session-header h2 {
    margin: 0;
    font-size: 1.2rem;
    line-height: 1.15;
}
.compact-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    margin-top: .35rem;
}
.compact-chip {
    display: inline-flex;
    align-items: center;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #334155;
    border-radius: 999px;
    padding: .18rem .52rem;
    font-size: .78rem;
    font-weight: 650;
}
.warning-banner {
    padding: .45rem .65rem;
    margin: .35rem 0 .5rem 0;
    font-size: .9rem;
}
.compact-workflow-row {
    border: 1px solid #d8dee5;
    border-radius: 8px;
    background: #ffffff;
    padding: .58rem .65rem;
    margin-bottom: .45rem;
}
.compact-summary {
    display: flex;
    flex-wrap: wrap;
    gap: .35rem;
    margin: .25rem 0 .35rem 0;
}
.compact-table-note {
    color: #64748b;
    font-size: .8rem;
}
</style>
"""


__all__ = ["COMPACT_OPERATOR_CSS"]

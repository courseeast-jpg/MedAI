"""Compact Streamlit CSS for the MedAI operator workflow."""
from __future__ import annotations


COMPACT_OPERATOR_CSS = """
<style>
section.main > div.block-container {
    padding-top: .35rem;
    padding-bottom: .75rem;
}
div[data-testid="stVerticalBlock"] {
    gap: .28rem;
}
div[data-testid="stHorizontalBlock"] {
    gap: .42rem;
}
div[data-testid="stTabs"] [role="tablist"] {
    margin-bottom: .25rem;
}
div[data-testid="stTabs"] [role="tab"] {
    padding-top: .35rem;
    padding-bottom: .35rem;
}
div[data-testid="stFileUploader"] section {
    padding: .35rem .5rem;
    min-height: 2.6rem;
}
div[data-testid="stFileUploader"] small {
    display: none;
}
div[data-baseweb="select"] {
    margin-bottom: 0;
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
    padding: .45rem .6rem;
    margin-bottom: .3rem;
}
.medai-header {
    padding: .45rem .6rem;
    margin-bottom: .3rem;
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
    font-size: 1.05rem;
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
    padding: .32rem .5rem;
    margin: .25rem 0 .35rem 0;
    font-size: .82rem;
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
    margin: .15rem 0 .25rem 0;
}
.compact-table-note {
    color: #64748b;
    font-size: .8rem;
}
</style>
"""


__all__ = ["COMPACT_OPERATOR_CSS"]

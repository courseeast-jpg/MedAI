"""Compact Streamlit CSS for the MedAI operator workflow."""
from __future__ import annotations


COMPACT_OPERATOR_CSS = """
<style>
section.main > div.block-container {
    padding-top: .35rem;
    padding-bottom: .75rem;
    max-width: 1180px;
}
div[data-testid="stVerticalBlock"] {
    gap: .18rem;
}
div[data-testid="stHorizontalBlock"] {
    gap: .22rem;
}
div[data-testid="stTabs"] [role="tablist"] {
    margin-bottom: .25rem;
}
div[data-testid="stTabs"] [role="tab"] {
    padding-top: .35rem;
    padding-bottom: .35rem;
}
div[data-testid="stFileUploader"] section {
    padding: .38rem .55rem;
    min-height: 3rem;
    border: 1px solid #94a3b8;
    background: #f8fafc;
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
    padding: .25rem .45rem;
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
    color: #475569;
    border-radius: 999px;
    padding: .12rem .42rem;
    font-size: .72rem;
    font-weight: 600;
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
    padding: .35rem .48rem;
    margin-bottom: .2rem;
}
.operator-path {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .35rem;
    margin: .2rem 0 .35rem 0;
}
.operator-path-step {
    border: 1px solid #d8dee5;
    border-radius: 8px;
    background: #ffffff;
    padding: .45rem .55rem;
    min-height: 3rem;
}
.operator-path-step span {
    display: block;
    color: #64748b;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: 0;
    font-weight: 700;
}
.operator-path-step strong {
    display: block;
    color: #0f172a;
    font-size: .9rem;
    line-height: 1.15;
    margin-top: .12rem;
}
.operator-run-setup,
.operator-files-queue,
.operator-start-rail,
.operator-results-review {
    border: 1px solid #d8dee5;
    border-radius: 8px;
    background: #ffffff;
    padding: .55rem .65rem;
    margin-bottom: .28rem;
}
.operator-start-rail {
    border-color: #93c5fd;
    background: #eff6ff;
}
.operator-section-title {
    color: #0f172a;
    font-size: .92rem;
    font-weight: 750;
    margin-bottom: .2rem;
}
.operator-add-queue-callout {
    border: 1px solid #f59e0b;
    background: #fffbeb;
    color: #78350f;
    border-radius: 8px;
    padding: .45rem .55rem;
    margin: .2rem 0 .3rem 0;
    font-weight: 700;
}
.operator-review-action-row {
    border: 1px solid #d8dee5;
    border-radius: 8px;
    background: #f8fafc;
    padding: .45rem .55rem;
    margin: .2rem 0 .35rem 0;
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
div[data-testid="stDataFrame"] {
    margin-top: .15rem;
}
button[kind="secondary"], button[kind="primary"] {
    min-height: 2.25rem;
}
button[kind="primary"] {
    min-height: 3rem;
    font-size: 1.02rem;
    font-weight: 750;
    border-radius: 8px;
}
button[kind="secondary"] {
    min-height: 2.55rem;
    font-weight: 650;
    border-radius: 8px;
}
button:disabled {
    opacity: .55;
}
@media (max-width: 820px) {
    .operator-path {
        grid-template-columns: 1fr 1fr;
    }
}
</style>
"""


__all__ = ["COMPACT_OPERATOR_CSS"]

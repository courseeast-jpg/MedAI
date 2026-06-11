# MEDAI-ACTUAL-STREAMLIT-UI-WIRING-FIX-11F Audit

- The visible Run & Review page is rendered by app.main.render_run_review_tab, which delegates to render_current_run_tab.
- The visible Streamlit tab list is constructed in app.main.main before st.tabs(tab_labels).
- Prior blocks validated helper models, but browser acceptance requires literal app.main tab wiring and dispatch checks.
- Specialty/domain options were present in Document category because the real renderer still used specialty labels as document categories.
- Separate Document category and Medical specialty / domain controls belong in render_current_run_tab and render_adapter_fallback_panel.
- 11B/11C/11E helpers remain useful, but actual main() and renderer source must wire the visible controls directly.
- This block fixes tab construction, document category options, MKB Explorer dispatch, Review Queue dispatch, and selected-specialty fallback wiring.

- external API used: `False`
- auto-accept enabled: `False`

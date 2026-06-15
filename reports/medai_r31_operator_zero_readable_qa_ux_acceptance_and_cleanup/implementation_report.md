# MEDAI-R31-OPERATOR-ZERO-READABLE-QA-UX-ACCEPTANCE-AND-CLEANUP — implementation report

- overall_result: `PASS`
- Old problem: MKB Explorer showed the legacy staging metadata table and raw-JSON 'Review staging detail' first; the readable comparator was last and nested inside the staging block, so operators kept hunting for it.
- UI order fixed: `render_mkb_tab` now calls `_render_qa_comparator_section()` FIRST, then renders the legacy table + raw-JSON detail inside a collapsed 'Advanced / legacy MKB staging table and raw detail' expander.
- comparator first (before legacy): `True`; legacy above comparator: `False`; review-staging-detail above comparator: `False`.
- counts visible: extracted `179`, not-extracted `317`.
- readable headings visible: content `True`, sections `True`, items `True`, source `True`, qa decision `True`.
- full_schema non-placeholder `True`, minimal_review readable `True`, not-extracted terminal reason `True`.
- local QA save persisted: extracted `True`, not-extracted `True`.
- Evidence: rendered_ui_text_probe.txt, ui_evidence.json, operator_zero_acceptance_matrix.json, 3 header-cropped screenshots. No provider calls, no extraction, 0 active/verified; leaks 0/0/0.

# MEDAI-UI-CAPABILITY-RESTORE-11B Audit

- `specialty/domain selector` -> app/specialty_selection.py, app/main.py:render_specialty_selector
- `MKB Explorer` -> app/mkb_explorer_model.py, app/main.py:render_mkb_tab, app/main.py:PRIMARY_OPERATOR_TABS
- `tier/status visibility` -> app/mkb_explorer_model.py:build_mkb_explorer_model, app/main.py:render_mkb_tab
- `specialty routing` -> app/local_adapter_fallback_processor.py:selected_specialty, app/test_launcher.py existing specialty parameter

- external API used: `False`
- auto-accept enabled: `False`

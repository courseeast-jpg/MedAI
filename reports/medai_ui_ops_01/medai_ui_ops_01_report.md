# MEDAI-UI-OPS-01 Report

Conclusion: `medai_ui_ops_panel_ready`

## Summary

- Allowlisted commands: `12`
- Unknown command rejected: `True`
- Free-form shell enabled: `False`
- Full test suite confirmation required: `True`
- External API used: `False`
- Import performed: `False`

## Button Groups

| Group | Button | Command ID |
| --- | --- | --- |
| Main checks | Quick health check | `quick_health_check` |
| Main checks | Final MVP validation | `final_mvp_validation` |
| Advanced: full test suite | Full test suite | `full_test_suite` |
| Advanced: terminology checks | Terminology preflight | `terminology_source_preflight` |
| Advanced: terminology checks | Terminology inventory | `terminology_inventory` |
| Advanced: terminology checks | B07-TERM validation | `b07_term_validation` |
| Advanced: routing and extraction checks | ROUTE-FIX validation | `route_fix_validation` |
| Advanced: routing and extraction checks | Focused routing tests | `focused_routing_tests` |
| Main checks | Git safety check | `git_safety_check` |
| Advanced: reports and recovery | Last validation reports | `show_last_validation_reports` |
| Advanced: reports and recovery | Release tags | `show_release_tags` |
| Advanced: reports and recovery | Verify release bundle | `verify_final_bundle` |

## Safety

- No runtime clinical files are changed by this panel.
- No free-form shell or terminal input is exposed.
- Commands are selected by internal command ID from a fixed allowlist.
- Private acknowledgment contents and terminology source rows are not read by the panel.

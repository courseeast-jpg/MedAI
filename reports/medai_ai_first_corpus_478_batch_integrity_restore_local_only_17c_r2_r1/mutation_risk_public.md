# Mutation risk (report-only)

- file_writable_before: `True`
- file_writable_after: `False`
- file_set_readonly_after_restore: `True`
- file_size_after_bytes: `2107904`
- untracked_unexpected_scripts: `0`
- dirty_report_edits: `65`

An external writer outside this session has been mutating files under
`MedAI_Private` and editing committed reports. This block does not attribute
the writer to a specific process, kills no process, and removes no file. The
rebuilt JSONL is sealed (SHA256 sidecar) and set read-only when supported;
a future live runner must verify the SHA256 before any provider call.

# 15Z-C would-be Vertex request preview

Preview is fingerprint-only. The would-be request body is validated offline
with exactly `contents` and `generationConfig` top-level keys. No source
text, token map, vault contents, credentials, PDF, image, OCR payload, or
provider call is included in this report.

| Case | Request keys | Request fingerprint | Payload fingerprint | Vault fingerprint |
| --- | --- | --- | --- | --- |
| sanitized_redacted_real_like_basic_note | contents, generationConfig | `sha256:571f5d3874b76cf3` | `sha256:1e6e44f88e1c7a10` | `sha256:78b934bc24506fba` |
| sanitized_redacted_real_like_lab_report | contents, generationConfig | `sha256:7d04904ee6328e37` | `sha256:b6ab337adfa05a5b` | `sha256:57b0744ae67e4aec` |
| sanitized_contact_fields_fixture | contents, generationConfig | `sha256:fe9df182b9281ef2` | `sha256:0a2a5b7833b4083c` | `sha256:88961a4a3ce4c6d4` |
| all_9_pii_classes_sanitized_fixture | contents, generationConfig | `sha256:3e8efac2962ab1a5` | `sha256:2229d92106235aa7` | `sha256:a9454cdfcf7bb5c2` |
| all_future_gates_simulated_pass | contents, generationConfig | `sha256:da57f338b799c7a6` | `sha256:b31674101818d6b5` | `sha256:28d8ed0412fa4784` |

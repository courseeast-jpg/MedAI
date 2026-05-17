# MEDAI-RU-LAB-OCR-GATE-02A-FIX

Conclusion: medai_ru_lab_ocr_gate_02a_runtime_propagation_ready

Changed scope: runtime shadow marker propagation only.

Root cause: the original marker trigger required both medium/high digit density and an explicit table-like pattern. Runtime Run & Review assembly also preserved missing/null marker fields instead of computing safe marker metadata when text was available.

Fix summary: the gate now accepts medium/high digit density or table-like structure as the numeric/table signal, and Run & Review assembly computes safe marker metadata when extractor marker fields are absent.

Safety flags: no OCR fallback, no OCR engine change, no parser change, no confidence change, no acceptance change, no clinical interpretation, no external API, no B07 change, no ROUTE-FIX change, no DB schema change.

Full pytest: passed, 2409 passed, 4 skipped, 22 warnings.

Recommended next step: repeat the same small Russian Run & Review smoke test and inspect the raw run record.

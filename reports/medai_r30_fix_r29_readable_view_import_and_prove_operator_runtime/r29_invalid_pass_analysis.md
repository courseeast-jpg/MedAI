# Why R29 PASS was invalid for operator runtime

## Exact root cause
A **long-running Streamlit process started before R29** (observed: a `streamlit run app/main.py --server.port 8561` process alive since ~06:00, hours before the R29 code change) kept the **pre-R29 `app.mkb_all_records_qa_comparator` module cached in its `sys.modules`**. When the R29 code landed, `app/main.py` (re-run by Streamlit) executes `from app.mkb_all_records_qa_comparator import readable_record_view`, but Python returns the already-cached pre-R29 module object, which has the older symbols (`QA_STATUSES`, `build_all_records_qa_comparator`, `get_comparator_record_detail`, `save_qa_status`) but NOT `readable_record_view` -> `ImportError: cannot import name 'readable_record_view'`.

This is confirmed because the import fails at the 4th name in the tuple (the three pre-R29 names import fine) and because `python -c`, `runpy`, and a FRESH Streamlit process all import the symbol correctly from the same committed file.

## Why R29's PASS did not catch it
R29 proved visibility with a bespoke script that launched its OWN fresh Streamlit (with `MEDAI_R29_LIVE_UI_PROOF=1`). A fresh process imports the current module, so R29 passed -- but it never exercised the operator's already-running process, so the stale-`sys.modules` failure on the plain operator command went undetected.

## Fix
- The committed code (HEAD `ee90051`) is correct: `readable_record_view` is defined and exported in the comparator module and imported by `app/main.py` (no code change required).
- Operational fix: **restart the Streamlit app** so it re-imports the current module. This block kills any stale `streamlit run app/main.py` process and clears bytecode.
- Proof: fresh-subprocess import smoke tests + a live Playwright proof on the EXACT operator command `streamlit run app/main.py --server.port 8561` (no ImportError; readable QA screen renders; counts 179 / 317; all five headings visible).

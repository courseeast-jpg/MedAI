# R27 Invalid PASS Analysis

R27 is treated as invalid because it reported PASS without live rendered UI proof.

- R27 checked helper/static tab labels and source strings.
- R27 did not start Streamlit.
- R27 did not inspect rendered DOM tab labels.
- R27 therefore could not detect a stale or different running Streamlit session that omitted MKB Explorer.

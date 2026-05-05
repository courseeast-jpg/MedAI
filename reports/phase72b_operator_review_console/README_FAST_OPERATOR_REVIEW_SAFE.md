# Fast Operator Review — Phase 72B (SAFE)

## One command to start

```bash
# Review tier-1 items (3 items, recommended starting point)
python scripts/run_phase72b_operator_review_console.py --tier 1

# Review tier-2 items
python scripts/run_phase72b_operator_review_console.py --tier 2

# Review all pending
python scripts/run_phase72b_operator_review_console.py --all

# Non-interactive summary only
python scripts/run_phase72b_operator_review_console.py --summary

# Launch Streamlit operator feedback UI
python scripts/run_phase72b_operator_review_console.py --launch-ui
```

## Terminal wizard

For each pending item the wizard shows:
1. safe_file_id, priority tier, problem class
2. Review goal
3. Operator question
4. Numbered answer options

**Keystrokes:**
- `1`–`10` — select answer
- `s` — skip this item
- `q` — quit and save summary

Answers are saved immediately to the gitignored private feedback file.
A fresh public summary is generated on exit.

## Streamlit UI

```bash
streamlit run app/operator_feedback.py
```

Provides clickable answer buttons, progress bar, and automatic advance.

## Privacy rules

- Do NOT type patient names or other PHI at any prompt.
- Answers are saved to `operator_feedback_PRIVATE.json` (gitignored).
- Public summary contains only counts and safe_file_ids — no notes.

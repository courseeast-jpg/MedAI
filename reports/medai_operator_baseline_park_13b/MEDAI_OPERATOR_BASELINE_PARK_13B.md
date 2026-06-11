# MEDAI-OPERATOR-BASELINE-PARK-13B

## Snapshot

- Current HEAD: `59b68c6`
- Branch: `clinical-knowledge-architecture (detached worktree)`
- Expected baseline HEAD reachable: `True`
- Baseline status: `parked`
- Privacy result: `passed`
- External API used: `False`
- Auto-accept: `False`

## Completed Blocks

- 12B file intake multiformat: `a8b32bc` - supported multiformat intake and queueing works
- 12C local image OCR routing: `86b34f1` - guarded local image OCR routing works
- 12D OCR review-bound routing: `2f5f99e` - OCR-derived records remain quarantined and review-bound
- 12E0 privacy regression repair: `1142032` - historical 11F privacy regression repaired
- 12A operator workflow UAT: `b6f8e16` - synthetic end-to-end operator UAT passed
- 13A operator console redesign: `fab50a8` - operator-first console layout implemented
- 13A-R1 stale queue state fix: `59b68c6` - stale queue state copy fixed

## Functional Baseline

- Supported file types: `PDF, TXT, PNG, JPG/JPEG, TIFF/TIF, BMP, DOCX`
- Local OCR/text recovery: guarded local recovery is available for supported image inputs
- Review-bound MKB persistence: OCR-derived and extracted records remain quarantined until human review
- MKB Explorer: defaults and counts emphasize review-bound records
- Review Queue: shows actionable records for human review
- Accept/reject/defer behavior: accept, reject, and defer apply to selected records only
- No cloud: `True`
- No auto-accept: `True`

## Browser Acceptance Summary

- Queue count visible: `True`
- Queued files observed: `12`
- Start reason accurate: `True`
- Run completed: `True`
- Stale queue message fixed: `True`
- Review-bound records visible: `True`
- Review Queue records visible: `True`
- Cloud APIs off: `True`
- Auto-accept off: `True`

## Known Limitations

- Real-world OCR quality may vary by scan or photo quality.
- The active 28 records observed in the local browser database are pre-existing and were not proven OCR-derived by the 12E dry run.
- Synthetic UAT does not use private medical files.
- MedAI is not a medical device and does not diagnose or recommend treatment.

## Operator Command

```powershell
$env:MEDAI_ALLOW_EXTERNAL_API='false'; $env:MEDAI_LOCAL_ONLY='true'; python -m streamlit run app/main.py
```

## Validation Receipts

- Key report files exist: `True`
- 13A-R1 validation receipt ready: `True`
- 12A UAT receipt ready: `True`
- No runtime DBs/images/uploads staged: `True`
- Raw OCR text in report: `False`
- Private paths in report: `False`
- Next recommended future block: None required for this parking snapshot.

# MEDAI-R24-MKB-WRITE-TARGET-RECONCILIATION-AND-VISIBILITY-FIX

## Root Cause

R23 wrote durable review-required rows to a local-private SQLite staging table, while the app MKB Explorer only read the normal active MKB `records` table.

## Fix

The Explorer model now merges public-safe R23 review staging metadata into the MKB Explorer view without promoting records or copying raw payloads.

## Result

- Overall result: `PASS`
- R23 durable records found: `496`
- R23 visible in MKB Explorer: `496`
- Active/verified records created: `0`
- Content/extracted: `179`
- Review-only metadata: `315`
- Non-sendable metadata: `2`
- Privacy result: `passed`

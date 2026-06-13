# Vertex Decision Audit — Operator UAT Transcript (15V, no-live, read-only)

_UAT passed: True_

| Step | Operator action | Observed |
| --- | --- | --- |
| 1 | Locate 'Vertex Decision Audit' in advanced operator navigation | `True` |
| 2 | Open the tab via app dispatch (routes to read-only 15T hook) | `True` |
| 3 | Panel title shown: 'Vertex Semantic Review Decision Audit' | `True` |
| 4 | No-live/read-only indicator shown | `True` |
| 5 | Read-only export statement shown | `True` |
| 6 | Provider route/model shown (vertex / gemini-2.5-flash-lite) | `True` |
| 7 | Decision summary shown (total 15) | `True` |
| 8 | Package-family breakdown shown | `True` |
| 9 | Safety status shown (writes 0, auto-accept off, review required) | `True` |
| 10 | Evidence/provenance preview shown (anchors, source evidence, source refs, audit reasons) | `True` |
| 11 | Unknown values shown | `True` |
| 12 | Uncertainty flags shown | `True` |
| 13 | Hallucinated field count shown and equals 0 | `True` |
| 14 | JSON export affordance visible + artifact exists | `True` |
| 15 | CSV export affordance visible + artifact exists | `True` |
| 16 | Markdown export affordance visible + artifact exists | `True` |
| 17 | Exports are read-only references to 15S artifacts | `True` |
| 18 | Decision store fingerprint unchanged before/after UAT | `True` |
| 19 | No active MKB write paths invoked | `True` |
| 20 | No provider/network call paths invoked | `True` |

No live provider call, no active MKB write, and the 15R decision store fingerprint is unchanged.

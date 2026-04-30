# MedAI Review Audit

- Source report: `G:\Codex\2026-04-22-connect-github\reports\batch_validation\latest_batch_validation.json`
- Reviewed files: `6`

## Review Fix Breakdown

- no_entities: `1`
- low_entity_count: `3`
- low_confidence: `2`
- low_coverage: `0`
- low_diversity: `0`
- extractor_issue: `0`

## 22.pdf

- confidence: 0.45
- entities: []
- why reviewed: ['empty_extraction', 'confidence_below_threshold', 'low_entity_count', 'low_coverage', 'low_diversity']
- recommended fix: no_entities

- preview:
[Page 1] Each time divisiot ach time 2 , ie so 10 20 aa <a > =a _ ine 1 i 7 .

## 23.pdf

- confidence: 0.45
- entities: ['INNGE EE']
- why reviewed: ['confidence_below_threshold', 'low_entity_count', 'low_extractor_weight']
- recommended fix: low_entity_count

- preview:
[Page 1] Each thee diviel” “a o 2 m0 iH ———T . Total volume: €25 mi Poa time a ae Tiverosl ease ss io 2 ww No INNGE EE Woe

## 3-2.pdf

- confidence: 0.45
- entities: ['Nitrite', 'Blood']
- why reviewed: ['confidence_below_threshold', 'low_extractor_weight']
- recommended fix: low_confidence

- preview:
[Page 1] Blood UA Bilirubin UA Normal value: Negative Normal value: Negative Value Value Trace Negative Urobilinogen UA Nitrite UA Normal range: 0.2 - 1.0 mg/dL Normal value: Negative 0.2 = Value Sees ‘ 02 1 Negative Microscopic Examination Value See below: Microscopic was indicated and was performe

## 8-1.pdf

- confidence: 0.45
- entities: ['Urine Cytology']
- why reviewed: ['confidence_below_threshold', 'low_entity_count', 'low_extractor_weight']
- recommended fix: low_entity_count

- preview:
[Page 1] CYTOLOGY, URINE Collected on [DATE] Results

## 8-2.pdf

- confidence: 0.45
- entities: ['Recommendation']
- why reviewed: ['confidence_below_threshold', 'low_entity_count', 'low_extractor_weight']
- recommended fix: low_entity_count

- preview:
[Page 1] Source: Clinician provided ICD10: Value Value Comment Comment DIAGNOSIS: Recommendation: Value Value Comment Comment Signed out by: Performed by: Value Value Comment Comment Gross description: Value Comment

## 9.pdf

- confidence: 0.45
- entities: ['Urine Culture', 'Final Report']
- why reviewed: ['confidence_below_threshold', 'low_extractor_weight']
- recommended fix: low_confidence

- preview:
[Page 1] Urine Culture, Routine Value Final report

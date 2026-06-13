# 15Z-B outbound-safe payload preview (tokens only)

Only accepted, fully-redacted outbound payloads are previewed. Every previewed
value below is a `[CLASS_n]` token; no raw PII-like value is present.

## redacted_real_like_basic_note_with_name_dob_mrn

```
Patient: [PATIENT_NAME_1]
DOB: [DATE_1]
MRN: [MRN_1]
Note: Sodium 140 mmol/L within reference range.
```

## redacted_real_like_lab_report_with_facility_accession_provider

```
Facility: [FACILITY_1]
Accession: [ACCESSION_1]
Provider: [PROVIDER_1]
Result: Hemoglobin 13.2 g/dL within range.
```

## redacted_real_like_contact_fields_phone_email_address

```
Phone: [PHONE_1]
Email: [EMAIL_1]
Address: [ADDRESS_1]
Result: Urinalysis clear, no abnormalities.
```

## multi_section_report_with_repeated_same_identifier

```
Section A
MRN: [MRN_1]
Patient: [PATIENT_NAME_1]
Section B
MRN: [MRN_1]
Note: Glucose 95 mg/dL within range.
```

## sanitized_redacted_real_like_ready_for_no_live_replay_only

```
Patient: [PATIENT_NAME_1]
DOB: [DATE_1]
MRN: [MRN_1]
Facility: [FACILITY_1]
Accession: [ACCESSION_1]
Provider: [PROVIDER_1]
Phone: [PHONE_1]
Email: [EMAIL_1]
Address: [ADDRESS_1]
Note: Comprehensive panel within reference ranges.
```

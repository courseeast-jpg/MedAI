from extractors.spacy_extractor import SpacyExtractor
from execution.supplemental_rules import supplemental_entities


def test_spacy_extractor_filters_page_number_test_result():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "Page 1\nHospital medication review. Diagnosis: seizure disorder. "
        "Levetiracetam 1000mg twice daily. Sodium 139 mmol/L."
    )

    test_results = [entity for entity in result["entities"] if entity["type"] == "test_result"]

    assert all(entity["text"].lower() != "page" for entity in test_results)


def test_spacy_extractor_keeps_real_numeric_test_result():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "Page 1\nHospital medication review. Diagnosis: seizure disorder. "
        "Levetiracetam 1000mg twice daily. Sodium 139 mmol/L."
    )

    test_results = [entity for entity in result["entities"] if entity["type"] == "test_result"]

    assert any(
        entity["text"] == "Sodium" and entity["value"] == "139" and entity["unit"] == "mmol/L"
        for entity in test_results
    )


def test_spacy_extractor_keeps_existing_medication_and_diagnosis_patterns():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "Diagnosis: seizure disorder. Levetiracetam 1000mg twice daily. Valproate 500mg nightly."
    )

    assert any(entity["type"] == "diagnosis" and entity["text"] == "seizure disorder" for entity in result["entities"])
    assert any(
        entity["type"] == "medication" and entity["text"].lower() == "levetiracetam" and entity["dose"] == "1000mg"
        for entity in result["entities"]
    )
    assert any(
        entity["type"] == "medication" and entity["text"].lower() == "valproate" and entity["dose"] == "500mg"
        for entity in result["entities"]
    )


def test_urinalysis_supplemental_rules_extract_common_findings():
    extractor = SpacyExtractor()

    result = extractor.extract("UA BLOOD positive RBC 10-20 /hpf calcium oxalate crystals present")
    entity_texts = {entity["text"].upper() for entity in result["entities"]}

    assert "BLOOD" in entity_texts
    assert "RBC" in entity_texts
    assert "CALCIUM OXALATE CRYSTALS" in entity_texts
    assert result["supplemental_rules_applied"] is True
    assert result["supplemental_entity_count"] >= 3
    assert result["final_entity_count_after_supplement"] == len(result["entities"])


def test_supplemental_rules_do_not_duplicate_existing_entities():
    additions = supplemental_entities(
        "RBC seen. BLOOD positive.",
        existing_entities=[
            {"type": "test_result", "text": "RBC"},
            {"type": "test_result", "text": "Blood"},
        ],
    )

    assert additions == []


def test_urology_rules_extract_urine_cytology():
    extractor = SpacyExtractor()

    result = extractor.extract("CYTOLOGY, URINE Collected on [DATE] Results")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert "Urine Cytology" in entity_texts
    assert result["supplemental_rules_applied"] is True


def test_urology_rules_extract_urine_culture_and_final_report():
    extractor = SpacyExtractor()

    result = extractor.extract("Urine Culture, Routine Value Final report")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert "Urine Culture" in entity_texts
    assert "Final Report" in entity_texts
    assert result["supplemental_rules_applied"] is True


def test_urology_rules_extract_ua_panel_specific_entities():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "Blood UA Bilirubin UA Urobilinogen UA Nitrite UA "
        "Microscopic Examination Value See below"
    )
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert "Blood UA" in entity_texts
    assert "Nitrite UA" in entity_texts
    assert "Bilirubin UA" in entity_texts
    assert "Urobilinogen UA" in entity_texts


def test_recommendation_alone_does_not_auto_accept_without_supporting_entities():
    extractor = SpacyExtractor()

    result = extractor.extract("Recommendation: Value")

    assert any(entity["type"] == "recommendation" and entity["text"] == "Recommendation" for entity in result["entities"])
    assert result["confidence"] < 0.65


def test_noisy_ocr_text_becomes_readable_before_extraction():
    extractor = SpacyExtractor()

    result = extractor.extract("UR0KULTURE |||| NEGAT1V ____ VERDHE")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert result["normalization_applied"] is True
    assert "Urine Culture" in result["normalized_text_preview"]
    assert "Negative" in result["normalized_text_preview"]
    assert "Yellow" in result["normalized_text_preview"]
    assert "Urine Culture" in entity_texts


def test_foreign_lab_terms_map_to_known_entities():
    extractor = SpacyExtractor()

    result = extractor.extract("UROKULTURE NEGATIV VERDHE")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert result["normalization_applied"] is True
    assert "Urine Culture" in entity_texts


def test_clean_english_report_does_not_apply_normalization():
    extractor = SpacyExtractor()

    result = extractor.extract("Urine Culture, Routine Value Final report")
    entity_texts = {entity["text"] for entity in result["entities"]}

    assert result["normalization_applied"] is False
    assert "Urine Culture" in entity_texts
    assert "Final Report" in entity_texts


def test_structured_lab_parser_extracts_ua_blood_trace_abnormal_block():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "UA Blood\nValue: Trace\nAbnormal\n"
    )

    structured_entities = [
        entity for entity in result["entities"]
        if entity.get("entity_type") == "lab_result" and entity.get("source") == "structured_lab_parser"
    ]

    assert result["structured_parser_used"] is True
    assert result["structured_entities_count"] >= 1
    assert result["structured_parser_version"] == "structured_lab_parser_v1"
    assert result["confidence"] == 0.83
    assert result["confidence_breakdown"]["entity_count"] == 0.6
    assert result["confidence_breakdown"]["extractor_weight"] == 0.8
    assert len(result["entities"]) > 0
    assert any(
        entity["type"] == "test_result"
        and entity["text"] == "UA Blood"
        and entity["test_name"] == "UA Blood"
        and entity["value"] == "Trace"
        and entity["status"] == "Abnormal"
        for entity in structured_entities
    )


def test_structured_lab_parser_extracts_ua_rbc_and_ua_crystals_layouts():
    extractor = SpacyExtractor()

    result = extractor.extract(
        "UA RBC\n5-10 /HPF\nAbnormal\n\n"
        "UA Crystals\nPositive\nHigh\n"
    )

    structured_entities = [
        entity for entity in result["entities"]
        if entity.get("entity_type") == "lab_result" and entity.get("source") == "structured_lab_parser"
    ]

    assert result["structured_parser_used"] is True
    assert result["structured_entities_count"] >= 2
    assert result["confidence"] == 0.83
    assert result["confidence_breakdown"]["entity_count"] == 0.6
    assert result["confidence_breakdown"]["extractor_weight"] == 0.8
    assert len(result["entities"]) >= 2
    assert any(
        entity["test_name"] == "UA RBC"
        and entity["value"] == "5-10"
        and entity["unit"] == "/HPF"
        and entity["status"] == "Abnormal"
        for entity in structured_entities
    )
    assert any(
        entity["test_name"] == "UA Crystals"
        and entity["value"] == "Positive"
        and entity["status"] == "High"
        for entity in structured_entities
    )

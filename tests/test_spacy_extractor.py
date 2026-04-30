from extractors.spacy_extractor import SpacyExtractor


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
    assert result["confidence"] == 0.625
    assert result["confidence_breakdown"]["entity_count"] == 0.3
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
    assert result["confidence"] == 0.7
    assert result["confidence_breakdown"]["entity_count"] == 0.3
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

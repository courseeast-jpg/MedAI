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

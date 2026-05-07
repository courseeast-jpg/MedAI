"""CKA-TERM-01G synthetic scale fixtures and streaming checks."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from clinical_knowledge.terminology.models import TerminologySystem
from clinical_knowledge.terminology.parsers import (
    ParseResult,
    parse_loinc_csv,
    parse_rxnorm_rxnconso,
    parse_snomed_concept_description,
    parse_umls_mrconso,
)


@dataclass(frozen=True)
class SyntheticScaleFixture:
    system: str
    row_count: int
    files: dict[str, Path]
    safe_fixture_id: str

    def safe_public_summary(self) -> dict:
        return {
            "system": self.system,
            "row_count": self.row_count,
            "file_count": len(self.files),
            "safe_fixture_id": self.safe_fixture_id,
        }


def generate_synthetic_umls_rrf(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for idx in range(rows):
            fields = [""] * 18
            fields[0] = f"C{idx:07d}"
            fields[1] = "ENG"
            fields[11] = "SYN"
            fields[13] = f"UMLS{idx:07d}"
            fields[14] = f"Synthetic UMLS Concept {idx}"
            handle.write("|".join(fields) + "|\n")


def generate_synthetic_rxnorm_rrf(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for idx in range(rows):
            fields = [""] * 18
            fields[0] = f"{1000000 + idx}"
            fields[1] = "ENG"
            fields[11] = "RXNORM"
            fields[12] = "IN"
            fields[13] = f"RX{idx:07d}"
            fields[14] = f"Synthetic RxNorm Concept {idx}"
            handle.write("|".join(fields) + "|\n")


def generate_synthetic_snomed_rf2(concept_path: Path, description_path: Path, rows: int) -> None:
    concept_path.parent.mkdir(parents=True, exist_ok=True)
    description_path.parent.mkdir(parents=True, exist_ok=True)
    with concept_path.open("w", encoding="utf-8", newline="") as concepts:
        concepts.write("id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n")
        for idx in range(rows):
            concepts.write(f"{900000000 + idx}\t20240101\t1\t0\t0\n")
    with description_path.open("w", encoding="utf-8", newline="") as descriptions:
        descriptions.write("id\teffectiveTime\tactive\tmoduleId\tconceptId\tlanguageCode\ttypeId\tterm\tcaseSignificanceId\n")
        for idx in range(rows):
            concept_id = 900000000 + idx
            descriptions.write(
                f"{800000000 + idx}\t20240101\t1\t0\t{concept_id}\ten\t900000000000003001\tSynthetic SNOMED Concept {idx}\t0\n"
            )


def generate_synthetic_loinc_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["LOINC_NUM", "COMPONENT", "LONG_COMMON_NAME"])
        for idx in range(rows):
            writer.writerow([f"SYN-{idx}", f"Synthetic Component {idx}", f"Synthetic LOINC Concept {idx}"])


def build_scale_fixtures(root: Path, *, rows_per_system: int = 120) -> list[SyntheticScaleFixture]:
    fixtures: list[SyntheticScaleFixture] = []
    umls = root / "umls" / "MRCONSO.RRF"
    generate_synthetic_umls_rrf(umls, rows_per_system)
    fixtures.append(SyntheticScaleFixture("umls", rows_per_system, {"mrconso": umls}, "scale_umls"))

    rxnorm = root / "rxnorm" / "RXNCONSO.RRF"
    generate_synthetic_rxnorm_rrf(rxnorm, rows_per_system)
    fixtures.append(SyntheticScaleFixture("rxnorm", rows_per_system, {"rxnconso": rxnorm}, "scale_rxnorm"))

    concept = root / "snomed_ct" / "sct2_Concept_Synthetic.txt"
    description = root / "snomed_ct" / "sct2_Description_Synthetic.txt"
    generate_synthetic_snomed_rf2(concept, description, rows_per_system)
    fixtures.append(SyntheticScaleFixture("snomed_ct", rows_per_system, {"concept": concept, "description": description}, "scale_snomed"))

    loinc = root / "loinc" / "Loinc.csv"
    generate_synthetic_loinc_csv(loinc, rows_per_system)
    fixtures.append(SyntheticScaleFixture("loinc", rows_per_system, {"loinc": loinc}, "scale_loinc"))
    return fixtures


def parse_scale_fixture(fixture: SyntheticScaleFixture, *, max_rows: int) -> ParseResult:
    system = TerminologySystem(fixture.system)
    if system == TerminologySystem.UMLS:
        return parse_umls_mrconso(path=str(fixture.files["mrconso"]), max_rows=max_rows)
    if system == TerminologySystem.RXNORM:
        return parse_rxnorm_rxnconso(path=str(fixture.files["rxnconso"]), max_rows=max_rows)
    if system == TerminologySystem.SNOMED_CT:
        return parse_snomed_concept_description(
            concept_path=str(fixture.files["concept"]),
            description_path=str(fixture.files["description"]),
            max_rows=max_rows,
        )
    if system == TerminologySystem.LOINC:
        return parse_loinc_csv(path=str(fixture.files["loinc"]), max_rows=max_rows)
    raise ValueError(f"unsupported synthetic system: {fixture.system}")

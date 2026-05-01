# Phase 42 Failed Lab Table Forensics

- Generated at: `2026-05-01T19:19:44.692211+00:00`
- Targets: `Test Results 3.pdf`, `Test Results 6.pdf`

## Summary

- Files analyzed: `2`
- Files missing: `0`
- Bottleneck category distribution: `{'B': 1, 'A': 1}`

## Category Legend

- **A** — OCR/Layout candidate generation
- **B** — Table block segmentation
- **C** — Row parser
- **D** — Classifier threshold/reasoning
- **E** — True manual-review boundary

## Test Results 3.pdf

### OCR/Layout

- Engine: `existing_pdf_pipeline`
- Quality score: `0.811`
- Quality band: `good`
- Route decision: `digital_clean_text`
- Quality warnings: `['low_medical_token_density']`
- Raw text length: `3900`

### Classification

- Final status: `review_ocr_quality`
- Entity count: `4`
- Confidence: `0.63`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, classifier_legacy_ocr_flag, legacy_normalized_low_coverage`
- Legacy OCR codes: `legacy_normalized_low_coverage`

### Lab Normalization

- Lab table detected: `False`
- Coverage ratio: `0.0`
- Coverage band: `none`
- Reason codes: `non_lab_document_skipped_lab_normalization, microbiology_pcr_report_detected, language_aware_ocr_required`
- Would upgrade review_ocr_quality→review: `False`

### Forensic Parse

- Candidate lines: `55`
- Parsed rows: `0`
- Rejected lines: `55`
- Rejection breakdown: `{'empty_or_noise': 1, 'unrecognized': 8, 'no_unit_or_qualitative': 41, 'name_only': 2, 'malformed_lab_row': 3}`

#### Signals

- value-only lines present: `False`
- name-only lines present: `True`
- range-only lines present: `False`
- appears table-like without separators: `True`
- text too fragmented (>50% short lines): `False`
- text too sparse (<3 lines): `False`
- short_line_ratio: `0.091`

#### Top candidate lines (up to 50)

```
[Page 1]
LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH
MeTOAOM nl..tp B pe>KHMe peailbHOrO BpeMeHH
AHApocpnop®,AHApOcpnop®CKpHH
OnHcaHHe 6naHKa peJynbTaToB
V1CCile,Q08aHIIIe np080,QIIITCfl MeTO,QOM nOI1111Mepa3HOVl 1...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII.
C 1...\eilbiO :3Tlt10IlOrlll4eCKOVl ,QIIIarHOCTIIIKIII 111HcpeKL\lt10HH0-80Cnaili!ITeilbHbiX 3a60ile8aHIIIVl M04enOI1080Vl
c~t~creMbl MY)K4lt1H 8 aHanlt131t1pyeMOM 6~t~oMarep~t~ane O,QHOspeMeHHO BblnOilHfltOr:
- onpe,QeneHIIIe Haillt14111f!/orcyrcT8111f! naroreHo8: Neisseria gonorrhoeae, Chlamydia trachomatis,
Mycoplasma genitalium, Trichomonas vaginalis;
- KOillt14eCT8eHHYIO 01...\eHKY 8CeX 6aKreplt1Vl (06Ll.\af! 6aKrep111ailbHafl Macca - 06M), HOpMOcpi10pb1 lt1
ycno8Ho-naroreHHbiX MIIIKpoopraHlt13M08; TepMIIIH "YnM, accOL\IIIItlposaHHble c 6aK8arlt1H030M" IIICnOilb3YIOT
Ailfl o6o3Ha4eHlt1fl rpynnbl M111KpoopraHIII3M08, 8nep8ble 8blf!8IleHHbiX y )KeHLl.\IIIH. B HaCTOf!Ll.\ee 8peMf!
AOKa3aHa ponb 3Tlt1X MIIIKpoopraHIII3MOB s pa38lt1Tlt1lt1 yporeHIIITailbHbiX 3a6one8aHIIIVI y MY)K4111H*.
- KOillt14eCT8eHHYIO Ol...\eHKY rp111608 po,Qa Candida.
Kon~t~4ecr8eHHble pe3ynbrarbl IIICcne,Qo8aHlt1fl npe,QcrasneHbl 8 reHoM-3KBIIIBaneHrax (r3), 3Ha4eHlt1fl
KOTOpbiX nponOpl...\IIIOHailbHbl MIIIKp06HOVl 06CeMeHeHHOCTlt1 yporeHlt1T8IlbHOrO 6lt10TOna. A6COiliOTHble
3Ha4eHlt1fl r3 np11180,QfiTCf! 8 CTOI16l...\e 6naHKa «Pe3yilbTaTbl. KOI11114eCT8eHHbiVl».
0THOClt1TeilbHble nOKa3aTeillt1 npe,QCTaBileHbl 8 CTOI16l...\e 6naHKa «Pe3yilbTaTbl. 0THOClt1TeilbHbiVl» B ABYX
cpopMaTaX: B Blti,Qe pa3HIIIl...\bl a6COiliOTHbiX 3Ha4eH~Vl Ka)K,Qoro 1113 noKa3aTeileVllt1 06M (lg1 0) Ill B npOL\eHTaX
(%) or 06M. 3Ha4eHIIIf! noKa3areneV! 8 npol...\eHrax (% ), rpa.QIIIl...\IIIOHHOM cpopMare Ailfl KOillt14ecrseHHbiX
,QaHHbiX, nplt18e,QeHbl cnpaB04HO, O,QHaKO 8 paC4eTHOM anrOpiiiTMe 3aKI1104eHIIIfl OHitl He IIICnOilb3YIOTCfl,
CYMMlt1p08aTb npOL\eHTbl (%) HeKOppeKTHO.
,[J,nfl APO)K)Keno,Qo6HbiX rp~t~6os 111 MIIIKOnna3M (Ureaplasma urealyticum, Ureaplasma parvum,
Mycoplasma hominis) 8bi,QatOTCfl TOilbKO a6conteTHble 3Ha4BHIIIf!.
np111 cpopMIIIpOBaHIIIIII 3aKI1104eHIIIfl lt1CnOilb3YIOTCfl noKa3aTBillll COOTHOWeHIIIVl pa3HbiX
Mlt1KpoopraHIII3M08/rpynn M111KpoopraHlt13M08 C 06M Ill Me)K,Qy C060V!, KOTOpble xapaKTep1113YIOT COCTOf!HIIIe
6IIIOL\eHo3a. CreneHb AIIIC61!1o3a Ol...\eHit18aercf! TOilbKO np111 06M>105.
,[J,nfl y,Qo6cr8a rpaKT08KIII pe3yilbTaros** s ra6Illt1l...\e IIICnOilb308aHa l...\BeT08af! MapK111p08Ka. B
3a8111CIIIMOCTIII OT 1113Mepf!eMOrO napaMerpa MapKepbl 0603Ha4aiOT Cile,QytOLl.\ee:
KoHTPOilbHble noKa3aTemll (reHOMHa~ ,QHK 4enoseKa, o6Ll.\a~ 6aKTepL1alibHa~ Macca, TpaH3L1TOpHa~ ML1Kpo¢nopa):
D •
COOT8eTCT8111e KOIIITeOitlf!M
He COOT8eTCT8111e KOIIITeOitlf!M
HopMoQ>nopa:
D •
D •
COOT8eTCT8lt1e KOlt1TeOlt1f!M HOOMbl
VMeOeHHOe OTKilOHeHitle OT KDlt1TeOlt1e8 HOOMbl
8bi08)KeHHOe OTKilOHeHitle OT KOIIITeOIIIe8 HOOMbl
COOT8eTCT8lt1e KOlt1TeOlt1f!M HOOMbl
VMeOeHHOe OTKilOHeHitle OT KDlt1TeOIIIe8 HOOMbl
8bi08)KeHHOe OTKilOHeHitle OT KOIIITeOIIIe8 HOOMbl
naToreHbl:
D •
He 8blf!8IleHO
06H80V)KeHO
Pe3yilbTaT, 8 KOTOpOM 3Ha4eHIIIe noKa3aTeils:l 05M Hlti)Ke noporo80ro 3H84eHIIIs:l C04eTaeTCs:l CO
3Ha4eHlt1eM noKa3arens:J «reHOMHas:J ,D,HK 4eil08eKa» 8b1We noporo8oro 3Ha4eHlt1s:l, rpaKryercs:J KaK
«HopMOL\eH03» 6e3 Ol..\eHKIII crpyKrypbi Mlt1Kpo6~t~oMa. B 3TOM cny4ae l..\8eT08as:J MapK~t~posKa 8 6naHKe
```

#### Rejected lines

| # | reason | partial | line |
| ---: | --- | --- | --- |
| 0 | empty_or_noise |  | [Page 1] |
| 1 | unrecognized |  | LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH |
| 2 | unrecognized |  | MeTOAOM nl..tp B pe>KHMe peailbHOrO BpeMeHH |
| 3 | unrecognized |  | AHApocpnop®,AHApOcpnop®CKpHH |
| 4 | no_unit_or_qualitative | name+value | OnHcaHHe 6naHKa peJynbTaToB |
| 5 | no_unit_or_qualitative | name+value | V1CCile,Q08aHIIIe np080,QIIITCfl MeTO,QOM nOI1111Mepa3HOVl 1...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII. |
| 6 | no_unit_or_qualitative | name+value | C 1...\eilbiO :3Tlt10IlOrlll4eCKOVl ,QIIIarHOCTIIIKIII 111HcpeKL\lt10HH0-80Cnaili!ITeilbHbiX 3a60ile8aHIIIVl M04enOI1080Vl |
| 7 | no_unit_or_qualitative | name+value | c~t~creMbl MY)K4lt1H 8 aHanlt131t1pyeMOM 6~t~oMarep~t~ane O,QHOspeMeHHO BblnOilHfltOr: |
| 8 | no_unit_or_qualitative | name+value | - onpe,QeneHIIIe Haillt14111f!/orcyrcT8111f! naroreHo8: Neisseria gonorrhoeae, Chlamydia trachomatis, |
| 9 | name_only | name | Mycoplasma genitalium, Trichomonas vaginalis; |
| 10 | no_unit_or_qualitative | name+value | - KOillt14eCT8eHHYIO 01...\eHKY 8CeX 6aKreplt1Vl (06Ll.\af! 6aKrep111ailbHafl Macca - 06M), HOpMOcpi10pb1 lt1 |
| 11 | no_unit_or_qualitative | name+value | ycno8Ho-naroreHHbiX MIIIKpoopraHlt13M08; TepMIIIH "YnM, accOL\IIIItlposaHHble c 6aK8arlt1H030M" IIICnOilb3YIOT |
| 12 | no_unit_or_qualitative | name+value | Ailfl o6o3Ha4eHlt1fl rpynnbl M111KpoopraHIII3M08, 8nep8ble 8blf!8IleHHbiX y )KeHLl.\IIIH. B HaCTOf!Ll.\ee 8peMf! |
| 13 | no_unit_or_qualitative | name+value | AOKa3aHa ponb 3Tlt1X MIIIKpoopraHIII3MOB s pa38lt1Tlt1lt1 yporeHIIITailbHbiX 3a6one8aHIIIVI y MY)K4111H*. |
| 14 | no_unit_or_qualitative | name+value | - KOillt14eCT8eHHYIO Ol...\eHKY rp111608 po,Qa Candida. |
| 15 | no_unit_or_qualitative | name+value | Kon~t~4ecr8eHHble pe3ynbrarbl IIICcne,Qo8aHlt1fl npe,QcrasneHbl 8 reHoM-3KBIIIBaneHrax (r3), 3Ha4eHlt1fl |
| 16 | no_unit_or_qualitative | name+value | KOTOpbiX nponOpl...\IIIOHailbHbl MIIIKp06HOVl 06CeMeHeHHOCTlt1 yporeHlt1T8IlbHOrO 6lt10TOna. A6COiliOTHble |
| 17 | no_unit_or_qualitative | name+value | 3Ha4eHlt1fl r3 np11180,QfiTCf! 8 CTOI16l...\e 6naHKa «Pe3yilbTaTbl. KOI11114eCT8eHHbiVl». |
| 18 | no_unit_or_qualitative | name+value | 0THOClt1TeilbHble nOKa3aTeillt1 npe,QCTaBileHbl 8 CTOI16l...\e 6naHKa «Pe3yilbTaTbl. 0THOClt1TeilbHbiVl» B ABYX |
| 19 | malformed_lab_row | partial | cpopMaTaX: B Blti,Qe pa3HIIIl...\bl a6COiliOTHbiX 3Ha4eH~Vl Ka)K,Qoro 1113 noKa3aTeileVllt1 06M (lg1 0) Ill B npOL\eHTaX |
| 20 | malformed_lab_row | partial | (%) or 06M. 3Ha4eHIIIf! noKa3areneV! 8 npol...\eHrax (% ), rpa.QIIIl...\IIIOHHOM cpopMare Ailfl KOillt14ecrseHHbiX |
| 21 | no_unit_or_qualitative | name+value | ,QaHHbiX, nplt18e,QeHbl cnpaB04HO, O,QHaKO 8 paC4eTHOM anrOpiiiTMe 3aKI1104eHIIIfl OHitl He IIICnOilb3YIOTCfl, |
| 22 | malformed_lab_row | partial | CYMMlt1p08aTb npOL\eHTbl (%) HeKOppeKTHO. |
| 23 | no_unit_or_qualitative | name+value | ,[J,nfl APO)K)Keno,Qo6HbiX rp~t~6os 111 MIIIKOnna3M (Ureaplasma urealyticum, Ureaplasma parvum, |
| 24 | no_unit_or_qualitative | name+value | Mycoplasma hominis) 8bi,QatOTCfl TOilbKO a6conteTHble 3Ha4BHIIIf!. |
| 25 | no_unit_or_qualitative | name+value | np111 cpopMIIIpOBaHIIIIII 3aKI1104eHIIIfl lt1CnOilb3YIOTCfl noKa3aTBillll COOTHOWeHIIIVl pa3HbiX |
| 26 | no_unit_or_qualitative | name+value | Mlt1KpoopraHIII3M08/rpynn M111KpoopraHlt13M08 C 06M Ill Me)K,Qy C060V!, KOTOpble xapaKTep1113YIOT COCTOf!HIIIe |
| 27 | no_unit_or_qualitative | name+value | 6IIIOL\eHo3a. CreneHb AIIIC61!1o3a Ol...\eHit18aercf! TOilbKO np111 06M>105. |
| 28 | no_unit_or_qualitative | name+value | ,[J,nfl y,Qo6cr8a rpaKT08KIII pe3yilbTaros** s ra6Illt1l...\e IIICnOilb308aHa l...\BeT08af! MapK111p08Ka. B |
| 29 | no_unit_or_qualitative | name+value | 3a8111CIIIMOCTIII OT 1113Mepf!eMOrO napaMerpa MapKepbl 0603Ha4aiOT Cile,QytOLl.\ee: |
| 30 | no_unit_or_qualitative | name+value | KoHTPOilbHble noKa3aTemll (reHOMHa~ ,QHK 4enoseKa, o6Ll.\a~ 6aKTepL1alibHa~ Macca, TpaH3L1TOpHa~ ML1Kpo¢nopa): |
| 31 | unrecognized |  | D • |
| 32 | no_unit_or_qualitative | name+value | COOT8eTCT8111e KOIIITeOitlf!M |
| 33 | no_unit_or_qualitative | name+value | He COOT8eTCT8111e KOIIITeOitlf!M |
| 34 | unrecognized |  | HopMoQ>nopa: |
| 35 | unrecognized |  | D • |
| 36 | unrecognized |  | D • |
| 37 | no_unit_or_qualitative | name+value | COOT8eTCT8lt1e KOlt1TeOlt1f!M HOOMbl |
| 38 | no_unit_or_qualitative | name+value | VMeOeHHOe OTKilOHeHitle OT KDlt1TeOlt1e8 HOOMbl |
| 39 | no_unit_or_qualitative | name+value | 8bi08)KeHHOe OTKilOHeHitle OT KOIIITeOIIIe8 HOOMbl |
| 40 | no_unit_or_qualitative | name+value | COOT8eTCT8lt1e KOlt1TeOlt1f!M HOOMbl |
| 41 | no_unit_or_qualitative | name+value | VMeOeHHOe OTKilOHeHitle OT KDlt1TeOIIIe8 HOOMbl |
| 42 | no_unit_or_qualitative | name+value | 8bi08)KeHHOe OTKilOHeHitle OT KOIIITeOIIIe8 HOOMbl |
| 43 | name_only | name | naToreHbl: |
| 44 | unrecognized |  | D • |
| 45 | no_unit_or_qualitative | name+value | He 8blf!8IleHO |
| 46 | no_unit_or_qualitative | name+value | 06H80V)KeHO |
| 47 | no_unit_or_qualitative | name+value | Pe3yilbTaT, 8 KOTOpOM 3Ha4eHIIIe noKa3aTeils:l 05M Hlti)Ke noporo80ro 3H84eHIIIs:l C04eTaeTCs:l CO |
| 48 | no_unit_or_qualitative | name+value | 3Ha4eHlt1eM noKa3arens:J «reHOMHas:J ,D,HK 4eil08eKa» 8b1We noporo8oro 3Ha4eHlt1s:l, rpaKryercs:J KaK |
| 49 | no_unit_or_qualitative | name+value | «HopMOL\eH03» 6e3 Ol..\eHKIII crpyKrypbi Mlt1Kpo6~t~oMa. B 3TOM cny4ae l..\8eT08as:J MapK~t~posKa 8 6naHKe |

### Diagnosis

- **Primary bottleneck:** `B` (Table block segmentation)
- Rationale: Lines exist but lab_table_detector did not recognize a table — segmentation is the bottleneck.

- Note: appears_table_like_no_separators=True — multi-column layout without | or \t separators.
- Note: rejection breakdown: {'empty_or_noise': 1, 'unrecognized': 8, 'no_unit_or_qualitative': 41, 'name_only': 2, 'malformed_lab_row': 3}

### Raw text preview

```
[Page 1]
LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH 
MeTOAOM nl..tp B pe>KHMe peailbHOrO BpeMeHH 
AHApocpnop®,AHApOcpnop®CKpHH 
OnHcaHHe 6naHKa peJynbTaToB 
V1CCile,Q08aHIIIe np080,QIIITCfl MeTO,QOM nOI1111Mepa3HOVl 1...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII. 
C 1...\eilbiO :3Tlt10IlOrlll4eCKOVl ,QIIIarHOCTIIIKIII 111HcpeKL\lt10HH0-80Cnaili!ITeilbHbiX 3a60ile8aHIIIVl M04enOI1080Vl 
c~t~creMbl MY)K4lt1H 8 aHanlt131t1pyeMOM 6~t~oMarep~t~ane O,QHOspeMeHHO BblnOil
```

## Test Results 6.pdf

### OCR/Layout

- Engine: `existing_pdf_pipeline`
- Quality score: `0.809`
- Quality band: `good`
- Route decision: `digital_clean_text`
- Quality warnings: `['layout_or_table_heavy', 'low_medical_token_density']`
- Raw text length: `2305`

### Classification

- Final status: `review`
- Entity count: `1`
- Confidence: `0.45`
- Reason codes: `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities`
- Legacy OCR codes: ``

### Lab Normalization

- Lab table detected: `False`
- Coverage ratio: `0.0`
- Coverage band: `none`
- Reason codes: `non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required`
- Would upgrade review_ocr_quality→review: `False`

### Forensic Parse

- Candidate lines: `106`
- Parsed rows: `0`
- Rejected lines: `106`
- Rejection breakdown: `{'empty_or_noise': 2, 'no_unit_or_qualitative': 29, 'value_only': 7, 'name_only': 17, 'noisy_high_punctuation': 44, 'unrecognized': 7}`

#### Signals

- value-only lines present: `True`
- name-only lines present: `True`
- range-only lines present: `False`
- appears table-like without separators: `True`
- text too fragmented (>50% short lines): `True`
- text too sparse (<3 lines): `False`
- short_line_ratio: `0.67`

#### Top candidate lines (up to 50)

```
[Page 1]
<D.I1.Q. na:qHeHTa
<l>e,n:eHLJ:eHKO CTenaH J1eoHH,ll;OBHLJ:
N2 KapTbr
720862
l ,rf
. / )'
·.:.c
.,_, ~ t:71,. J«
MeC5II.J.
)J,HarH03 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT.
~----~------~--
l1PEIIAPAT1I I L(ATA
CseLJ:H ,IUIKJio¢eHaK 100Mr ,f
}--
no 1 cseqe Ha HOLJ:h
. J
cf-
J .
(1-
Cse'1H BHTarrpocT ¢opn rro
1 CBCLJ:e Ha H0'1b
MeTpOHH,n:a3on no 500Mr 2
. .J
;;..
")._
,i-
p B .ueHb rrocne e,n:br
J1eso¢noKCai.J.HH no 500Mr
.).-
,:;
.-f
1 _IJ_ 3a 1 '1ac .uo o6e.ua
<l>nyKOHa3011 no 150 Mr 1 p
.:+-
,-./-
.f.
\+-
nOCJie 33BTp3K3
J1HHeKC cpOpT3 no lKancyne
.J·
• .!.-
,.l
1.-
1 p .UO 33RTpaKa
TaMcyJio3HH no 0.4Mr 1p
.1-
.1-
,l-
,J..
```

#### Rejected lines

| # | reason | partial | line |
| ---: | --- | --- | --- |
| 0 | empty_or_noise |  | [Page 1] |
| 1 | no_unit_or_qualitative | name+value | <D.I1.Q. na:qHeHTa |
| 2 | no_unit_or_qualitative | name+value | <l>e,n:eHLJ:eHKO CTenaH J1eoHH,ll;OBHLJ: |
| 3 | no_unit_or_qualitative | name+value | N2 KapTbr |
| 4 | value_only | value | 720862 |
| 5 | name_only | name | l ,rf |
| 6 | noisy_high_punctuation |  | . / )' |
| 7 | noisy_high_punctuation |  | ·.:.c |
| 8 | noisy_high_punctuation |  | .,_, ~ t:71,. J« |
| 9 | no_unit_or_qualitative | name+value | MeC5II.J. |
| 10 | no_unit_or_qualitative | name+value | )J,HarH03 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT. |
| 11 | noisy_high_punctuation |  | ~----~------~-- |
| 12 | no_unit_or_qualitative | name+value | l1PEIIAPAT1I I L(ATA |
| 13 | no_unit_or_qualitative | name+value | CseLJ:H ,IUIKJio¢eHaK 100Mr ,f |
| 14 | noisy_high_punctuation |  | }-- |
| 15 | no_unit_or_qualitative | name+value | no 1 cseqe Ha HOLJ:h |
| 16 | name_only | name | . J |
| 17 | name_only | name | cf- |
| 18 | name_only | name | J . |
| 19 | noisy_high_punctuation |  | (1- |
| 20 | no_unit_or_qualitative | name+value | Cse'1H BHTarrpocT ¢opn rro |
| 21 | no_unit_or_qualitative | name+value | 1 CBCLJ:e Ha H0'1b |
| 22 | no_unit_or_qualitative | name+value | MeTpOHH,n:a3on no 500Mr 2 |
| 23 | name_only | name | . .J |
| 24 | empty_or_noise |  | ;;.. |
| 25 | noisy_high_punctuation |  | ")._ |
| 26 | noisy_high_punctuation |  | ,i- |
| 27 | unrecognized |  | p B .ueHb rrocne e,n:br |
| 28 | no_unit_or_qualitative | name+value | J1eso¢noKCai.J.HH no 500Mr |
| 29 | noisy_high_punctuation |  | .).- |
| 30 | noisy_high_punctuation |  | ,:; |
| 31 | noisy_high_punctuation |  | .-f |
| 32 | no_unit_or_qualitative | name+value | 1 _IJ_ 3a 1 '1ac .uo o6e.ua |
| 33 | no_unit_or_qualitative | name+value | <l>nyKOHa3011 no 150 Mr 1 p |
| 34 | noisy_high_punctuation |  | .:+- |
| 35 | noisy_high_punctuation |  | ,-./- |
| 36 | name_only | name | .f. |
| 37 | noisy_high_punctuation |  | \+- |
| 38 | no_unit_or_qualitative | name+value | nOCJie 33BTp3K3 |
| 39 | no_unit_or_qualitative | name+value | J1HHeKC cpOpT3 no lKancyne |
| 40 | noisy_high_punctuation |  | .J· |
| 41 | noisy_high_punctuation |  | • .!.- |
| 42 | noisy_high_punctuation |  | ,.l |
| 43 | value_only | value | 1.- |
| 44 | no_unit_or_qualitative | name+value | 1 p .UO 33RTpaKa |
| 45 | no_unit_or_qualitative | name+value | TaMcyJio3HH no 0.4Mr 1p |
| 46 | value_only | value | .1- |
| 47 | value_only | value | .1- |
| 48 | noisy_high_punctuation |  | ,l- |
| 49 | noisy_high_punctuation |  | ,J.. |

### Diagnosis

- **Primary bottleneck:** `A` (OCR/Layout candidate generation)
- Rationale: Selected text is sparse or fragmented — OCR/Layout is dropping lines before parsing.

- Note: appears_table_like_no_separators=True — multi-column layout without | or \t separators.
- Note: rejection breakdown: {'empty_or_noise': 2, 'no_unit_or_qualitative': 29, 'value_only': 7, 'name_only': 17, 'noisy_high_punctuation': 44, 'unrecognized': 7}

### Raw text preview

```
[Page 1]
<D.I1.Q. na:qHeHTa 
<l>e,n:eHLJ:eHKO CTenaH J1eoHH,ll;OBHLJ: 
N2 KapTbr 
720862 
., 
l ,rf 
. / )' 
l 
[ 
·.:.c
.,_, ~ t:71,. J« 
MeC5II.J. 
)J,HarH03 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT. 
~----~------~--
l1PEIIAPAT1I I L(ATA 
1 
2 
3 
4 
5 
6 
7 
8 
9 
lO 
11 
12 
l3 
14 
15 
16 
17 
CseLJ:H ,IUIKJio¢eHaK 100Mr ,f 
}--
J 
no 1 cseqe Ha HOLJ:h 
. J 
.j 
cf-
J . 
-! 
~-
(1-
Cse'1H BHTarrpocT ¢opn rro 
_j 
,+ 
.+ 
rf 
/ 
t! 
.J 
1 CBCLJ:e Ha H0'1b 
I 
MeTpOHH,n:a3on no 500Mr 2 
.
```


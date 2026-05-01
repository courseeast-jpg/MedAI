# Phase 35 Holdout Review Audit

- Source report: `G:\Codex\2026-04-22-connect-github\reports\holdout_validation\latest_holdout_validation.json`
- Reviewed files: `6`

## Remaining Issue Breakdown

- ocr_noise_remaining: `6`
- low_coverage_after_normalization: `0`
- missing_domain_rules: `0`
- correctly_flagged_review: `0`

## Results 1.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.45
- entity count: 2
- entities: ['Nitrite', 'Ketones']
- why reviewed: ['confidence_below_threshold', 'low_coverage', 'low_extractor_weight']
- normalization applied: False
- extraction method: tesseract fallback
- text length: 1007

- normalized preview:


- original preview:
[Page 1] ee ee * . . e é * ’ ~~ 2 . ’ - . ’ ’ - . 4 - ; . BIO MED LABORATOR MIKROBIOLOGJIK ‘ é EMRI — STEPAN’ FEDECHENKO ADRESA UKRAINE DATA 9 11 2022 : ° eae ’ # Se URINE KOMPLET # KARAKTERISTIKAT FIZIKE : ¥ . ’ * NGJYRA.......%eeeeee0ee006eE VERDHE ’ a * ’ ae PAMIAS . See cas ce en's» o'siee e's E

## Results 2.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.45
- entity count: 1
- entities: ['Urine Culture']
- why reviewed: ['confidence_below_threshold', 'low_entity_count', 'low_coverage', 'low_extractor_weight']
- normalization applied: False
- extraction method: tesseract fallback
- text length: 939

- normalized preview:


- original preview:
[Page 1] s . . a Y, ’ ‘ : BIO MED LABORATOR MIKROBIOLOGIIK i . \ [PERSON] — STEPAN FEDECHENKO ADRESA UKRAINE DATA @9 [DATE] * > ‘ a : 3 = ’ * i UROKULTURE Y * ie = 1 NGA KULTIVIMI’ PER 24h REZULTOI NEGATIV i * * . 2 ae ANTIBIOGRAME « CIPROFLOXACINI......... LEVOFLOXACINI........... ' * . * ’ MPMPE T

## Test Results 2.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.7
- entity count: 6
- entities: ['CKOH', 'Macca', 'He BbiHBneHO', 'BneHO', 'IIr', 'HBneHO']
- why reviewed: ['low_coverage']
- normalization applied: True
- extraction method: unknown
- text length: 2689

- normalized preview:
[Page 1] N2 ):[PERSON] npo6>IpK>1 <!>.1.1.0. naI..1>IeHra non BoJpacr OpraHL'IJai..\L'IH Bpa4 np>IMe4aH>Ie loIccneAOBaHl'!e Ml'IKpocjlnopbl yporeHl'!TanbHoro TpaKTa MY>[ID_REMOVED] MeTO,QOM ni.-IP B pe>KHMe peanbHOrO BpeMeHl'l [DATE], [DATE] 101462359601 <lleAeHYeHKO CrenaH MYlf<CKOH 51 HaJBaHHe HCC

- original preview:
[Page 1] N2 ):[PERSON] npo6>IpK>1 <!>.1.1.0. naI..1>IeHra non BoJpacr OpraHL'IJai..\L'IH Bpa4 np>IMe4aH>Ie loIccneAOBaHl'!e Ml'IKpocjlnopbl yporeHl'!TanbHoro TpaKTa MY>[ID_REMOVED] MeTO,QOM ni.-IP B pe>KHMe peanbHOrO BpeMeHl'l [DATE], [DATE] 101462359601 <lleAeHYeHKO CrenaH MYlf<CKOH 51 HaJBaHHe HCC

## Test Results 3.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.63
- entity count: 4
- entities: ['QcrasneHbl', 'QCTaBileHbl', 'QHaKO', 'Ha fltiCTOrpaMMe']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- normalization applied: True
- extraction method: unknown
- text length: 3723

- normalized preview:
[Page 1] LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH MeTOAOM [URL_REMOVED] B pe>KHMe peailbHOrO [ID_REMOVED] AHApocpnop®,AHApOcpnop®CKpHH OnHcaHHe 6naHKa peJynbTaToB VICCile,QO8aHIIIe [LOCATION],[PERSON]...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII. C 1...\eilbiO :[DAT

- original preview:
[Page 1] LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH MeTOAOM [URL_REMOVED] B pe>KHMe peailbHOrO [ID_REMOVED] AHApocpnop®,AHApOcpnop®CKpHH OnHcaHHe 6naHKa peJynbTaToB VICCile,QO8aHIIIe [LOCATION],[PERSON]...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII. C 1...\eilbiO :[DAT

## Test Results 5.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.63
- entity count: 3
- entities: ['LlnY', 'JleHIIIHrpaACKIII', 'Klebsiella oxytoca']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- normalization applied: True
- extraction method: unknown
- text length: 2312

- normalized preview:
[Page 1] fi;KDL [URL_REMOVED] I'!HTIIflll'!poBIIHHfUII OCI"EIMI!. Mfl~ c:>epT$1,UoIpc:la!IHil [PERSON] OOOTMTCI'Il ~8 'Tplil• ISc:oaH.,._, rOCT P ltIOCHlOO'1·2OHi, r OOT P IIK:O , 51 89-~015. roor P 11100 14001· [DATE], rOOT F' I'IOO/M3K 27001·2006, OHSo\S 1 OCKJI ::<DO7. SAl SA BCXXJ:2014 ~:; ro~a

- original preview:
[Page 1] fi;KDL [URL_REMOVED] I'!HTIIflll'!poBIIHHfUII OCI"EIMI!. Mfl~ c:>epT$1,UoIpc:la!IHil [PERSON] OOOTMTCI'Il ~8 'Tplil• ISc:oaH.,._, rOCT P ltIOCHlOO'1·2OHi, r OOT P IIK:O , 51 89-~015. roor P 11100 14001· [DATE], rOOT F' I'IOO/M3K 27001·2006, OHSo\S 1 OCKJI ::<DO7. SAl SA BCXXJ:2014 ~:; ro~a

## Test Results 6.pdf

- remaining issue: ocr_noise_remaining
- confidence: 0.63
- entity count: 3
- entities: ['ATA', 'HTb', 'He MCBee']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- normalization applied: True
- extraction method: unknown
- text length: 2325

- normalized preview:
[Page 1] <D.II.Q. na:qHeHTa <l>e,n:eHLJ:eHKO CTenaH JIeoHH,ll;OBHLJ: N2 KapTbr [DATE] ., l ,rf . / )' l [ ·.:.c .,_, ~ t:71,. J« MeC5II.J. )J,HarHO3 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT. ~ ~ ~-- lIPEIIAPATII I L(ATA 1 2 3 4 5 6 7 8 9 lO 11 12 l3 14 15 16 17 CseLJ:H ,IUIKJio¢eHaK IOOMr ,f }--

- original preview:
[Page 1] <D.II.Q. na:qHeHTa <l>e,n:eHLJ:eHKO CTenaH JIeoHH,ll;OBHLJ: N2 KapTbr [DATE] ., l ,rf . / )' l [ ·.:.c .,_, ~ t:71,. J« MeC5II.J. )J,HarHO3 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT. ~ ~ ~-- lIPEIIAPATII I L(ATA 1 2 3 4 5 6 7 8 9 lO 11 12 l3 14 15 16 17 CseLJ:H ,IUIKJio¢eHaK IOOMr ,f }--

# MedAI Review Audit

- Source report: `G:\Codex\2026-04-22-connect-github\reports\phase38_ocr_layout\latest_phase38_batch_validation.json`
- Reviewed files: `6`

## Review Fix Breakdown

- no_entities: `0`
- low_entity_count: `1`
- low_confidence: `1`
- low_coverage: `4`
- low_diversity: `0`
- extractor_issue: `0`

## Results 1.pdf

- confidence: 0.45
- entities: ['Nitrite', 'Ketones']
- why reviewed: ['confidence_below_threshold', 'low_coverage', 'low_extractor_weight']
- recommended fix: low_confidence

- preview:
[Page 1] ee ee * . . e é * ’ ~~ 2 . ’ - . ’ ’ - . 4 - ; . BIO MED LABORATOR MIKROBIOLOGJIK ‘ é EMRI — STEPAN’ FEDECHENKO ADRESA UKRAINE DATA 9 11 2022 : ° eae ’ # Se URINE KOMPLET # KARAKTERISTIKAT FIZIKE : ¥ . ’ * NGJYRA.......%eeeeee0ee006eE VERDHE ’ a * ’ ae PAMIAS . See cas ce en's» o'siee e's E

## Results 2.pdf

- confidence: 0.45
- entities: ['Urine Culture']
- why reviewed: ['confidence_below_threshold', 'low_entity_count', 'low_coverage', 'low_extractor_weight']
- recommended fix: low_entity_count

- preview:
[Page 1] s . . a Y, ’ ‘ : BIO MED LABORATOR MIKROBIOLOGIIK i . \ J EMRI — STEPAN FEDECHENKO ADRESA UKRAINE DATA @9 11 2022 * > ‘ a : 3 = ’ * i UROKULTURE Y * ie = 1 NGA KULTIVIMI’ PER 24h REZULTOI NEGATIV i * * . 2 ae ANTIBIOGRAME « CIPROFLOXACINI......... LEVOFLOXACINI........... ' * . * ’ MPMPE TR

## Test Results 2.pdf

- confidence: 0.7
- entities: ['CKOH', 'Macca', 'He BbiHBneHO', 'BneHO', 'IIr', 'HBneHO']
- why reviewed: ['low_coverage']
- recommended fix: low_coverage

- preview:
[Page 1] N2 ):lara HoMep npo6>1pK>1 <!>.1.1.0. na1..1>1eHra non BoJpacr OpraHL'IJai..\L'IH Bpa4 np>1Me4aH>1e lo1ccneAOBaHl'!e Ml'IKpocjlnopbl yporeHl'!TanbHoro TpaKTa MY>KI.fl'IH MeTO,QOM ni.-IP B pe>KHMe peanbHOrO BpeMeHl'l 16 AeKa6pb 2021, 12:35:49 101462359601 <lleAeHYeHKO CrenaH MYlf<CKOH 51 HaJ

## Test Results 3.pdf

- confidence: 0.63
- entities: ['QcrasneHbl', 'QCTaBileHbl', 'QHaKO', 'Ha fltiCTOrpaMMe']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- recommended fix: low_coverage

- preview:
[Page 1] LllccneAoBaHHe MHKpocpnopbl yporeHHTailbHoro TpaKTa MY>KliHH MeTOAOM nl..tp B pe>KHMe peailbHOrO BpeMeHH AHApocpnop®,AHApOcpnop®CKpHH OnHcaHHe 6naHKa peJynbTaToB V1CCile,Q08aHIIIe np080,QIIITCfl MeTO,QOM nOI1111Mepa3HOVl 1...\enHOVl peaKl...\111111 8 pe)KII!Me peailbHOrO 8peMeHIII. C 1...\e

## Test Results 5.pdf

- confidence: 0.63
- entities: ['LlnY', 'JleHIIIHrpaACKIII', 'Klebsiella oxytoca']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- recommended fix: low_coverage

- preview:
[Page 1] fi;KDL www.kdl.ru I'!HTIIflll'!poBIIHHfUII OCI"EIMI!. Mfl~ c:>epT$1,Uo1pc:la!IHil Hil OOOTMTC1'1l ~8 'Tplil• ISc:oaH.,._, rOCT P lt10CHl00'1·20Hi, r OOT P IIK:O , 51 89-~015. roor P 11100 14001· 2019, rOOT F' I'IOO/M3K 27001·2006, OHSo\S 1 OCKJ1 ::<D07. SAl SA BCXXJ:2014 ~:; ro~a ~,. CepBHC

## Test Results 6.pdf

- confidence: 0.63
- entities: ['ATA', 'HTb', 'He MCBee']
- why reviewed: ['confidence_below_threshold', 'low_coverage']
- recommended fix: low_coverage

- preview:
[Page 1] <D.I1.Q. na:qHeHTa <l>e,n:eHLJ:eHKO CTenaH J1eoHH,ll;OBHLJ: N2 KapTbr 720862 ., l ,rf . / )' l [ ·.:.c .,_, ~ t:71,. J« MeC5II.J. )J,HarH03 XpoHHLJ:eCKHH 6aKTepHanbHhiH ypeTponpoCTaTHT. ~----~------~-- l1PEIIAPAT1I I L(ATA 1 2 3 4 5 6 7 8 9 lO 11 12 l3 14 15 16 17 CseLJ:H ,IUIKJio¢eHaK 100M

# Phase57 Full Corpus Inventory Audit

- Timestamp: `2026-05-03T19:47:36.219903+00:00`
- Total discovered: `546`
- Total supported: `536`
- Total processed: `536`
- Unsupported: `10`
- Accepted: `91`
- Review: `445`
- Review OCR quality: `11`
- Empty: `357`
- Errors: `10`
- External API used: `False`
- Local-only forced: `True`
- Raw PHI logged in public reports: `False`
- Report PDF/image artifacts tracked: `False`
- Conclusion: `PASS_SAFETY_WEAK_AUTOMATION`

## Distributions

- By extension: `{".docx": 3, ".jpg": 8, ".mp3": 2, ".msg": 1, ".ogg": 1, ".pdf": 524, ".rtf": 3, ".tif": 3, ".txt": 1}`
- By file type: `{"image": 11, "pdf": 524, "txt": 1, "unsupported": 10}`
- By status: `{"accepted": 91, "error": 10, "review": 434, "review_ocr_quality": 11}`
- By document class/type: `{"digital_pdf": 115, "image_ocr": 11, "mixed_pdf": 387, "scanned_pdf": 22, "text_file": 1, "unknown": 10}`
- By OCR status: `{"empty": 2, "good": 503, "poor_ocr": 5, "unknown": 11, "usable_with_review": 25}`
- By language hint: `{"unknown": 546}`

## Reason-Code Clusters

- `table_structure_loss`: `513`
- `extraction_low_confidence`: `440`
- `safety_gate_low_confidence`: `440`
- `low_text_density`: `421`
- `extraction_low_coverage`: `402`
- `extraction_sparse_entities`: `389`
- `lab_report_detected`: `219`
- `accepted_clean_input`: `92`
- `possible_multi_document_pdf`: `22`
- `unsupported_format`: `10`
- `classifier_legacy_ocr_flag`: `6`
- `document_type_prescription_not_lab`: `5`
- `non_lab_document_skipped_lab_normalization`: `5`
- `poor_input_ocr`: `5`
- `legacy_normalized_low_coverage`: `4`
- `language_aware_ocr_required`: `4`
- `pdf_portfolio_or_embedded_files_detected`: `3`
- `legacy_suspicious_ocr_context`: `2`
- `empty_or_near_empty_text`: `2`
- `lab_table_recovered`: `1`

## Files Requiring Attention

- `corpus_file_000004`
- `corpus_file_000010`
- `corpus_file_000013`
- `corpus_file_000015`
- `corpus_file_000024`
- `corpus_file_000025`
- `corpus_file_000030`
- `corpus_file_000037`
- `corpus_file_000042`
- `corpus_file_000046`
- `corpus_file_000050`
- `corpus_file_000057`
- `corpus_file_000064`
- `corpus_file_000065`
- `corpus_file_000066`
- `corpus_file_000073`
- `corpus_file_000077`
- `corpus_file_000081`
- `corpus_file_000082`
- `corpus_file_000083`
- `corpus_file_000084`
- `corpus_file_000085`
- `corpus_file_000086`
- `corpus_file_000087`
- `corpus_file_000088`
- `corpus_file_000090`
- `corpus_file_000091`
- `corpus_file_000092`
- `corpus_file_000093`
- `corpus_file_000094`
- `corpus_file_000095`
- `corpus_file_000096`
- `corpus_file_000097`
- `corpus_file_000098`
- `corpus_file_000099`
- `corpus_file_000100`
- `corpus_file_000101`
- `corpus_file_000102`
- `corpus_file_000103`
- `corpus_file_000104`
- `corpus_file_000105`
- `corpus_file_000106`
- `corpus_file_000107`
- `corpus_file_000108`
- `corpus_file_000109`
- `corpus_file_000110`
- `corpus_file_000111`
- `corpus_file_000112`
- `corpus_file_000113`
- `corpus_file_000114`
- `corpus_file_000115`
- `corpus_file_000116`
- `corpus_file_000117`
- `corpus_file_000118`
- `corpus_file_000119`
- `corpus_file_000120`
- `corpus_file_000121`
- `corpus_file_000122`
- `corpus_file_000123`
- `corpus_file_000124`
- `corpus_file_000125`
- `corpus_file_000126`
- `corpus_file_000127`
- `corpus_file_000128`
- `corpus_file_000129`
- `corpus_file_000130`
- `corpus_file_000131`
- `corpus_file_000132`
- `corpus_file_000133`
- `corpus_file_000134`
- `corpus_file_000135`
- `corpus_file_000136`
- `corpus_file_000137`
- `corpus_file_000138`
- `corpus_file_000139`
- `corpus_file_000140`
- `corpus_file_000141`
- `corpus_file_000142`
- `corpus_file_000143`
- `corpus_file_000144`
- `corpus_file_000145`
- `corpus_file_000146`
- `corpus_file_000147`
- `corpus_file_000148`
- `corpus_file_000149`
- `corpus_file_000150`
- `corpus_file_000151`
- `corpus_file_000152`
- `corpus_file_000153`
- `corpus_file_000154`
- `corpus_file_000155`
- `corpus_file_000156`
- `corpus_file_000157`
- `corpus_file_000158`
- `corpus_file_000159`
- `corpus_file_000160`
- `corpus_file_000161`
- `corpus_file_000162`
- `corpus_file_000163`
- `corpus_file_000164`
- `corpus_file_000165`
- `corpus_file_000166`
- `corpus_file_000167`
- `corpus_file_000168`
- `corpus_file_000169`
- `corpus_file_000170`
- `corpus_file_000172`
- `corpus_file_000173`
- `corpus_file_000174`
- `corpus_file_000175`
- `corpus_file_000176`
- `corpus_file_000177`
- `corpus_file_000178`
- `corpus_file_000179`
- `corpus_file_000180`
- `corpus_file_000181`
- `corpus_file_000182`
- `corpus_file_000183`
- `corpus_file_000184`
- `corpus_file_000185`
- `corpus_file_000187`
- `corpus_file_000188`
- `corpus_file_000189`
- `corpus_file_000190`
- `corpus_file_000191`
- `corpus_file_000192`
- `corpus_file_000193`
- `corpus_file_000194`
- `corpus_file_000195`
- `corpus_file_000196`
- `corpus_file_000197`
- `corpus_file_000198`
- `corpus_file_000199`
- `corpus_file_000200`
- `corpus_file_000201`
- `corpus_file_000202`
- `corpus_file_000203`
- `corpus_file_000204`
- `corpus_file_000205`
- `corpus_file_000206`
- `corpus_file_000207`
- `corpus_file_000208`
- `corpus_file_000209`
- `corpus_file_000210`
- `corpus_file_000211`
- `corpus_file_000212`
- `corpus_file_000213`
- `corpus_file_000214`
- `corpus_file_000215`
- `corpus_file_000216`
- `corpus_file_000217`
- `corpus_file_000218`
- `corpus_file_000219`
- `corpus_file_000220`
- `corpus_file_000221`
- `corpus_file_000222`
- `corpus_file_000223`
- `corpus_file_000224`
- `corpus_file_000225`
- `corpus_file_000226`
- `corpus_file_000227`
- `corpus_file_000228`
- `corpus_file_000229`
- `corpus_file_000230`
- `corpus_file_000231`
- `corpus_file_000232`
- `corpus_file_000233`
- `corpus_file_000234`
- `corpus_file_000235`
- `corpus_file_000236`
- `corpus_file_000237`
- `corpus_file_000238`
- `corpus_file_000239`
- `corpus_file_000240`
- `corpus_file_000241`
- `corpus_file_000242`
- `corpus_file_000243`
- `corpus_file_000244`
- `corpus_file_000245`
- `corpus_file_000246`
- `corpus_file_000247`
- `corpus_file_000248`
- `corpus_file_000249`
- `corpus_file_000250`
- `corpus_file_000251`
- `corpus_file_000252`
- `corpus_file_000253`
- `corpus_file_000254`
- `corpus_file_000255`
- `corpus_file_000256`
- `corpus_file_000257`
- `corpus_file_000258`
- `corpus_file_000259`
- `corpus_file_000260`
- `corpus_file_000261`
- `corpus_file_000262`
- `corpus_file_000263`
- `corpus_file_000265`
- `corpus_file_000266`
- `corpus_file_000268`
- `corpus_file_000269`
- `corpus_file_000270`
- `corpus_file_000271`
- `corpus_file_000272`
- `corpus_file_000273`
- `corpus_file_000274`
- `corpus_file_000276`
- `corpus_file_000277`
- `corpus_file_000278`
- `corpus_file_000279`
- `corpus_file_000280`
- `corpus_file_000281`
- `corpus_file_000282`
- `corpus_file_000283`
- `corpus_file_000284`
- `corpus_file_000285`
- `corpus_file_000286`
- `corpus_file_000287`
- `corpus_file_000288`
- `corpus_file_000289`
- `corpus_file_000290`
- `corpus_file_000291`
- `corpus_file_000292`
- `corpus_file_000293`
- `corpus_file_000294`
- `corpus_file_000295`
- `corpus_file_000296`
- `corpus_file_000297`
- `corpus_file_000298`
- `corpus_file_000299`
- `corpus_file_000300`
- `corpus_file_000301`
- `corpus_file_000302`
- `corpus_file_000303`
- `corpus_file_000304`
- `corpus_file_000305`
- `corpus_file_000306`
- `corpus_file_000307`
- `corpus_file_000308`
- `corpus_file_000309`
- `corpus_file_000310`
- `corpus_file_000311`
- `corpus_file_000312`
- `corpus_file_000313`
- `corpus_file_000314`
- `corpus_file_000315`
- `corpus_file_000316`
- `corpus_file_000317`
- `corpus_file_000318`
- `corpus_file_000319`
- `corpus_file_000320`
- `corpus_file_000321`
- `corpus_file_000322`
- `corpus_file_000323`
- `corpus_file_000324`
- `corpus_file_000325`
- `corpus_file_000326`
- `corpus_file_000327`
- `corpus_file_000328`
- `corpus_file_000329`
- `corpus_file_000330`
- `corpus_file_000331`
- `corpus_file_000332`
- `corpus_file_000333`
- `corpus_file_000334`
- `corpus_file_000335`
- `corpus_file_000336`
- `corpus_file_000337`
- `corpus_file_000338`
- `corpus_file_000339`
- `corpus_file_000340`
- `corpus_file_000341`
- `corpus_file_000342`
- `corpus_file_000343`
- `corpus_file_000344`
- `corpus_file_000345`
- `corpus_file_000346`
- `corpus_file_000347`
- `corpus_file_000348`
- `corpus_file_000349`
- `corpus_file_000350`
- `corpus_file_000351`
- `corpus_file_000352`
- `corpus_file_000353`
- `corpus_file_000354`
- `corpus_file_000355`
- `corpus_file_000356`
- `corpus_file_000357`
- `corpus_file_000358`
- `corpus_file_000359`
- `corpus_file_000360`
- `corpus_file_000361`
- `corpus_file_000362`
- `corpus_file_000363`
- `corpus_file_000364`
- `corpus_file_000365`
- `corpus_file_000366`
- `corpus_file_000367`
- `corpus_file_000368`
- `corpus_file_000369`
- `corpus_file_000370`
- `corpus_file_000371`
- `corpus_file_000372`
- `corpus_file_000373`
- `corpus_file_000374`
- `corpus_file_000375`
- `corpus_file_000376`
- `corpus_file_000377`
- `corpus_file_000378`
- `corpus_file_000379`
- `corpus_file_000380`
- `corpus_file_000381`
- `corpus_file_000382`
- `corpus_file_000383`
- `corpus_file_000384`
- `corpus_file_000385`
- `corpus_file_000386`
- `corpus_file_000387`
- `corpus_file_000388`
- `corpus_file_000389`
- `corpus_file_000390`
- `corpus_file_000391`
- `corpus_file_000392`
- `corpus_file_000393`
- `corpus_file_000394`
- `corpus_file_000395`
- `corpus_file_000396`
- `corpus_file_000397`
- `corpus_file_000398`
- `corpus_file_000399`
- `corpus_file_000400`
- `corpus_file_000401`
- `corpus_file_000402`
- `corpus_file_000403`
- `corpus_file_000404`
- `corpus_file_000405`
- `corpus_file_000406`
- `corpus_file_000407`
- `corpus_file_000408`
- `corpus_file_000409`
- `corpus_file_000410`
- `corpus_file_000411`
- `corpus_file_000412`
- `corpus_file_000413`
- `corpus_file_000414`
- `corpus_file_000415`
- `corpus_file_000416`
- `corpus_file_000417`
- `corpus_file_000418`
- `corpus_file_000419`
- `corpus_file_000420`
- `corpus_file_000422`
- `corpus_file_000423`
- `corpus_file_000424`
- `corpus_file_000426`
- `corpus_file_000427`
- `corpus_file_000429`
- `corpus_file_000430`
- `corpus_file_000432`
- `corpus_file_000433`
- `corpus_file_000434`
- `corpus_file_000435`
- `corpus_file_000436`
- `corpus_file_000437`
- `corpus_file_000438`
- `corpus_file_000439`
- `corpus_file_000440`
- `corpus_file_000441`
- `corpus_file_000442`
- `corpus_file_000443`
- `corpus_file_000445`
- `corpus_file_000446`
- `corpus_file_000447`
- `corpus_file_000448`
- `corpus_file_000449`
- `corpus_file_000451`
- `corpus_file_000453`
- `corpus_file_000454`
- `corpus_file_000455`
- `corpus_file_000457`
- `corpus_file_000458`
- `corpus_file_000460`
- `corpus_file_000462`
- `corpus_file_000463`
- `corpus_file_000464`
- `corpus_file_000465`
- `corpus_file_000467`
- `corpus_file_000469`
- `corpus_file_000470`
- `corpus_file_000471`
- `corpus_file_000472`
- `corpus_file_000473`
- `corpus_file_000474`
- `corpus_file_000475`
- `corpus_file_000476`
- `corpus_file_000477`
- `corpus_file_000478`
- `corpus_file_000479`
- `corpus_file_000480`
- `corpus_file_000482`
- `corpus_file_000483`
- `corpus_file_000484`
- `corpus_file_000485`
- `corpus_file_000486`
- `corpus_file_000487`
- `corpus_file_000488`
- `corpus_file_000489`
- `corpus_file_000490`
- `corpus_file_000491`
- `corpus_file_000492`
- `corpus_file_000493`
- `corpus_file_000494`
- `corpus_file_000495`
- `corpus_file_000497`
- `corpus_file_000498`
- `corpus_file_000499`
- `corpus_file_000500`
- `corpus_file_000501`
- `corpus_file_000502`
- `corpus_file_000504`
- `corpus_file_000505`
- `corpus_file_000506`
- `corpus_file_000507`
- `corpus_file_000508`
- `corpus_file_000510`
- `corpus_file_000511`
- `corpus_file_000512`
- `corpus_file_000513`
- `corpus_file_000514`
- `corpus_file_000515`
- `corpus_file_000516`
- `corpus_file_000517`
- `corpus_file_000518`
- `corpus_file_000519`
- `corpus_file_000520`
- `corpus_file_000521`
- `corpus_file_000522`
- `corpus_file_000523`
- `corpus_file_000524`
- `corpus_file_000525`
- `corpus_file_000526`
- `corpus_file_000527`
- `corpus_file_000528`
- `corpus_file_000530`
- `corpus_file_000532`
- `corpus_file_000534`
- `corpus_file_000535`
- `corpus_file_000537`
- `corpus_file_000538`
- `corpus_file_000539`
- `corpus_file_000540`
- `corpus_file_000541`
- `corpus_file_000542`
- `corpus_file_000545`
- `corpus_file_000546`

## Per-File Inventory

| Safe File ID | Filename Hash | Content Hash | Extension | Type | Size | Pages | Status | Extractor | Confidence | OCR Status | PDF Flags | Reason Codes | Error Category |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- |
| `corpus_file_000001` | `4a1e2fd7f4b85fe6` | `69eb6c1ed8ef8419` | `.pdf` | `pdf` | `58715` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000002` | `f7444ed0c3057c22` | `590eac76e4d26fbe` | `.pdf` | `pdf` | `58784` | `1` | `accepted` | `spacy` | `0.767` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000003` | `5fb2211837c71ab9` | `86f1b02e55eb2e2d` | `.pdf` | `pdf` | `56629` | `1` | `accepted` | `spacy` | `0.767` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000004` | `b0a4aa1be9e59b70` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000005` | `969c54d13e81cfef` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000006` | `c33287e01598248d` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000007` | `e4834530a5a58633` | `abe44b5eb8f0f1b4` | `.pdf` | `pdf` | `60986` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000008` | `119e52e10fdab6b3` | `d481030ecdd73c6d` | `.pdf` | `pdf` | `59968` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000009` | `868d218d012facf5` | `4f4e36cf42d6b2ed` | `.pdf` | `pdf` | `61119` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000010` | `03990b1bfc9e8bd2` | `5239f4a4a84416e6` | `.pdf` | `pdf` | `57646` | `1` | `review_ocr_quality` | `spacy` | `0.63` | `good` | `` | `classifier_legacy_ocr_flag, extraction_low_confidence, extraction_low_coverage, lab_report_detected, legacy_normalized_low_coverage, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000011` | `53acb8b1cd0959fc` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000012` | `7a10596c4aee3c0c` | `0f8966cb6918def9` | `.pdf` | `pdf` | `75564` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000013` | `170f1028a0906822` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000014` | `5d9e672cb2adc041` | `040974af5b0bfbb1` | `.pdf` | `pdf` | `56900` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000015` | `5be6c948da845c0f` | `aeaa77e39da7263c` | `.pdf` | `pdf` | `6641343` | `5` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000016` | `a954c25737082263` | `6b68d3a62407b91d` | `.pdf` | `pdf` | `59447` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000017` | `039f4bd42135db2a` | `9ed164593968e245` | `.pdf` | `pdf` | `59763` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000018` | `584b9dd01d438dea` | `8fa99e10fc827a12` | `.pdf` | `pdf` | `59794` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000019` | `f960123d6a1b3e3e` | `966a2b3ed6b79ad3` | `.pdf` | `pdf` | `61184` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000020` | `dfb52a10347eea98` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000021` | `7a4d7db2d0d830f4` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000022` | `9fd455b605bd2d17` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000023` | `bc2e0f3d313641a5` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000024` | `045cc89d029097f6` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000025` | `f10976d20a200ed8` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000026` | `62797c7b1fdf73a8` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000027` | `bf573f98f7233b2d` | `811a1195baef8c27` | `.pdf` | `pdf` | `152003` | `3` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000028` | `2e11c28d03c5cf2e` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000029` | `dbfada3fdb032273` | `23279b1383047e7c` | `.pdf` | `pdf` | `105848` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000030` | `006ce0dd09ec5c9b` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000031` | `42a1991a732929d3` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000032` | `fe8f49989b3845fa` | `444ad95a0efe0c09` | `.pdf` | `pdf` | `60329` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000033` | `ca035d8de50eb7cb` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000034` | `2e30011a68a47f9c` | `23279b1383047e7c` | `.pdf` | `pdf` | `105848` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000035` | `1d8fa240538d4860` | `582b8ef4ccdad9b4` | `.pdf` | `pdf` | `78961` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000036` | `d1b5030e61ad7660` | `a96189bc5b9a7f04` | `.pdf` | `pdf` | `60697` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000037` | `472e8ee7801bc80b` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000038` | `381dda19fe1106e7` | `0ab73166e2fa1933` | `.pdf` | `pdf` | `59871` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000039` | `41385556ffdd5e1c` | `2f0e3740ef798bd5` | `.pdf` | `pdf` | `58768` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000040` | `52871300d2957132` | `9b9106a9da7d198a` | `.pdf` | `pdf` | `59349` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000041` | `ad3adae5b0d5dff0` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000042` | `24ee6a18dc0dc3e2` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000043` | `5d90285a2776b7f2` | `582b8ef4ccdad9b4` | `.pdf` | `pdf` | `78961` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000044` | `c70dff17253011b1` | `c66286b69d47505e` | `.pdf` | `pdf` | `132755` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000045` | `5d90285a2776b7f2` | `7064dd64b786c514` | `.pdf` | `pdf` | `124000` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000046` | `fb9fa7ff05996d62` | `ad7279062eca4200` | `.pdf` | `pdf` | `79709` | `2` | `review` | `spacy` | `0.63` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, lab_table_recovered, lab_table_recovered_review_only, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000047` | `87d5bf5b3eea0d91` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000048` | `ebce19a3c5ad9e33` | `3ff1ec262b09f09a` | `.pdf` | `pdf` | `150555` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000049` | `3f69d6dad161634a` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000050` | `419a1c37bc31ec82` | `beb7448705415af6` | `.pdf` | `pdf` | `87115` | `3` | `review_ocr_quality` | `spacy` | `0.7` | `good` | `` | `classifier_legacy_ocr_flag, extraction_low_coverage, lab_report_detected, legacy_normalized_low_coverage, table_structure_loss` | `` |
| `corpus_file_000051` | `5d90285a2776b7f2` | `0f8966cb6918def9` | `.pdf` | `pdf` | `75564` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000052` | `8bf0b6ad44fd1ba4` | `50eb7f4db2a290b2` | `.pdf` | `pdf` | `80831` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, table_structure_loss` | `` |
| `corpus_file_000053` | `1a3e5fd549d08052` | `56bbd1fd55ae2579` | `.pdf` | `pdf` | `62216` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000054` | `c172d202ba3427c2` | `aa94430b1fc1971a` | `.pdf` | `pdf` | `59839` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000055` | `5f7ed886a6fb2e45` | `3aa36d51d1233a04` | `.pdf` | `pdf` | `58850` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000056` | `f2a9be9f4ad40c6f` | `e6490759550a363c` | `.pdf` | `pdf` | `59713` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000057` | `e2949d49cebecdcd` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000058` | `176d83274c09b546` | `e6490759550a363c` | `.pdf` | `pdf` | `59713` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000059` | `a032d85a5417efa6` | `50eb7f4db2a290b2` | `.pdf` | `pdf` | `80831` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, table_structure_loss` | `` |
| `corpus_file_000060` | `8108e597a40adc92` | `66d749f3e701cf81` | `.pdf` | `pdf` | `56840` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000061` | `384f37b02e843a13` | `83d6ef94b203e422` | `.pdf` | `pdf` | `60210` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000062` | `4de3305662d29fda` | `34b8762619f52bd6` | `.pdf` | `pdf` | `60071` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000063` | `6b7442f430385a01` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000064` | `61e16f23a91dcf03` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000065` | `695e5c483795ec9b` | `cc7ecf09c56583e9` | `.pdf` | `pdf` | `57129` | `1` | `review` | `spacy` | `0.562` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000066` | `ba7bc09679361ae8` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000067` | `4e3ce28a0608697d` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000068` | `b0358d927922238c` | `2fe49209e693c831` | `.pdf` | `pdf` | `60326` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000069` | `1475ab5bbfa71510` | `86c587134ba5d41b` | `.pdf` | `pdf` | `59941` | `1` | `accepted` | `spacy` | `0.693` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000070` | `f93a061d1dec06cd` | `811a1195baef8c27` | `.pdf` | `pdf` | `152003` | `3` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000071` | `62472a86c6574b59` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000072` | `f9cbaaa971388199` | `0f8966cb6918def9` | `.pdf` | `pdf` | `75564` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000073` | `b9eae8ba1263d695` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000074` | `68aac35750b90c43` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000075` | `4146696f841ffede` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000076` | `7ff3aa8eae325a4f` | `1afc7c2891756c0f` | `.pdf` | `pdf` | `78960` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000077` | `77fdc926376467fa` | `beb7448705415af6` | `.pdf` | `pdf` | `87115` | `3` | `review_ocr_quality` | `spacy` | `0.7` | `good` | `` | `classifier_legacy_ocr_flag, extraction_low_coverage, lab_report_detected, legacy_normalized_low_coverage, table_structure_loss` | `` |
| `corpus_file_000078` | `d26c1319dd661db6` | `3ff1ec262b09f09a` | `.pdf` | `pdf` | `150555` | `2` | `accepted` | `spacy` | `0.762` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000079` | `dd80504fa6c4b75e` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000080` | `03b892dd444f1d15` | `582b8ef4ccdad9b4` | `.pdf` | `pdf` | `78961` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000081` | `4811ab3293e923db` | `95fd5d97d94d1663` | `.pdf` | `pdf` | `10357673` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `embedded_files` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, pdf_portfolio_or_embedded_files_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000082` | `d2ced014a21fe4e1` | `78ddc482a581c4bb` | `.pdf` | `pdf` | `1881874` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000083` | `42d59612cd1f9ff4` | `fec0fb5bbceae470` | `.pdf` | `pdf` | `459322` | `1` | `review` | `spacy` | `0.487` | `good` | `` | `extraction_low_confidence, extraction_sparse_entities, safety_gate_low_confidence` | `` |
| `corpus_file_000084` | `13ba687ec8098fe1` | `6fc5d5c45ec472e2` | `.pdf` | `pdf` | `1283192` | `1` | `review` | `spacy` | `0.693` | `good` | `possible_multi_document` | `accepted_clean_input, possible_multi_document_pdf` | `` |
| `corpus_file_000085` | `24ec449a790af92a` | `e26b391cf5608236` | `.pdf` | `pdf` | `1172393` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000086` | `6a8362f68462d5c7` | `0e8687957bb04804` | `.pdf` | `pdf` | `1648422` | `1` | `review` | `spacy` | `0.425` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000087` | `09b4004a37477756` | `35258d7a4f47713a` | `.pdf` | `pdf` | `131440` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000088` | `53a37d9cfa634599` | `eead589cda4065ee` | `.pdf` | `pdf` | `26368` | `2` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000089` | `943781f8398f4c8d` | `bc08717b43ec7815` | `.pdf` | `pdf` | `274812` | `2` | `accepted` | `spacy` | `0.767` | `usable_with_review` | `` | `accepted_clean_input, table_structure_loss` | `` |
| `corpus_file_000090` | `3aa5fdacab67e4f2` | `2d8561c2c0b7bc99` | `.pdf` | `pdf` | `18997` | `1` | `review_ocr_quality` | `spacy` | `0.5` | `good` | `` | `classifier_legacy_ocr_flag, extraction_low_confidence, extraction_low_coverage, legacy_normalized_low_coverage, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000091` | `4b2c4ef068926d41` | `f6d2fcecb98bb6d3` | `.pdf` | `pdf` | `815765` | `2` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `document_type_prescription_not_lab, extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000092` | `7dd2fbed2ceb5608` | `e50ab8a1183d6222` | `.pdf` | `pdf` | `21040` | `4` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000093` | `0e5ab31b94445bb4` | `7a171bc226eede06` | `.pdf` | `pdf` | `1344287` | `4` | `review` | `phi3` | `0.064` | `good` | `` | `extraction_low_confidence, safety_gate_low_confidence` | `` |
| `corpus_file_000094` | `80b26e65849c2762` | `d5e3ca84b7803098` | `.pdf` | `pdf` | `42317` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000095` | `4ec50fa8d8d056ec` | `76e9c1b6cd6b2f58` | `.pdf` | `pdf` | `423608` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000096` | `543c0c4583d5c02a` | `6ba12ba6b228eae9` | `.pdf` | `pdf` | `470675` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000097` | `d2e413cf81246207` | `a4e638c06bdc86d0` | `.pdf` | `pdf` | `695101` | `2` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000098` | `d3691aa90d56941b` | `7a25044d8d5f4fe9` | `.pdf` | `pdf` | `379227` | `1` | `review_ocr_quality` | `rules_based` | `0.45` | `poor_ocr` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, poor_input_ocr, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000099` | `fcfc31720db92b38` | `8033eb3f8d1cb114` | `.pdf` | `pdf` | `259825` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000100` | `2c05137b0e500325` | `5be41b0e65110f9c` | `.pdf` | `pdf` | `1482096` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000101` | `ded2daff7dc96724` | `d61cbbcb701ea5ad` | `.pdf` | `pdf` | `716417` | `3` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000102` | `7a2d878c9eb4d94f` | `92862155dcae3465` | `.pdf` | `pdf` | `28747034` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `embedded_files` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, pdf_portfolio_or_embedded_files_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000103` | `19320f1cf64afcbb` | `ad584b56510ccef7` | `.pdf` | `pdf` | `3314294` | `4` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000104` | `bae87726253d53ed` | `f1751d2bad294a5a` | `.pdf` | `pdf` | `4646890` | `5` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000105` | `7549cdb95aebc3fa` | `063b4c000b1c71de` | `.pdf` | `pdf` | `4594862` | `4` | `review` | `phi3` | `0.024` | `good` | `possible_multi_document` | `extraction_low_confidence, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000106` | `ce6c89647130e000` | `a46eaf8e424d6b7a` | `.pdf` | `pdf` | `127678` | `2` | `review` | `phi3` | `0.0` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000107` | `59395242f522ac21` | `f8a430d04f57db3b` | `.tif` | `image` | `207749` | `` | `review` | `phi3` | `0.042` | `good` | `` | `low_text_density, extraction_low_confidence, safety_gate_low_confidence` | `` |
| `corpus_file_000108` | `37623779d164524b` | `87c1e6986d7b5cbf` | `.pdf` | `pdf` | `3226679` | `3` | `review` | `phi3` | `0.033` | `good` | `` | `extraction_low_confidence, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000109` | `a6940a8a8f8cb787` | `e6f3f90d528203b4` | `.tif` | `image` | `46625` | `` | `review` | `phi3` | `0.046` | `good` | `` | `low_text_density, extraction_low_confidence, safety_gate_low_confidence` | `` |
| `corpus_file_000110` | `3d12368621824c38` | `8d9d7b7ceb9d6e81` | `.pdf` | `pdf` | `121339` | `2` | `review` | `phi3` | `0.0` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000111` | `6f328973fdff38a0` | `ff60b03b9c3a8b12` | `.pdf` | `pdf` | `7578912` | `6` | `review` | `rules_based` | `0.45` | `usable_with_review` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000112` | `c50ccded45b40b29` | `090bf7800bb9a516` | `.pdf` | `pdf` | `783268` | `2` | `review` | `phi3` | `0.204` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence` | `` |
| `corpus_file_000113` | `a192b48270c2de5e` | `69b6a22257be9a0d` | `.pdf` | `pdf` | `948680` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000114` | `616b0f124ca02945` | `b48cc37f0c73ec2a` | `.pdf` | `pdf` | `974417` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000115` | `7792628b883b2221` | `99b661e0731040b9` | `.pdf` | `pdf` | `7965471` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000116` | `df051b3d12f4de32` | `752faf0fe48d0f4a` | `.pdf` | `pdf` | `2780723` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `document_type_prescription_not_lab, extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, language_aware_ocr_required, non_lab_document_skipped_lab_normalization, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000117` | `948a4a99aec25a0a` | `43d62bc8989f5b9a` | `.pdf` | `pdf` | `672151` | `4` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000118` | `51dd528d1c9ed01a` | `cf082bb5427571dd` | `.docx` | `unsupported` | `14489` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000119` | `40b79df9cd2e52ca` | `c489fca1d4968c7c` | `.pdf` | `pdf` | `248524` | `5` | `review` | `phi3` | `0.056` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000120` | `0daec4f37f2274e1` | `66b875b6fc62403c` | `.pdf` | `pdf` | `1452842` | `1` | `review` | `phi3` | `0.0` | `usable_with_review` | `` | `extraction_low_confidence, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000121` | `948a4a99aec25a0a` | `f8a4a37e5bc22b44` | `.pdf` | `pdf` | `697139` | `6` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000122` | `40b79df9cd2e52ca` | `8e8225a37b34e671` | `.pdf` | `pdf` | `286245` | `6` | `review` | `phi3` | `0.051` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000123` | `4e26fff85335d9d5` | `d2bc9b68e387d4d2` | `.pdf` | `pdf` | `388310` | `4` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000124` | `2973cc187de1d335` | `0abced4bc9008cf2` | `.pdf` | `pdf` | `4855028` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000125` | `0e9d8e9ec9fd4763` | `8819c49d0d717d39` | `.pdf` | `pdf` | `464239` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000126` | `8e88f18a6e394388` | `16b4a3b448d21fe4` | `.pdf` | `pdf` | `105956` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000127` | `d9468ed66b236f25` | `a1250d0fece4e24b` | `.pdf` | `pdf` | `2602834` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000128` | `a554f10ab62a6691` | `7792cae8d516b17b` | `.pdf` | `pdf` | `43187` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000129` | `2e70f0912e64ac88` | `c3e2ccae2daa9b7e` | `.pdf` | `pdf` | `417429` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000130` | `678f43aaf3b405d7` | `da50d1882a3f851a` | `.pdf` | `pdf` | `147423` | `4` | `review` | `phi3` | `0.248` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000131` | `965740ef4fe0827a` | `0202db4b9d517b84` | `.pdf` | `pdf` | `403917` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000132` | `b09b1c4969cc2435` | `b44becb800b25ad0` | `.pdf` | `pdf` | `142740` | `4` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000133` | `99eca8d72c511408` | `eec153e533f57d69` | `.pdf` | `pdf` | `406054` | `4` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000134` | `5fcd606461ea7cb4` | `8917df2f42e4c10c` | `.pdf` | `pdf` | `163754` | `3` | `review` | `phi3` | `0.078` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000135` | `06806807ae1c805f` | `c4347b958db906b3` | `.pdf` | `pdf` | `283385` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000136` | `948a4a99aec25a0a` | `0c93b28e4a144106` | `.pdf` | `pdf` | `731795` | `8` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000137` | `40b79df9cd2e52ca` | `bde878fe78d0d2d2` | `.pdf` | `pdf` | `413859` | `9` | `review` | `phi3` | `0.09` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000138` | `948a4a99aec25a0a` | `cd51d1ca24ef9f79` | `.pdf` | `pdf` | `591085` | `6` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000139` | `5fcd606461ea7cb4` | `96fc17d9d468a1e5` | `.pdf` | `pdf` | `310894` | `7` | `review` | `phi3` | `0.073` | `good` | `possible_multi_document` | `extraction_low_confidence, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000140` | `2d09c8e08333f62c` | `60156823f3bb0b96` | `.docx` | `unsupported` | `2036329` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000141` | `601da66394de0059` | `37f4b5fad8fafcdf` | `.pdf` | `pdf` | `2718373` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000142` | `eab50b0885b9ae88` | `f79d38648c767102` | `.pdf` | `pdf` | `1138180` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000143` | `8f01f6e0a9f97fff` | `d9edfbce1add7056` | `.jpg` | `image` | `201004` | `` | `review` | `rules_based` | `0.45` | `good` | `` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required` | `` |
| `corpus_file_000144` | `052ba1cc6ceb0b1c` | `40e100ea8804cef5` | `.jpg` | `image` | `151490` | `` | `review` | `rules_based` | `0.45` | `good` | `` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, non_lab_document_skipped_lab_normalization, document_type_prescription_not_lab, language_aware_ocr_required` | `` |
| `corpus_file_000145` | `b6cb6d26c3b0ee5e` | `85c7e59c97622694` | `.jpg` | `image` | `96333` | `` | `review_ocr_quality` | `rules_based` | `0.45` | `poor_ocr` | `` | `poor_input_ocr, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_suspicious_ocr_context` | `` |
| `corpus_file_000146` | `0f993772bcf116df` | `ad1c85e4cc0aae1d` | `.pdf` | `pdf` | `1217383` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000147` | `ed264363d9a7c12a` | `ae4fc18d8a71f191` | `.pdf` | `pdf` | `10199161` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `embedded_files` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, pdf_portfolio_or_embedded_files_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000148` | `90f9c0d57fa7305c` | `d54a2443d1b100a4` | `.pdf` | `pdf` | `325063` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000149` | `253514bf6222b0f7` | `1d4975f5a91cbbdc` | `.pdf` | `pdf` | `403419` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000150` | `7fdef3013ebe1a36` | `f0fc05847132b8a4` | `.pdf` | `pdf` | `268307` | `1` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000151` | `525d1207a88ddd91` | `e323d4c665985aef` | `.pdf` | `pdf` | `126610` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000152` | `93457bfdd45f0aa1` | `d0c0b696936614f2` | `.pdf` | `pdf` | `1163476` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000153` | `4f53f7f17f5d2c9f` | `31691fdce27ff72e` | `.rtf` | `unsupported` | `43845` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000154` | `5ecffe05c774c200` | `4baee25c5093bbc6` | `.mp3` | `unsupported` | `27238817` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000155` | `4a8c737641146134` | `520c7c496a20b8ce` | `.rtf` | `unsupported` | `44865` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000156` | `9458e54263b281e8` | `11fa1fffc429bdba` | `.rtf` | `unsupported` | `48033` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000157` | `931fefa2721a1be4` | `978a6a4de7db4934` | `.pdf` | `pdf` | `342459` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000158` | `64984530051dc299` | `92123ddbac604553` | `.pdf` | `pdf` | `344598` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000159` | `085e5586e54b1b3c` | `c43532b73b5f501f` | `.msg` | `unsupported` | `735232` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000160` | `51cd80ac25510a7a` | `3c79ed7e7f7fa7c4` | `.pdf` | `pdf` | `619355` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000161` | `f27ea3d009766dfa` | `ac7768e935cb0b13` | `.txt` | `txt` | `11` | `` | `review` | `claude` | `0.5` | `unknown` | `` | `table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `corpus_file_000162` | `bf1d53c7844fe77d` | `e8d31a40ab5ab73b` | `.docx` | `unsupported` | `15050` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000163` | `f58797a08a2e0a65` | `e551909c8abf77a4` | `.ogg` | `unsupported` | `85829` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000164` | `2370452c5a4a865d` | `270e96e128281c6a` | `.jpg` | `image` | `5069524` | `` | `review` | `rules_based` | `0.45` | `good` | `` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `corpus_file_000165` | `6780404a9fa5b657` | `59353ccb6855f40b` | `.pdf` | `pdf` | `3065016` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000166` | `441a4fc475b1f097` | `270e96e128281c6a` | `.jpg` | `image` | `5069524` | `` | `review` | `rules_based` | `0.45` | `good` | `` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, lab_report_detected` | `` |
| `corpus_file_000167` | `0a1d7c6304477b52` | `0c0fcbcce0e4f65b` | `.pdf` | `pdf` | `331609` | `1` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000168` | `416bfff98193eca9` | `136d4790a586a28e` | `.pdf` | `pdf` | `201872` | `1` | `review_ocr_quality` | `rules_based` | `0.45` | `poor_ocr` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, poor_input_ocr, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000169` | `973978bafc8d72ff` | `1091b8dba00e22a3` | `.pdf` | `pdf` | `4260266` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000170` | `94fb93c65568851d` | `014a1ffb0418b061` | `.pdf` | `pdf` | `501360` | `2` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000171` | `6cac65197b79da8c` | `ace21aefbe83ac5b` | `.pdf` | `pdf` | `890250` | `1` | `accepted` | `spacy` | `0.9` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000172` | `7ce31d001adf6272` | `ef6bbbf74f21a3bf` | `.pdf` | `pdf` | `731794` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000173` | `58fc59e09efaccd8` | `493d1d5be50439ee` | `.pdf` | `pdf` | `1178933` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000174` | `9c4c74f7a0396659` | `20b7efefeb542b52` | `.pdf` | `pdf` | `738144` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000175` | `927ae81cbefc0ab9` | `59bc6b340d7a4d65` | `.pdf` | `pdf` | `2878998` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000176` | `73201a9ae16e54be` | `076ade41661d9056` | `.pdf` | `pdf` | `1720880` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000177` | `91829b1417485ba9` | `5567b51fbf1a7d11` | `.pdf` | `pdf` | `94595` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000178` | `f2728f27fedbfa7e` | `2499b6c69839711e` | `.pdf` | `pdf` | `126163` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000179` | `cbcc2c4ed76f7c05` | `769c7b0e4bab086b` | `.pdf` | `pdf` | `101014` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000180` | `6d1141620b91df54` | `cb2f338baf80134a` | `.pdf` | `pdf` | `62887` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000181` | `8bf0b6ad44fd1ba4` | `50eb7f4db2a290b2` | `.pdf` | `pdf` | `80831` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000182` | `f2b5b814f67de7eb` | `beb7448705415af6` | `.pdf` | `pdf` | `87115` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000183` | `5d21385fbec9a500` | `ad7279062eca4200` | `.pdf` | `pdf` | `79709` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000184` | `c4a491bc06a43cda` | `a9daf124b06e6a1c` | `.pdf` | `pdf` | `77963` | `1` | `review` | `phi3` | `0.57` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000185` | `baa498b08c569880` | `1acc986c8464b747` | `.pdf` | `pdf` | `72163` | `1` | `review` | `phi3` | `0.485` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000186` | `866e80b1fdcac3fb` | `5e0f5d84eaa6224a` | `.pdf` | `pdf` | `66424` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000187` | `67d04689cb7dd0dc` | `86e9d533711e0d86` | `.pdf` | `pdf` | `104501` | `2` | `review` | `phi3` | `0.569` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000188` | `35be113d09157786` | `dd8610f91868f259` | `.pdf` | `pdf` | `1743058` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000189` | `b031be5dc2d80473` | `e1008e8176d19c26` | `.pdf` | `pdf` | `57779` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000190` | `1f399f8833d0f885` | `822f7ac78e09bbf7` | `.pdf` | `pdf` | `64098` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000191` | `91c7d2ef8b1475dc` | `82e4d819b46afb90` | `.pdf` | `pdf` | `59689` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000192` | `af5da1f833321b52` | `03fa6f2e8f17cd8a` | `.pdf` | `pdf` | `61616` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000193` | `1a101a5824e64ce1` | `2fe49209e693c831` | `.pdf` | `pdf` | `60326` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000194` | `7378aca805e16c9f` | `abe44b5eb8f0f1b4` | `.pdf` | `pdf` | `60986` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000195` | `16e8b499bf4cbb9c` | `bb9d6e9adadd95c6` | `.pdf` | `pdf` | `61194` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000196` | `c0590e0f8541a1ea` | `076cf0499fc6e610` | `.pdf` | `pdf` | `60673` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000197` | `540ad2456df851fb` | `6a2cb054e5e7316a` | `.pdf` | `pdf` | `62138` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000198` | `c48e7f3bb80a750f` | `e53045d369bd3fd1` | `.pdf` | `pdf` | `61440` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000199` | `99075c6ee8e8e0d6` | `5fa82087810999c8` | `.pdf` | `pdf` | `60440` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000200` | `c8ee23a68ee3650a` | `b45fd8993c4ac11e` | `.pdf` | `pdf` | `61443` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000201` | `7db75f3d9d312ec2` | `7dc0bc82ebd16655` | `.pdf` | `pdf` | `59957` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000202` | `9ca120c47ea8c7e6` | `fa051dd4cc98fdd2` | `.pdf` | `pdf` | `54838` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000203` | `370114891326be6d` | `28a81161e4685eb7` | `.pdf` | `pdf` | `59716` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000204` | `32e30f55ce9a82dc` | `b4327f784cb48558` | `.pdf` | `pdf` | `59074` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000205` | `0c1c1cc95ad23e26` | `faca248cccb974c3` | `.pdf` | `pdf` | `59848` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000206` | `cd844900d0c709a8` | `6b68d3a62407b91d` | `.pdf` | `pdf` | `59447` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000207` | `e1cb4d8837c3549f` | `b48c297236d96ff5` | `.pdf` | `pdf` | `57222` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000208` | `b8b675e96ab134f0` | `c94242e4b562211d` | `.pdf` | `pdf` | `58621` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000209` | `0129037b9dfaae16` | `ef05cc0dbe0fc44a` | `.pdf` | `pdf` | `63235` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000210` | `8e0c2a7cab4e75df` | `c232011580b119e2` | `.pdf` | `pdf` | `63523` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000211` | `91d0eb4fa788b545` | `29f4d4485ea438db` | `.pdf` | `pdf` | `61096` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000212` | `6e0e28a6f8466c53` | `abbf3f187f8a85a1` | `.pdf` | `pdf` | `55092` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000213` | `6803804cf53bf944` | `bf5044fcc54eea93` | `.pdf` | `pdf` | `55711` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000214` | `107db796f08c0746` | `936e0cf51060682c` | `.pdf` | `pdf` | `58603` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000215` | `d370910a0e0f949f` | `eacdcd54657a26a9` | `.pdf` | `pdf` | `58232` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000216` | `c2741acc00eff0ec` | `cd9bae23b9e247e8` | `.pdf` | `pdf` | `55405` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000217` | `a5563398df283114` | `f9156a6b4e56d0e6` | `.pdf` | `pdf` | `54501` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000218` | `9fc19977fee47613` | `baa046cd12251f6e` | `.pdf` | `pdf` | `60754` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000219` | `c7b9be02378c3c64` | `12293add74ea4072` | `.pdf` | `pdf` | `58920` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000220` | `3f2f3edebafe8914` | `1f752f1686e4a5cc` | `.pdf` | `pdf` | `58031` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000221` | `5606e428095db55b` | `a4f2b51e0caee07b` | `.pdf` | `pdf` | `58223` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000222` | `75b448ab5dc681df` | `9b5589ca092300a6` | `.pdf` | `pdf` | `57383` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000223` | `3836a6ed37e247b8` | `a96189bc5b9a7f04` | `.pdf` | `pdf` | `60697` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000224` | `dc7794c13bd899bd` | `0d050b6605f92d8d` | `.pdf` | `pdf` | `63425` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000225` | `93559712e8357f6b` | `aa94430b1fc1971a` | `.pdf` | `pdf` | `59839` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000226` | `e5a2cfd189a86f97` | `56bbd1fd55ae2579` | `.pdf` | `pdf` | `62216` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000227` | `192d406a1da3c4b6` | `0ab73166e2fa1933` | `.pdf` | `pdf` | `59871` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000228` | `9d1aec5a31fc9293` | `7a4fa110f67f6ac0` | `.pdf` | `pdf` | `62658` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000229` | `e6992b8f7673dec1` | `92b584f27e077768` | `.pdf` | `pdf` | `56108` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000230` | `d40c8ac65d9d7b4d` | `158874ee28ed8178` | `.pdf` | `pdf` | `57235` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000231` | `2b1a029a26d6c294` | `af093d4a2fda8d74` | `.pdf` | `pdf` | `55765` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000232` | `4ad7a7837a8a2dec` | `ebd532d404f19c79` | `.pdf` | `pdf` | `60759` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000233` | `bfa1898edbb6c6f6` | `368016fb40cb808e` | `.pdf` | `pdf` | `61655` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000234` | `237d61ba3b7b702c` | `beb5091182587db6` | `.pdf` | `pdf` | `61350` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000235` | `e6c461bd63abb748` | `2f7a2dc6c259f209` | `.pdf` | `pdf` | `56638` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000236` | `52d5a0005f3e9424` | `8fef3554ebf9b8c9` | `.pdf` | `pdf` | `59682` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000237` | `52bfe358ca0c3dac` | `34c2aa6a20b0b999` | `.pdf` | `pdf` | `58581` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000238` | `71aec6b2fdb6fcee` | `0887600f2e272d0b` | `.pdf` | `pdf` | `59727` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000239` | `bd48d3fba7b6aa43` | `12f0442e8c92f29a` | `.pdf` | `pdf` | `57010` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000240` | `c9e143b81c08438a` | `3bff80944d6727f9` | `.pdf` | `pdf` | `56654` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000241` | `a8351c7786d45778` | `3074a60a11d65ee4` | `.pdf` | `pdf` | `57604` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000242` | `13ff614c9a650c83` | `3ae1fb9d6962f1ec` | `.pdf` | `pdf` | `55922` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000243` | `7c6e4ea4e5500c84` | `34c181bdc2b1cf8d` | `.pdf` | `pdf` | `59778` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000244` | `f4ad618bd77222ca` | `f0b02f826e3278bc` | `.pdf` | `pdf` | `58675` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000245` | `cc2586a6e5643e17` | `f193047fe1f4c340` | `.pdf` | `pdf` | `59320` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000246` | `9d1e1b735aaf7ac2` | `9066d391ed5de9f6` | `.pdf` | `pdf` | `56278` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000247` | `ed181a97b33955eb` | `0659b797949ba239` | `.pdf` | `pdf` | `56246` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000248` | `22e8c31d4bb03035` | `83d6ef94b203e422` | `.pdf` | `pdf` | `60210` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000249` | `5941e76c2b3b21e4` | `c6910c00a22c4274` | `.pdf` | `pdf` | `56160` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000250` | `e515d86435378e90` | `9f40ee989170e0da` | `.pdf` | `pdf` | `57037` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000251` | `ea11ba49c299e717` | `b7f5f6c067f85ec2` | `.pdf` | `pdf` | `58591` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000252` | `1bdbf23e18418db4` | `9ecc1a5b783c9217` | `.pdf` | `pdf` | `59319` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000253` | `54eb8d2e26095719` | `5649dbd5f3ac5c0b` | `.pdf` | `pdf` | `59010` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000254` | `d942429de847b123` | `5ffaf06101ee9eb2` | `.pdf` | `pdf` | `56925` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000255` | `c3625a17fc477f99` | `b06bca94222be6a6` | `.pdf` | `pdf` | `59447` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000256` | `46ec5bb3061684d6` | `8b0c84a34013bee7` | `.pdf` | `pdf` | `59403` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000257` | `ab1d326067f36168` | `c425888b16f6cb6e` | `.pdf` | `pdf` | `58681` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000258` | `0d79e52cf4ac4f2b` | `01cfe439944c6ffa` | `.pdf` | `pdf` | `58737` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000259` | `87d7e2990333e34d` | `ad82bed6e2ba8fe7` | `.pdf` | `pdf` | `58623` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000260` | `f8278ded24cb4b4c` | `0990307041cbbec7` | `.pdf` | `pdf` | `58187` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000261` | `0ee2572a3e8b53d2` | `8bbb2a1b4605676c` | `.pdf` | `pdf` | `57315` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000262` | `76aa7dea374070cb` | `95a1248ee68ff596` | `.pdf` | `pdf` | `56050` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000263` | `93a0c11df0171800` | `bf1d6370f35ba551` | `.pdf` | `pdf` | `59023` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000264` | `ea4f038c5ea75bf9` | `0a3bccf784414a62` | `.pdf` | `pdf` | `57181` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000265` | `df9d6ef0ff92776a` | `b300c3c752f31d0d` | `.pdf` | `pdf` | `63312` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000266` | `69c6e9610f8b93e1` | `7d9862d1ddebda52` | `.pdf` | `pdf` | `60829` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000267` | `b5fc9aca11fbbe16` | `9d57aca36190e3ef` | `.pdf` | `pdf` | `57913` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000268` | `0db1f3cf567c37cc` | `81ca70573592906f` | `.pdf` | `pdf` | `60661` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000269` | `2aaaf8103ef99b5f` | `cc7ecf09c56583e9` | `.pdf` | `pdf` | `57129` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000270` | `85a7f53f1699bd81` | `dbd6df41db5e96a9` | `.pdf` | `pdf` | `59113` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000271` | `ad6fe2f81997ecae` | `153b9d9268eee534` | `.pdf` | `pdf` | `63736` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000272` | `12df7133343f5245` | `040974af5b0bfbb1` | `.pdf` | `pdf` | `56900` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000273` | `6a0f57f4e8c2053b` | `e6fc8c9191081dd3` | `.pdf` | `pdf` | `57341` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000274` | `22b5f2ab53dc2d3a` | `aeaa77e39da7263c` | `.pdf` | `pdf` | `6641343` | `5` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000275` | `85123cd18adc5c7c` | `c23467a149a8acd6` | `.pdf` | `pdf` | `998245` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, table_structure_loss` | `` |
| `corpus_file_000276` | `c963dda4bf8fe747` | `eb899fed8fffedc1` | `.pdf` | `pdf` | `1250727` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000277` | `0dc87617b61dcb8d` | `f88fb3429e8c5e51` | `.pdf` | `pdf` | `1358467` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000278` | `6e101da56a5f0424` | `fe71f8b8cb8e95c2` | `.pdf` | `pdf` | `1331699` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000279` | `4ad7a7837a8a2dec` | `29a7607d084ff4f7` | `.pdf` | `pdf` | `60097` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000280` | `104be1bc7feee56f` | `fa7d40d84829bf8f` | `.pdf` | `pdf` | `59872` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000281` | `b031be5dc2d80473` | `79218149771c8654` | `.pdf` | `pdf` | `58345` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000282` | `1f399f8833d0f885` | `408ae9a4763452f6` | `.pdf` | `pdf` | `63831` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000283` | `91c7d2ef8b1475dc` | `f28b91dc5e64d361` | `.pdf` | `pdf` | `59037` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000284` | `af5da1f833321b52` | `42cedd2bbbee8917` | `.pdf` | `pdf` | `60841` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000285` | `1a101a5824e64ce1` | `e3ddf440367e420b` | `.pdf` | `pdf` | `58040` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000286` | `259c3f8ac0e50592` | `a8c671ae7901bc2d` | `.pdf` | `pdf` | `58118` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000287` | `7378aca805e16c9f` | `d481030ecdd73c6d` | `.pdf` | `pdf` | `59968` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000288` | `16e8b499bf4cbb9c` | `c20fe2ea9e925689` | `.pdf` | `pdf` | `60657` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000289` | `c0590e0f8541a1ea` | `1f498b6e71462ea5` | `.pdf` | `pdf` | `60428` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000290` | `540ad2456df851fb` | `a9f609633e2d696a` | `.pdf` | `pdf` | `61690` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000291` | `c48e7f3bb80a750f` | `bda0345f51226856` | `.pdf` | `pdf` | `61220` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000292` | `99075c6ee8e8e0d6` | `5087d9229d29a0eb` | `.pdf` | `pdf` | `59855` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000293` | `c8ee23a68ee3650a` | `7e4a7f3f65c07a95` | `.pdf` | `pdf` | `61201` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000294` | `7db75f3d9d312ec2` | `dae4fc4cc4f9a2e5` | `.pdf` | `pdf` | `59203` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000295` | `9ca120c47ea8c7e6` | `ac428da0409b2cf0` | `.pdf` | `pdf` | `54593` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000296` | `370114891326be6d` | `c0424e8e4c677dd2` | `.pdf` | `pdf` | `59259` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000297` | `32e30f55ce9a82dc` | `945005cf203631f8` | `.pdf` | `pdf` | `59612` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000298` | `0c1c1cc95ad23e26` | `ce3957829fdcfa98` | `.pdf` | `pdf` | `58470` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000299` | `cd844900d0c709a8` | `9ed164593968e245` | `.pdf` | `pdf` | `59763` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000300` | `e1cb4d8837c3549f` | `fb89df621ec1cc59` | `.pdf` | `pdf` | `56495` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000301` | `b8b675e96ab134f0` | `0bfd1a04ca4b69e3` | `.pdf` | `pdf` | `59198` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000302` | `0129037b9dfaae16` | `ab83c0dd2bb4f2d9` | `.pdf` | `pdf` | `62217` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000303` | `8e0c2a7cab4e75df` | `a3aec3f279e019cc` | `.pdf` | `pdf` | `63509` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000304` | `91d0eb4fa788b545` | `9e47783255968dd7` | `.pdf` | `pdf` | `60863` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000305` | `6e0e28a6f8466c53` | `a55003ea39b58265` | `.pdf` | `pdf` | `54862` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000306` | `107db796f08c0746` | `2fbaaf81a3cb3fe1` | `.pdf` | `pdf` | `57901` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000307` | `d370910a0e0f949f` | `444ad95a0efe0c09` | `.pdf` | `pdf` | `60329` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000308` | `c2741acc00eff0ec` | `25e4f010983ab17f` | `.pdf` | `pdf` | `55631` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000309` | `a5563398df283114` | `3988804f1453a684` | `.pdf` | `pdf` | `54250` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000310` | `75b448ab5dc681df` | `0d92a073cae6622f` | `.pdf` | `pdf` | `57120` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000311` | `dc7794c13bd899bd` | `357edf7d3321deee` | `.pdf` | `pdf` | `63184` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000312` | `93559712e8357f6b` | `3aa36d51d1233a04` | `.pdf` | `pdf` | `58850` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000313` | `e5a2cfd189a86f97` | `e6490759550a363c` | `.pdf` | `pdf` | `59713` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000314` | `192d406a1da3c4b6` | `2f0e3740ef798bd5` | `.pdf` | `pdf` | `58768` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000315` | `8063b62acc853d3f` | `3dddd49fce7c7f33` | `.pdf` | `pdf` | `57133` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000316` | `9d1aec5a31fc9293` | `495fd3cef7a40adf` | `.pdf` | `pdf` | `61909` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000317` | `e6992b8f7673dec1` | `573add8ae4d5dc38` | `.pdf` | `pdf` | `55582` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000318` | `d40c8ac65d9d7b4d` | `c24b25d1c6c9e7d9` | `.pdf` | `pdf` | `56932` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000319` | `2b1a029a26d6c294` | `0d8707736461a01c` | `.pdf` | `pdf` | `55539` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000320` | `bfa1898edbb6c6f6` | `ac70a518b1458cd4` | `.pdf` | `pdf` | `61433` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000321` | `237d61ba3b7b702c` | `68a6a477fa425bb8` | `.pdf` | `pdf` | `60613` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000322` | `e6c461bd63abb748` | `610715eab59f72a0` | `.pdf` | `pdf` | `56552` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000323` | `52d5a0005f3e9424` | `5a79eaf6da4bb2a9` | `.pdf` | `pdf` | `58915` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000324` | `52bfe358ca0c3dac` | `18cb046383c85bd2` | `.pdf` | `pdf` | `58904` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000325` | `bd48d3fba7b6aa43` | `79dc8a77c4a09909` | `.pdf` | `pdf` | `57325` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000326` | `c9e143b81c08438a` | `2855688da192c11b` | `.pdf` | `pdf` | `56243` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000327` | `7c6e4ea4e5500c84` | `a5b0e8279bcb4b1c` | `.pdf` | `pdf` | `59850` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000328` | `ed181a97b33955eb` | `cdfd82de01210329` | `.pdf` | `pdf` | `55477` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000329` | `22e8c31d4bb03035` | `34b8762619f52bd6` | `.pdf` | `pdf` | `60071` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000330` | `2aaaf8103ef99b5f` | `8dd09a14933eb803` | `.pdf` | `pdf` | `55378` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000331` | `e4b123b450d2ca88` | `dc3ed15bb443f2d6` | `.pdf` | `pdf` | `61720` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000332` | `6a0f57f4e8c2053b` | `a091452c30400b5a` | `.pdf` | `pdf` | `57411` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000333` | `755c39b5e955f5b1` | `491b40d41f76f758` | `.pdf` | `pdf` | `1434302` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000334` | `f30bd04aba2b2456` | `222f0f65fd20663e` | `.pdf` | `pdf` | `1158932` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000335` | `35be113d09157786` | `718e01d9ac3d9a53` | `.pdf` | `pdf` | `1505699` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000336` | `22b5f2ab53dc2d3a` | `aeaa77e39da7263c` | `.pdf` | `pdf` | `6641343` | `5` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000337` | `35be113d09157786` | `01f33b19e3ef9032` | `.pdf` | `pdf` | `1568670` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000338` | `91c7d2ef8b1475dc` | `d1090614686decff` | `.pdf` | `pdf` | `58016` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000339` | `af5da1f833321b52` | `38b2b567f9f0ad8b` | `.pdf` | `pdf` | `60692` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000340` | `1a101a5824e64ce1` | `86c587134ba5d41b` | `.pdf` | `pdf` | `59941` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000341` | `7378aca805e16c9f` | `4f4e36cf42d6b2ed` | `.pdf` | `pdf` | `61119` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000342` | `e2642cbb89497989` | `7a4d1f955f4caeff` | `.pdf` | `pdf` | `59399` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000343` | `7db75f3d9d312ec2` | `ab0137ca69e1d869` | `.pdf` | `pdf` | `59242` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000344` | `cd844900d0c709a8` | `8fa99e10fc827a12` | `.pdf` | `pdf` | `59794` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000345` | `5b0873e8c6c44584` | `966a2b3ed6b79ad3` | `.pdf` | `pdf` | `61184` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000346` | `a5563398df283114` | `0cc535763fff1f1d` | `.pdf` | `pdf` | `54556` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000347` | `192d406a1da3c4b6` | `9b9106a9da7d198a` | `.pdf` | `pdf` | `59349` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000348` | `52bfe358ca0c3dac` | `4e47735edc377dab` | `.pdf` | `pdf` | `58889` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000349` | `22e8c31d4bb03035` | `14bb0f15d480ba75` | `.pdf` | `pdf` | `58634` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000350` | `e4b123b450d2ca88` | `dcba866fe257c06f` | `.pdf` | `pdf` | `60953` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000351` | `afddf188b6c40bfc` | `4d10d107e223037c` | `.pdf` | `pdf` | `812611` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000352` | `5954917b891f349d` | `a854bf4b19f823a2` | `.pdf` | `pdf` | `57730` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000353` | `9184b5ee85f867b1` | `2e33642beaf4e2c2` | `.pdf` | `pdf` | `60581` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000354` | `cf102c2be24bc8fb` | `d91458cbc80ec0b3` | `.pdf` | `pdf` | `57510` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000355` | `aa95ed6fec058c70` | `e9938db89fe36569` | `.pdf` | `pdf` | `59623` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000356` | `0653a74d9fde5484` | `3bf3b674dadb1106` | `.pdf` | `pdf` | `60687` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000357` | `995f39aad782c639` | `4dec8bbe451aefbc` | `.pdf` | `pdf` | `55199` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000358` | `b8c0a59c9e61fa03` | `69eb6c1ed8ef8419` | `.pdf` | `pdf` | `58715` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000359` | `2737f7fbc78d58d9` | `5239f4a4a84416e6` | `.pdf` | `pdf` | `57646` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000360` | `00d4b2cc68dee5b0` | `3c34723be1af6dde` | `.pdf` | `pdf` | `59104` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000361` | `2b46050e0e2ed133` | `60a92d21b64e71b1` | `.pdf` | `pdf` | `61037` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000362` | `48236082d18d32c5` | `e3e3da99d961810c` | `.pdf` | `pdf` | `59701` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000363` | `fb6944d4ee03c01e` | `a4d683dbfdeb0867` | `.pdf` | `pdf` | `56005` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000364` | `f6f786d3bbf12994` | `a2e6c647d6fbd5fb` | `.pdf` | `pdf` | `60588` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000365` | `bebbed5dfdc3cae9` | `b2563796d0555817` | `.pdf` | `pdf` | `59482` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000366` | `709be98988280326` | `5f2b017bbcf0eeed` | `.pdf` | `pdf` | `59533` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000367` | `ad27f6b0bae41420` | `d647aab22348af8c` | `.pdf` | `pdf` | `56719` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000368` | `f53b5658b3d8658b` | `15a60cc5f28a53b8` | `.pdf` | `pdf` | `58799` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000369` | `294cbccdf207aa00` | `2bb1db967e699ab0` | `.pdf` | `pdf` | `63932` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000370` | `b9e23c17e52a69ac` | `cc74b2a56bbe140c` | `.pdf` | `pdf` | `66880` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000371` | `a13ad3810f63327b` | `a341429df507b676` | `.pdf` | `pdf` | `59238` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000372` | `b221dff7b7bb77ae` | `976668eae213d6fa` | `.pdf` | `pdf` | `60258` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000373` | `1bb14e6373fe6112` | `c0e4a7aca10d48ce` | `.pdf` | `pdf` | `57805` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000374` | `036a646701ed0cc1` | `e4082c5135d5009b` | `.pdf` | `pdf` | `58638` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000375` | `971335a60e6a4689` | `ab71385b01efdb01` | `.pdf` | `pdf` | `56283` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000376` | `df536245d0ae584b` | `e704afc7af59114f` | `.pdf` | `pdf` | `57457` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000377` | `a1ffffbe83fb00f4` | `2bee4b82674ec55b` | `.pdf` | `pdf` | `62670` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000378` | `c091638e7fa19860` | `6d2770e684029e11` | `.pdf` | `pdf` | `57293` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000379` | `40fe12a461c5d6e8` | `1c4d4072a126ec9a` | `.pdf` | `pdf` | `60350` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000380` | `cb6d2ff8c787df12` | `e9a6bcbfcfec4611` | `.pdf` | `pdf` | `61741` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000381` | `28093817d781c7f8` | `285f43a517c7d92b` | `.pdf` | `pdf` | `56805` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000382` | `97e18869fa1bc5bf` | `03913fc9d043ca0d` | `.pdf` | `pdf` | `57370` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000383` | `63d17d4ac6bcb4a0` | `3b0503146e42878a` | `.pdf` | `pdf` | `55474` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000384` | `7eee1b6807bf8e5d` | `54bfafed7e4cfa11` | `.pdf` | `pdf` | `59341` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000385` | `4378c6e6d8fd9bce` | `0361f2e6e31227ef` | `.pdf` | `pdf` | `61302` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000386` | `7298d724d3cd5cab` | `66d749f3e701cf81` | `.pdf` | `pdf` | `56840` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000387` | `270e24ade98b9393` | `de111fb468db4a1f` | `.pdf` | `pdf` | `59363` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000388` | `bb936e139eb36c04` | `797f5e866d84b873` | `.pdf` | `pdf` | `60214` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000389` | `b7652527a7da41fd` | `b7016c49f5d1f08a` | `.pdf` | `pdf` | `56302` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000390` | `692c3026d4cfc127` | `c8b2f5ebeefe51ef` | `.pdf` | `pdf` | `56486` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000391` | `95bf1f7f8872c836` | `a5aaf162797ff2ec` | `.pdf` | `pdf` | `60688` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000392` | `aef65f8ecc81d56e` | `49a2488159a8d3fb` | `.pdf` | `pdf` | `60260` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000393` | `16f6b7c5e55303d1` | `fd6858bf289c7e95` | `.pdf` | `pdf` | `57498` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000394` | `644e2e97cf4036ee` | `7c187c4dff146d5b` | `.pdf` | `pdf` | `56875` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000395` | `e6402f7f9b6dc319` | `c523a8d760a40b47` | `.pdf` | `pdf` | `58820` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000396` | `a5d425168fb9fd36` | `8b12428e2a532831` | `.pdf` | `pdf` | `63606` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000397` | `f0539dffc3a63a4c` | `408d2f23d1453afa` | `.pdf` | `pdf` | `58447` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000398` | `2eff01a0ce7b33fc` | `cc68ac638cc990e5` | `.pdf` | `pdf` | `59725` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000399` | `d5316362cfad9a25` | `626efc06d87c2305` | `.pdf` | `pdf` | `64246` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000400` | `5166e223d1ad1a07` | `6091f1e305b95ddd` | `.pdf` | `pdf` | `57028` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000401` | `a2e531fa2308f220` | `7550cec142c85ac3` | `.pdf` | `pdf` | `56750` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000402` | `a6dfa185bf814533` | `c2e9a1dfc966bea5` | `.pdf` | `pdf` | `57539` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000403` | `cf047d80e833a1c7` | `a7f4e755779a0c4e` | `.pdf` | `pdf` | `57496` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000404` | `ffb9724df8b95b1d` | `41ce7c5587771417` | `.pdf` | `pdf` | `60966` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000405` | `197d88ad1003b62f` | `e701b8a62bd98f43` | `.pdf` | `pdf` | `56374` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000406` | `e343d6e27bdd33d2` | `14aacb41b7f6a9dd` | `.pdf` | `pdf` | `56965` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000407` | `c3b70179e99ff9ef` | `9e9822f07abc0e3f` | `.pdf` | `pdf` | `58366` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000408` | `415372882b200efd` | `f473b81bad882bdb` | `.pdf` | `pdf` | `60342` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000409` | `d1e2e0f045c1dea4` | `ee5a8109a56124f0` | `.pdf` | `pdf` | `64146` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000410` | `42db70d430e9a590` | `2761ebfb2198f165` | `.pdf` | `pdf` | `60615` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000411` | `86bea34548cdc2c6` | `dbe96d03ffaf66db` | `.pdf` | `pdf` | `59562` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000412` | `61f4058d0f094f40` | `fbebcc2b4d96c058` | `.pdf` | `pdf` | `56611` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000413` | `e10844d312622cb8` | `308437f53c636ad4` | `.pdf` | `pdf` | `3671663` | `3` | `review` | `phi3` | `0.349` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000414` | `079460e2457faaf3` | `590eac76e4d26fbe` | `.pdf` | `pdf` | `58784` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000415` | `8b83a67ddc72fcf4` | `cf28699f2f715bb8` | `.pdf` | `pdf` | `149189` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000416` | `f2728f27fedbfa7e` | `811a1195baef8c27` | `.pdf` | `pdf` | `152003` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000417` | `feb86d81246356ad` | `f6499cbea67358e5` | `.pdf` | `pdf` | `54418` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000418` | `0648632428c6de04` | `f93023281956bc62` | `.pdf` | `pdf` | `60012` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000419` | `c16daa65d1d978e2` | `a174956d0db54380` | `.pdf` | `pdf` | `105938` | `2` | `review` | `phi3` | `0.663` | `good` | `` | `lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000420` | `b082cc04a29f8523` | `fc36cfeafaa39f13` | `.pdf` | `pdf` | `123173` | `2` | `review` | `phi3` | `0.616` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000421` | `483ac21c66a8f798` | `6c4d2e682c98d489` | `.pdf` | `pdf` | `68603` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000422` | `079460e2457faaf3` | `86f1b02e55eb2e2d` | `.pdf` | `pdf` | `56629` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000423` | `a2cd39dbd6e49159` | `bc9a91a594ad0e09` | `.pdf` | `pdf` | `56471` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000424` | `8b83a67ddc72fcf4` | `de0c612ba30ba050` | `.pdf` | `pdf` | `139108` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000425` | `e72570c1366f776c` | `1f4af21cf53d2d6a` | `.pdf` | `pdf` | `143069` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000426` | `7f2ada658b9df18c` | `24dc2fd5a664395c` | `.pdf` | `pdf` | `92830` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000427` | `be34648088970895` | `0f675bf0406c2ecd` | `.pdf` | `pdf` | `92519` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000428` | `3589322bed91c404` | `ecf0724c4b350f98` | `.pdf` | `pdf` | `61429` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000429` | `3003d0a2a79a4808` | `386f03d61b430aa3` | `.pdf` | `pdf` | `66095` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000430` | `0648632428c6de04` | `4e43c4089552ba14` | `.pdf` | `pdf` | `56814` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000431` | `10e30d333b48f341` | `536d5305f2c6e632` | `.pdf` | `pdf` | `111822` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000432` | `ace990a799065d5a` | `f292958758af6c35` | `.pdf` | `pdf` | `68688` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000433` | `9919d1e29e163f54` | `e9479b3ba905f784` | `.pdf` | `pdf` | `74576` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000434` | `daaff6f09958e9ef` | `0a23f64472e580ea` | `.pdf` | `pdf` | `62182` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000435` | `4a2bcafc879266a0` | `96d65a312d66a2c1` | `.pdf` | `pdf` | `52750` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000436` | `6fff0f6713480aa7` | `7e44a1734a89c886` | `.pdf` | `pdf` | `63802` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000437` | `7a62c6d00bac2d7f` | `7b3fca623fa41b8e` | `.pdf` | `pdf` | `58723` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000438` | `6e878e2746295609` | `26dd3e3c9e029f1b` | `.pdf` | `pdf` | `59089` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000439` | `197d85e68b4a31bc` | `2037ab434199bd69` | `.pdf` | `pdf` | `118158` | `2` | `review` | `phi3` | `0.609` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000440` | `63a36be83016e45b` | `2197fc89907bc0ea` | `.pdf` | `pdf` | `60282` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000441` | `05fa82a152d96809` | `cc02a8e4aa544b60` | `.pdf` | `pdf` | `1299875` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000442` | `4d29280b42ad07b1` | `15d493917a6c2429` | `.pdf` | `pdf` | `61587` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000443` | `93505ee81c4c0182` | `daaf54a0184b26b4` | `.pdf` | `pdf` | `335219` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000444` | `61c2fc3ddfa4d562` | `23279b1383047e7c` | `.pdf` | `pdf` | `105848` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000445` | `8055abfb0a2e50a0` | `4b553c9d820e1fd8` | `.pdf` | `pdf` | `331835` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000446` | `cc97267a1b6981cf` | `80b029ca7450c500` | `.pdf` | `pdf` | `140420` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000447` | `0b9f1ab167b9b391` | `fccc10c2da8bb693` | `.pdf` | `pdf` | `323776` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000448` | `1fbf0fefb6c85e48` | `48b446330a3c79c8` | `.pdf` | `pdf` | `41935` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000449` | `e4816dbec11753e9` | `f34e70b29f45b92a` | `.pdf` | `pdf` | `139893` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000450` | `5f5bbc480f8da984` | `298492eaa7534a57` | `.pdf` | `pdf` | `95523` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000451` | `3725352973cf67c2` | `f286992c4dae2777` | `.pdf` | `pdf` | `317697` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000452` | `2fe168f65bd71d92` | `d577b8c0fb327e5f` | `.pdf` | `pdf` | `51413` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000453` | `2fb513976010560f` | `e3e13b473da92152` | `.pdf` | `pdf` | `319895` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000454` | `331cbc157c7e5788` | `53a5d1859c2ce25a` | `.pdf` | `pdf` | `59357` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000455` | `3187b85561f24023` | `ebd1a632d99a4544` | `.pdf` | `pdf` | `326209` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000456` | `3d439a1a605a150c` | `1e2ddfb22815f391` | `.pdf` | `pdf` | `56170` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000457` | `0121d575316083db` | `411cfb513b534a7e` | `.pdf` | `pdf` | `336247` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000458` | `866503d2bd89764d` | `114203a58ce5c2fa` | `.pdf` | `pdf` | `102726` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000459` | `35be113d09157786` | `72e9c8a1ce95969d` | `.pdf` | `pdf` | `73152` | `1` | `accepted` | `spacy` | `0.9` | `good` | `` | `accepted_clean_input` | `` |
| `corpus_file_000460` | `483ac21c66a8f798` | `7b507426ed694255` | `.pdf` | `pdf` | `68049` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000461` | `4d386411b1f733f9` | `bd56bedb2b4a4c8a` | `.pdf` | `pdf` | `62512` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000462` | `a2cd39dbd6e49159` | `c358301189781cdc` | `.pdf` | `pdf` | `56656` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000463` | `35cb1454d6d6255c` | `723b50b03a5597fc` | `.pdf` | `pdf` | `120036` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000464` | `8b83a67ddc72fcf4` | `6e746abacedb10ad` | `.pdf` | `pdf` | `139913` | `3` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000465` | `f2728f27fedbfa7e` | `2e38579ff9320579` | `.pdf` | `pdf` | `139595` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000466` | `3589322bed91c404` | `22d0989f6f3f7527` | `.pdf` | `pdf` | `63476` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000467` | `3003d0a2a79a4808` | `95de0861ecc6d5a0` | `.pdf` | `pdf` | `68407` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000468` | `10e30d333b48f341` | `23ce4e70dd7d2268` | `.pdf` | `pdf` | `116250` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000469` | `d8eefafd2a85ebd2` | `0e1e98402fcfa1ce` | `.pdf` | `pdf` | `2707773` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000470` | `ace990a799065d5a` | `e0065a0a6b963781` | `.pdf` | `pdf` | `83269` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000471` | `ffcf5a87c11a763f` | `5765913c3ce021e2` | `.pdf` | `pdf` | `74914` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000472` | `daaff6f09958e9ef` | `984114b158c01dd2` | `.pdf` | `pdf` | `62845` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000473` | `4a2bcafc879266a0` | `ab18069c3a8897e5` | `.pdf` | `pdf` | `53432` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000474` | `6fff0f6713480aa7` | `65a98a8c8d4d5c92` | `.pdf` | `pdf` | `63057` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000475` | `7a62c6d00bac2d7f` | `58c75f357ba8d3fd` | `.pdf` | `pdf` | `57857` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000476` | `6e878e2746295609` | `f50ca3db5b3a48d1` | `.pdf` | `pdf` | `54846` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000477` | `41d9a4b5474f5d72` | `3b5f2449995a5cb1` | `.pdf` | `pdf` | `24465` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000478` | `d8eefafd2a85ebd2` | `3ba6fae98f6b6ec0` | `.pdf` | `pdf` | `3034130` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000479` | `e7b88e5fbeeeb31d` | `b3456618b4bbeee6` | `.pdf` | `pdf` | `60528` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000480` | `be34648088970895` | `2c1b844bebb97169` | `.pdf` | `pdf` | `37458` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000481` | `e70915f36bbd2af2` | `d0d11afed2cefba2` | `.pdf` | `pdf` | `75213` | `1` | `accepted` | `spacy` | `0.9` | `good` | `` | `accepted_clean_input` | `` |
| `corpus_file_000482` | `d8eefafd2a85ebd2` | `a9f63b8f6b1e27fd` | `.pdf` | `pdf` | `253214` | `4` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000483` | `9de9e57f99dc54cb` | `eeaab68c66f71220` | `.pdf` | `pdf` | `275166` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000484` | `125e9d8b5cf60b8a` | `c154daeb4957ce9f` | `.pdf` | `pdf` | `122041` | `7` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000485` | `194d24fc748111ef` | `2fd15adfe235f653` | `.pdf` | `pdf` | `1916180` | `9` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000486` | `64cae37a6dc47a18` | `71739edf92b1af03` | `.pdf` | `pdf` | `201014` | `10` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000487` | `5fa2c62a746d31cf` | `808a05ec9dfaa25f` | `.pdf` | `pdf` | `218932` | `6` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000488` | `eefdda65b29818f6` | `4827fd7060e5e285` | `.pdf` | `pdf` | `633760` | `1` | `review_ocr_quality` | `claude` | `0.5` | `empty` | `` | `empty_or_near_empty_text, extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000489` | `973978bafc8d72ff` | `118e1b8575ae47ab` | `.pdf` | `pdf` | `877018` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000490` | `5c528f80ffebc1bc` | `5a19325581efda6e` | `.pdf` | `pdf` | `1213148` | `2` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000491` | `c45955eade7d9c72` | `3f8297c6dd625e1e` | `.pdf` | `pdf` | `1155235` | `2` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000492` | `446c9e8616c1df87` | `523c55feee695168` | `.pdf` | `pdf` | `912019` | `1` | `review_ocr_quality` | `rules_based` | `0.45` | `poor_ocr` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, poor_input_ocr, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000493` | `02730b17c5d41a69` | `d6c779e738979ac5` | `.pdf` | `pdf` | `4791551` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000494` | `c5f587ed6dfc60a9` | `5e8ec036c0b41176` | `.pdf` | `pdf` | `662167` | `1` | `review_ocr_quality` | `claude` | `0.5` | `empty` | `` | `empty_or_near_empty_text, extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000495` | `1ff673421c6e0b5a` | `11b5ccb38fecfe81` | `.pdf` | `pdf` | `2825555` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000496` | `b44c9779c43be451` | `3f330590ca971c1c` | `.pdf` | `pdf` | `1151591` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, table_structure_loss` | `` |
| `corpus_file_000497` | `f9a889376c909061` | `7e731c1b8ca7c354` | `.pdf` | `pdf` | `50066` | `2` | `review` | `phi3` | `0.064` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence` | `` |
| `corpus_file_000498` | `189b62913328d3c1` | `4ede2df7bce2aa4f` | `.pdf` | `pdf` | `6800205` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000499` | `455c718547867264` | `1e79acda28bb30a9` | `.tif` | `image` | `199286` | `` | `review` | `rules_based` | `0.45` | `good` | `` | `low_text_density, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities` | `` |
| `corpus_file_000500` | `841906333ffe78cc` | `32b86a2545270907` | `.pdf` | `pdf` | `122848` | `2` | `review` | `phi3` | `0.0` | `good` | `` | `extraction_low_confidence, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000501` | `0a0731a6b1af7ab2` | `5e3c596409e4925e` | `.pdf` | `pdf` | `1106445` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000502` | `8a95b8f11e90d2cf` | `ab6b19197834b828` | `.pdf` | `pdf` | `4078118` | `4` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000503` | `f4db6f31d187f25d` | `0f6bf333b6d8b774` | `.pdf` | `pdf` | `1473281` | `1` | `accepted` | `spacy` | `0.838` | `usable_with_review` | `` | `accepted_clean_input, table_structure_loss` | `` |
| `corpus_file_000504` | `6bdebed7ddcf76b0` | `202055b3f456ad2f` | `.pdf` | `pdf` | `807461` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000505` | `a5cb84573bda8565` | `18464f46db9c9c31` | `.pdf` | `pdf` | `5786303` | `1` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000506` | `a94042fd67bb3f0c` | `c6ae4a6c341f6537` | `.pdf` | `pdf` | `26983` | `2` | `review` | `phi3` | `0.299` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000507` | `baa498b08c569880` | `1acc986c8464b747` | `.pdf` | `pdf` | `72163` | `1` | `review` | `phi3` | `0.485` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000508` | `bd1280b947cbe243` | `282baff4d3f4f722` | `.pdf` | `pdf` | `9458914` | `10` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000509` | `866e80b1fdcac3fb` | `5e0f5d84eaa6224a` | `.pdf` | `pdf` | `66424` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000510` | `67d04689cb7dd0dc` | `86e9d533711e0d86` | `.pdf` | `pdf` | `104501` | `2` | `review` | `phi3` | `0.569` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000511` | `fb61968bfaa85b06` | `f2a643256f2c30c4` | `.pdf` | `pdf` | `10010189` | `5` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000512` | `bf693112a862d862` | `c5b9bfd4406b5705` | `.pdf` | `pdf` | `125382` | `5` | `review` | `rules_based` | `0.45` | `good` | `possible_multi_document` | `extraction_low_confidence, extraction_low_coverage, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000513` | `f8278ded24cb4b4c` | `0990307041cbbec7` | `.pdf` | `pdf` | `58187` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000514` | `d702369270f20c89` | `a0ea90833b48f0eb` | `.pdf` | `pdf` | `50905` | `4` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000515` | `a94042fd67bb3f0c` | `d8c3f41c7b5aa67f` | `.pdf` | `pdf` | `28994` | `1` | `review` | `phi3` | `0.344` | `usable_with_review` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000516` | `76a90239b420e4b7` | `b6dcb440936e5788` | `.pdf` | `pdf` | `3402703` | `2` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000517` | `d702369270f20c89` | `5ebc59c86d333bbb` | `.pdf` | `pdf` | `94591` | `2` | `review` | `phi3` | `0.279` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000518` | `2eff01a0ce7b33fc` | `cc68ac638cc990e5` | `.pdf` | `pdf` | `59725` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000519` | `d5316362cfad9a25` | `626efc06d87c2305` | `.pdf` | `pdf` | `64246` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000520` | `415372882b200efd` | `f473b81bad882bdb` | `.pdf` | `pdf` | `60342` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000521` | `c16daa65d1d978e2` | `a174956d0db54380` | `.pdf` | `pdf` | `105938` | `2` | `review` | `phi3` | `0.663` | `good` | `` | `lab_report_detected, low_text_density, table_structure_loss` | `` |
| `corpus_file_000522` | `c4a491bc06a43cda` | `a9daf124b06e6a1c` | `.pdf` | `pdf` | `77963` | `1` | `review` | `phi3` | `0.57` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000523` | `b082cc04a29f8523` | `fc36cfeafaa39f13` | `.pdf` | `pdf` | `123173` | `2` | `review` | `phi3` | `0.616` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000524` | `bd1280b947cbe243` | `026f9123d89f946f` | `.pdf` | `pdf` | `9942835` | `7` | `review` | `rules_based` | `0.45` | `good` | `` | `document_type_prescription_not_lab, extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, language_aware_ocr_required, non_lab_document_skipped_lab_normalization, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000525` | `e7b88e5fbeeeb31d` | `2e8f76d5cf7ca8c7` | `.pdf` | `pdf` | `1718152` | `2` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000526` | `b961b03474de6d70` | `dc48ebd63fd1fc34` | `.pdf` | `pdf` | `46578` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000527` | `ca8ac8e1c74b9034` | `0e8fc605b2b81413` | `.pdf` | `pdf` | `54036` | `5` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000528` | `32003e3ce34f9cf4` | `06af6e47559b51a7` | `.mp3` | `unsupported` | `16720456` | `` | `error` | `None` |  | `None` | `` | `unsupported_format` | `unsupported_format` |
| `corpus_file_000529` | `896e537158a301b8` | `64d2b4c5f8c45c3f` | `.jpg` | `image` | `3800485` | `` | `accepted` | `spacy` | `0.83` | `usable_with_review` | `` | `low_text_density, accepted_clean_input` | `` |
| `corpus_file_000530` | `e0fac98c024dac6f` | `f11d2bfe92966a84` | `.jpg` | `image` | `3609920` | `` | `review_ocr_quality` | `rules_based` | `0.45` | `poor_ocr` | `` | `poor_input_ocr, table_structure_loss, extraction_low_coverage, extraction_low_confidence, safety_gate_low_confidence, extraction_sparse_entities, classifier_legacy_ocr_flag, legacy_suspicious_ocr_context` | `` |
| `corpus_file_000531` | `ce38df13e1c4ff15` | `877f2ca72e0a8f44` | `.jpg` | `image` | `3035180` | `` | `accepted` | `spacy` | `0.83` | `usable_with_review` | `` | `low_text_density, accepted_clean_input` | `` |
| `corpus_file_000532` | `6b81e88db307020f` | `2b83f07b76168cac` | `.pdf` | `pdf` | `54543` | `1` | `review` | `rules_based` | `0.45` | `good` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000533` | `43b6c63fcf45574c` | `f82f79e24c0bb695` | `.pdf` | `pdf` | `78621` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000534` | `c16daa65d1d978e2` | `0c8a597db3ab75ac` | `.pdf` | `pdf` | `106974` | `2` | `review` | `phi3` | `0.612` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000535` | `a8c84b8d1cdd172d` | `ab03384de42f33e5` | `.pdf` | `pdf` | `80371` | `2` | `review` | `phi3` | `0.424` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000536` | `ae272f30f58beeb5` | `608b8d74c44c93c7` | `.pdf` | `pdf` | `60956` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, low_text_density, table_structure_loss` | `` |
| `corpus_file_000537` | `84f9a395fd2fafef` | `473ebb441ca38f4b` | `.pdf` | `pdf` | `78256` | `1` | `review` | `phi3` | `0.631` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000538` | `197d85e68b4a31bc` | `773df8c30ca153eb` | `.pdf` | `pdf` | `122046` | `2` | `review` | `phi3` | `0.609` | `good` | `` | `extraction_low_confidence, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000539` | `a94042fd67bb3f0c` | `74f55b08e66fd351` | `.pdf` | `pdf` | `31706` | `2` | `review` | `phi3` | `0.311` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000540` | `ebe4fc53222e01ad` | `479c06c0470aa9a8` | `.pdf` | `pdf` | `27222` | `1` | `review` | `phi3` | `0.479` | `usable_with_review` | `` | `extraction_low_confidence, safety_gate_low_confidence` | `` |
| `corpus_file_000541` | `ee824dd6b1d7ac8f` | `33a74a2f6c35bbe6` | `.pdf` | `pdf` | `453826` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, lab_report_detected, low_text_density, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000542` | `5ebc11d8a0156d41` | `cc221a23e926e8c1` | `.pdf` | `pdf` | `480404` | `1` | `review` | `rules_based` | `0.45` | `usable_with_review` | `` | `extraction_low_confidence, extraction_low_coverage, extraction_sparse_entities, safety_gate_low_confidence, table_structure_loss` | `` |
| `corpus_file_000543` | `dfec3512401361ad` | `afeaa6908e130bcd` | `.pdf` | `pdf` | `32842` | `2` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected, table_structure_loss` | `` |
| `corpus_file_000544` | `52c0d690eda98c34` | `ef0abadfeaa8d0d6` | `.pdf` | `pdf` | `74404` | `1` | `accepted` | `spacy` | `0.838` | `good` | `` | `accepted_clean_input, lab_report_detected` | `` |
| `corpus_file_000545` | `2816ffe1a2aa5b66` | `657b63a6498e48a1` | `.pdf` | `pdf` | `82256` | `2` | `review` | `phi3` | `0.609` | `good` | `` | `extraction_low_confidence, lab_report_detected, safety_gate_low_confidence` | `` |
| `corpus_file_000546` | `cfd5b25dc5f16bd3` | `e1af1a10a699d310` | `.pdf` | `pdf` | `53971` | `2` | `review` | `phi3` | `0.569` | `good` | `possible_multi_document` | `extraction_low_confidence, lab_report_detected, possible_multi_document_pdf, safety_gate_low_confidence, table_structure_loss` | `` |

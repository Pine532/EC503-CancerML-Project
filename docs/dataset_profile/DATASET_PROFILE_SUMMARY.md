# Dataset Profile Summary

This file summarizes the datasets exactly as the canonical modeling pipeline loads them.

## Dataset-Level Summary

| Dataset Mode | Dataset Name | Target | Raw Rows | Raw Columns | Modeling Rows | Modeling Columns | Predictor Features Before Encoding | Categorical Features | Numerical Features | Estimated One-Hot Columns | Estimated Encoded Feature Columns | Target Mean | Target Std | Target Min | Target 25% | Target Median | Target 75% | Target Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gdsc_metadata_only | gdsc | LN_IC50 | 242035 | 19 | 242035 | 15 | 12 | 12 | 0 | 358 | 358 | 2.8171108215423395 | 2.7621906813657313 | -8.747724 | 1.508054 | 3.236744 | 4.7001105 | 13.820189 |
| gdsc_metadata_plus_expression | gdsc | LN_IC50 | 236791 | 519 | 236791 | 515 | 512 | 12 | 500 | 368 | 868 | 2.8257086277008057 | 2.762622833251953 | -8.747723579406738 | 1.5159475207328796 | 3.245102882385254 | 4.707631587982178 | 13.820189476013184 |
| gdsc_auc_metadata_only | gdsc | AUC | 242035 | 19 | 242035 | 15 | 12 | 12 | 0 | 358 | 358 | 0.8825932131262008 | 0.1469977553830758 | 0.006282 | 0.849452 | 0.944197 | 0.974934 | 0.998904 |
| gdsc_auc_metadata_plus_expression | gdsc | AUC | 236791 | 519 | 236791 | 515 | 512 | 12 | 500 | 368 | 868 | 0.8830408453941345 | 0.14655107259750366 | 0.006281999871134758 | 0.8501765131950378 | 0.9445120096206665 | 0.9749860167503357 | 0.9989039897918701 |
| secondary_screen_auc | secondary_screen_auc | target_auc | 690192 | 15 | 690192 | 11 | 8 | 8 | 0 | 3241 | 3241 | 0.9575658842860398 | 0.28237067051722414 | 0.0041740679996407 | 0.806168809851867 | 0.9156653885772892 | 1.12657550899011 | 4.88916233189892 |
| secondary_screen_ic50 | secondary_screen_ic50 | target_log_ic50 | 355784 | 16 | 355784 | 11 | 8 | 8 | 0 | 3172 | 3172 | -0.02718610759451552 | 3.071689499216146 | -240.44832727396584 | -0.3788371756610514 | 0.1756286878540254 | 0.5010697006783689 | 299.1668388541871 |

## Categorical Feature Summary

| Dataset Mode | Feature | Unique Values | Missing Count | Most Common Value | Most Common Count | Most Common Percent | Top Values |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gdsc_metadata_only | TCGA_DESC | 32 | 1067 | UNCLASSIFIED | 45690 | 18.87743508170306 | UNCLASSIFIED: 45690 (18.9%); LUAD: 15653 (6.5%); SCLC: 13570 (5.6%); BRCA: 13106 (5.4%); SKCM: 12637 (5.2%); COREAD: 12538 (5.2%); HNSC: 9358 (3.9%); ESCA: 9126 (3.8%) |
| gdsc_metadata_only | GDSC Tissue descriptor 1 | 19 | 9366 | lung_NSCLC | 26977 | 11.145908649575475 | lung_NSCLC: 26977 (11.1%); urogenital_system: 25707 (10.6%); leukemia: 20484 (8.5%); aero_dig_tract: 18583 (7.7%); lymphoma: 16747 (6.9%); lung_SCLC: 13750 (5.7%); breast: 13388 (5.5%); nervous_system: 12894 (5.3%) |
| gdsc_metadata_only | GDSC Tissue descriptor 2 | 54 | 9366 | lung_NSCLC_adenocarcinoma | 16112 | 6.656888466544094 | lung_NSCLC_adenocarcinoma: 16112 (6.7%); lung_small_cell_carcinoma: 13750 (5.7%); breast: 13388 (5.5%); large_intestine: 12438 (5.1%); melanoma: 12097 (5.0%); glioma: 11822 (4.9%); ovary: 10434 (4.3%); head and neck: 9457 (3.9%) |
| gdsc_metadata_only | Cancer Type (matching TCGA label) | 31 | 51446 | Missing | 51446 | 21.255603528415314 | Missing: 51446 (21.3%); LUAD: 15483 (6.4%); SCLC: 13750 (5.7%); BRCA: 13106 (5.4%); COAD/READ: 12438 (5.1%); SKCM: 12097 (5.0%); HNSC: 9178 (3.8%); ESCA: 9126 (3.8%) |
| gdsc_metadata_only | Microsatellite instability Status (MSI) | 2 | 12353 | MSS/MSI-L | 214104 | 88.45993348069494 | MSS/MSI-L: 214104 (88.5%); MSI-H: 15578 (6.4%); Missing: 12353 (5.1%) |
| gdsc_metadata_only | Screen Medium | 2 | 9366 | R | 129756 | 53.61042824384903 | R: 129756 (53.6%); D/F12: 102913 (42.5%); Missing: 9366 (3.9%) |
| gdsc_metadata_only | Growth Properties | 3 | 9366 | Adherent | 168430 | 69.58910901315926 | Adherent: 168430 (69.6%); Suspension: 56814 (23.5%); Missing: 9366 (3.9%); Semi-Adherent: 7425 (3.1%) |
| gdsc_metadata_only | CNA | 2 | 9366 | Y | 231651 | 95.70971140537526 | Y: 231651 (95.7%); Missing: 9366 (3.9%); N: 1018 (0.4%) |
| gdsc_metadata_only | Gene Expression | 2 | 9366 | Y | 227885 | 94.15373809572995 | Y: 227885 (94.2%); Missing: 9366 (3.9%); N: 4784 (2.0%) |
| gdsc_metadata_only | Methylation | 2 | 9366 | Y | 225081 | 92.99522796289793 | Y: 225081 (93.0%); Missing: 9366 (3.9%); N: 7588 (3.1%) |
| gdsc_metadata_only | TARGET | 185 | 27155 | Missing | 27155 | 11.219451732187494 | Missing: 27155 (11.2%); PARP1, PARP2: 4714 (1.9%); MEK1, MEK2: 4547 (1.9%); TOP1: 4324 (1.8%); EGFR: 3836 (1.6%); TNKS1, TNKS2: 3699 (1.5%); AKT1, AKT2, AKT3: 3308 (1.4%); DOT1L: 2873 (1.2%) |
| gdsc_metadata_only | TARGET_PATHWAY | 24 | 0 | Unclassified | 24979 | 10.320408205424837 | Unclassified: 24979 (10.3%); PI3K/MTOR signaling: 22724 (9.4%); Other: 21402 (8.8%); DNA replication: 17649 (7.3%); Other, kinases: 17277 (7.1%); ERK MAPK signaling: 13350 (5.5%); Genome integrity: 12221 (5.0%); Cell cycle: 11620 (4.8%) |
| gdsc_metadata_plus_expression | TCGA_DESC | 33 | 0 | UNCLASSIFIED | 45038 | 19.020148569835847 | UNCLASSIFIED: 45038 (19.0%); LUAD: 15473 (6.5%); SCLC: 13032 (5.5%); BRCA: 12750 (5.4%); SKCM: 12277 (5.2%); COREAD: 12077 (5.1%); HNSC: 9358 (4.0%); ESCA: 9126 (3.9%) |
| gdsc_metadata_plus_expression | GDSC Tissue descriptor 1 | 20 | 0 | lung_NSCLC | 26619 | 11.241559011955692 | lung_NSCLC: 26619 (11.2%); urogenital_system: 25163 (10.6%); leukemia: 20125 (8.5%); aero_dig_tract: 18583 (7.8%); lymphoma: 16210 (6.8%); lung_SCLC: 13212 (5.6%); breast: 13032 (5.5%); nervous_system: 12714 (5.4%) |
| gdsc_metadata_plus_expression | GDSC Tissue descriptor 2 | 55 | 0 | lung_NSCLC_adenocarcinoma | 15932 | 6.7282962612599295 | lung_NSCLC_adenocarcinoma: 15932 (6.7%); lung_small_cell_carcinoma: 13212 (5.6%); breast: 13032 (5.5%); large_intestine: 12257 (5.2%); melanoma: 11917 (5.0%); glioma: 11642 (4.9%); ovary: 10070 (4.3%); head and neck: 9457 (4.0%) |
| gdsc_metadata_plus_expression | Cancer Type (matching TCGA label) | 32 | 0 | Unknown | 50155 | 21.18112597184859 | Unknown: 50155 (21.2%); LUAD: 15303 (6.5%); SCLC: 13212 (5.6%); BRCA: 12750 (5.4%); COAD/READ: 12257 (5.2%); SKCM: 11917 (5.0%); HNSC: 9178 (3.9%); ESCA: 9126 (3.9%) |
| gdsc_metadata_plus_expression | Microsatellite instability Status (MSI) | 3 | 0 | MSS/MSI-L | 209677 | 88.5493958807556 | MSS/MSI-L: 209677 (88.5%); MSI-H: 15401 (6.5%); Unknown: 11713 (4.9%) |
| gdsc_metadata_plus_expression | Screen Medium | 3 | 0 | R | 126948 | 53.61183490926598 | R: 126948 (53.6%); D/F12: 100937 (42.6%); Unknown: 8906 (3.8%) |
| gdsc_metadata_plus_expression | Growth Properties | 4 | 0 | Adherent | 165260 | 69.79150390006377 | Adherent: 165260 (69.8%); Suspension: 55560 (23.5%); Unknown: 8906 (3.8%); Semi-Adherent: 7065 (3.0%) |
| gdsc_metadata_plus_expression | CNA | 3 | 0 | Y | 226867 | 95.80896233387249 | Y: 226867 (95.8%); Unknown: 8906 (3.8%); N: 1018 (0.4%) |
| gdsc_metadata_plus_expression | Gene Expression | 2 | 0 | Y | 227885 | 96.23887732219552 | Y: 227885 (96.2%); Unknown: 8906 (3.8%) |
| gdsc_metadata_plus_expression | Methylation | 3 | 0 | Y | 222269 | 93.86716555950184 | Y: 222269 (93.9%); Unknown: 8906 (3.8%); N: 5616 (2.4%) |
| gdsc_metadata_plus_expression | TARGET | 186 | 0 | Unknown | 27068 | 11.43117770523373 | Unknown: 27068 (11.4%); PARP1, PARP2: 4575 (1.9%); MEK1, MEK2: 4433 (1.9%); TOP1: 4236 (1.8%); EGFR: 3725 (1.6%); TNKS1, TNKS2: 3589 (1.5%); AKT1, AKT2, AKT3: 3247 (1.4%); DOT1L: 2789 (1.2%) |
| gdsc_metadata_plus_expression | TARGET_PATHWAY | 24 | 0 | Unclassified | 24899 | 10.515180053295945 | Unclassified: 24899 (10.5%); PI3K/MTOR signaling: 22148 (9.4%); Other: 21033 (8.9%); DNA replication: 17323 (7.3%); Other, kinases: 16879 (7.1%); ERK MAPK signaling: 13034 (5.5%); Genome integrity: 11861 (5.0%); Cell cycle: 11335 (4.8%) |
| gdsc_auc_metadata_only | TCGA_DESC | 32 | 1067 | UNCLASSIFIED | 45690 | 18.87743508170306 | UNCLASSIFIED: 45690 (18.9%); LUAD: 15653 (6.5%); SCLC: 13570 (5.6%); BRCA: 13106 (5.4%); SKCM: 12637 (5.2%); COREAD: 12538 (5.2%); HNSC: 9358 (3.9%); ESCA: 9126 (3.8%) |
| gdsc_auc_metadata_only | GDSC Tissue descriptor 1 | 19 | 9366 | lung_NSCLC | 26977 | 11.145908649575475 | lung_NSCLC: 26977 (11.1%); urogenital_system: 25707 (10.6%); leukemia: 20484 (8.5%); aero_dig_tract: 18583 (7.7%); lymphoma: 16747 (6.9%); lung_SCLC: 13750 (5.7%); breast: 13388 (5.5%); nervous_system: 12894 (5.3%) |
| gdsc_auc_metadata_only | GDSC Tissue descriptor 2 | 54 | 9366 | lung_NSCLC_adenocarcinoma | 16112 | 6.656888466544094 | lung_NSCLC_adenocarcinoma: 16112 (6.7%); lung_small_cell_carcinoma: 13750 (5.7%); breast: 13388 (5.5%); large_intestine: 12438 (5.1%); melanoma: 12097 (5.0%); glioma: 11822 (4.9%); ovary: 10434 (4.3%); head and neck: 9457 (3.9%) |
| gdsc_auc_metadata_only | Cancer Type (matching TCGA label) | 31 | 51446 | Missing | 51446 | 21.255603528415314 | Missing: 51446 (21.3%); LUAD: 15483 (6.4%); SCLC: 13750 (5.7%); BRCA: 13106 (5.4%); COAD/READ: 12438 (5.1%); SKCM: 12097 (5.0%); HNSC: 9178 (3.8%); ESCA: 9126 (3.8%) |
| gdsc_auc_metadata_only | Microsatellite instability Status (MSI) | 2 | 12353 | MSS/MSI-L | 214104 | 88.45993348069494 | MSS/MSI-L: 214104 (88.5%); MSI-H: 15578 (6.4%); Missing: 12353 (5.1%) |
| gdsc_auc_metadata_only | Screen Medium | 2 | 9366 | R | 129756 | 53.61042824384903 | R: 129756 (53.6%); D/F12: 102913 (42.5%); Missing: 9366 (3.9%) |
| gdsc_auc_metadata_only | Growth Properties | 3 | 9366 | Adherent | 168430 | 69.58910901315926 | Adherent: 168430 (69.6%); Suspension: 56814 (23.5%); Missing: 9366 (3.9%); Semi-Adherent: 7425 (3.1%) |
| gdsc_auc_metadata_only | CNA | 2 | 9366 | Y | 231651 | 95.70971140537526 | Y: 231651 (95.7%); Missing: 9366 (3.9%); N: 1018 (0.4%) |
| gdsc_auc_metadata_only | Gene Expression | 2 | 9366 | Y | 227885 | 94.15373809572995 | Y: 227885 (94.2%); Missing: 9366 (3.9%); N: 4784 (2.0%) |
| gdsc_auc_metadata_only | Methylation | 2 | 9366 | Y | 225081 | 92.99522796289793 | Y: 225081 (93.0%); Missing: 9366 (3.9%); N: 7588 (3.1%) |
| gdsc_auc_metadata_only | TARGET | 185 | 27155 | Missing | 27155 | 11.219451732187494 | Missing: 27155 (11.2%); PARP1, PARP2: 4714 (1.9%); MEK1, MEK2: 4547 (1.9%); TOP1: 4324 (1.8%); EGFR: 3836 (1.6%); TNKS1, TNKS2: 3699 (1.5%); AKT1, AKT2, AKT3: 3308 (1.4%); DOT1L: 2873 (1.2%) |
| gdsc_auc_metadata_only | TARGET_PATHWAY | 24 | 0 | Unclassified | 24979 | 10.320408205424837 | Unclassified: 24979 (10.3%); PI3K/MTOR signaling: 22724 (9.4%); Other: 21402 (8.8%); DNA replication: 17649 (7.3%); Other, kinases: 17277 (7.1%); ERK MAPK signaling: 13350 (5.5%); Genome integrity: 12221 (5.0%); Cell cycle: 11620 (4.8%) |
| gdsc_auc_metadata_plus_expression | TCGA_DESC | 33 | 0 | UNCLASSIFIED | 45038 | 19.020148569835847 | UNCLASSIFIED: 45038 (19.0%); LUAD: 15473 (6.5%); SCLC: 13032 (5.5%); BRCA: 12750 (5.4%); SKCM: 12277 (5.2%); COREAD: 12077 (5.1%); HNSC: 9358 (4.0%); ESCA: 9126 (3.9%) |
| gdsc_auc_metadata_plus_expression | GDSC Tissue descriptor 1 | 20 | 0 | lung_NSCLC | 26619 | 11.241559011955692 | lung_NSCLC: 26619 (11.2%); urogenital_system: 25163 (10.6%); leukemia: 20125 (8.5%); aero_dig_tract: 18583 (7.8%); lymphoma: 16210 (6.8%); lung_SCLC: 13212 (5.6%); breast: 13032 (5.5%); nervous_system: 12714 (5.4%) |
| gdsc_auc_metadata_plus_expression | GDSC Tissue descriptor 2 | 55 | 0 | lung_NSCLC_adenocarcinoma | 15932 | 6.7282962612599295 | lung_NSCLC_adenocarcinoma: 15932 (6.7%); lung_small_cell_carcinoma: 13212 (5.6%); breast: 13032 (5.5%); large_intestine: 12257 (5.2%); melanoma: 11917 (5.0%); glioma: 11642 (4.9%); ovary: 10070 (4.3%); head and neck: 9457 (4.0%) |
| gdsc_auc_metadata_plus_expression | Cancer Type (matching TCGA label) | 32 | 0 | Unknown | 50155 | 21.18112597184859 | Unknown: 50155 (21.2%); LUAD: 15303 (6.5%); SCLC: 13212 (5.6%); BRCA: 12750 (5.4%); COAD/READ: 12257 (5.2%); SKCM: 11917 (5.0%); HNSC: 9178 (3.9%); ESCA: 9126 (3.9%) |
| gdsc_auc_metadata_plus_expression | Microsatellite instability Status (MSI) | 3 | 0 | MSS/MSI-L | 209677 | 88.5493958807556 | MSS/MSI-L: 209677 (88.5%); MSI-H: 15401 (6.5%); Unknown: 11713 (4.9%) |
| gdsc_auc_metadata_plus_expression | Screen Medium | 3 | 0 | R | 126948 | 53.61183490926598 | R: 126948 (53.6%); D/F12: 100937 (42.6%); Unknown: 8906 (3.8%) |
| gdsc_auc_metadata_plus_expression | Growth Properties | 4 | 0 | Adherent | 165260 | 69.79150390006377 | Adherent: 165260 (69.8%); Suspension: 55560 (23.5%); Unknown: 8906 (3.8%); Semi-Adherent: 7065 (3.0%) |
| gdsc_auc_metadata_plus_expression | CNA | 3 | 0 | Y | 226867 | 95.80896233387249 | Y: 226867 (95.8%); Unknown: 8906 (3.8%); N: 1018 (0.4%) |
| gdsc_auc_metadata_plus_expression | Gene Expression | 2 | 0 | Y | 227885 | 96.23887732219552 | Y: 227885 (96.2%); Unknown: 8906 (3.8%) |
| gdsc_auc_metadata_plus_expression | Methylation | 3 | 0 | Y | 222269 | 93.86716555950184 | Y: 222269 (93.9%); Unknown: 8906 (3.8%); N: 5616 (2.4%) |
| gdsc_auc_metadata_plus_expression | TARGET | 186 | 0 | Unknown | 27068 | 11.43117770523373 | Unknown: 27068 (11.4%); PARP1, PARP2: 4575 (1.9%); MEK1, MEK2: 4433 (1.9%); TOP1: 4236 (1.8%); EGFR: 3725 (1.6%); TNKS1, TNKS2: 3589 (1.5%); AKT1, AKT2, AKT3: 3247 (1.4%); DOT1L: 2789 (1.2%) |
| gdsc_auc_metadata_plus_expression | TARGET_PATHWAY | 24 | 0 | Unclassified | 24899 | 10.515180053295945 | Unclassified: 24899 (10.5%); PI3K/MTOR signaling: 22148 (9.4%); Other: 21033 (8.9%); DNA replication: 17323 (7.3%); Other, kinases: 16879 (7.1%); ERK MAPK signaling: 13034 (5.5%); Genome integrity: 11861 (5.0%); Cell cycle: 11335 (4.8%) |
| secondary_screen_auc | ccle_tissue | 20 | 0 | LUNG | 134114 | 19.431404594663515 | LUNG: 134114 (19.4%); TRACT: 75271 (10.9%); SKIN: 58583 (8.5%); SYSTEM: 49622 (7.2%); PANCREAS: 47087 (6.8%); OVARY: 42636 (6.2%); INTESTINE: 38439 (5.6%); OESOPHAGUS: 34898 (5.1%) |
| secondary_screen_auc | screen_id | 4 | 0 | HTS002 | 592912 | 85.90537125901199 | HTS002: 592912 (85.9%); MTS010: 63528 (9.2%); MTS006: 32930 (4.8%); MTS005: 822 (0.1%) |
| secondary_screen_auc | name | 1448 | 0 | talazoparib | 1357 | 0.19661195725247468 | talazoparib: 1357 (0.2%); selinexor: 939 (0.1%); tosedostat: 938 (0.1%); doxycycline: 936 (0.1%); oxiracetam: 932 (0.1%); P276-00: 931 (0.1%); dinaciclib: 929 (0.1%); narasin: 927 (0.1%) |
| secondary_screen_auc | moa | 531 | 0 | Unknown | 30084 | 4.35878712010571 | Unknown: 30084 (4.4%); EGFR inhibitor: 20307 (2.9%); HDAC inhibitor: 12937 (1.9%); CDK inhibitor: 10644 (1.5%); tubulin polymerization inhibitor: 10436 (1.5%); topoisomerase inhibitor: 10271 (1.5%); MEK inhibitor: 10118 (1.5%); glucocorticoid receptor agonist: 9756 (1.4%) |
| secondary_screen_auc | target | 791 | 0 | Unknown | 126092 | 18.269119317523238 | Unknown: 126092 (18.3%); MTOR: 8547 (1.2%); EGFR: 7860 (1.1%); HSP90AA1: 5808 (0.8%); MET: 4857 (0.7%); TOP2A: 4251 (0.6%); NR3C1: 4123 (0.6%); BCL2: 3939 (0.6%) |
| secondary_screen_auc | disease.area | 100 | 0 | Unknown | 449031 | 65.05885318867793 | Unknown: 449031 (65.1%); oncology: 48427 (7.0%); infectious disease: 29415 (4.3%); hematologic malignancy: 24714 (3.6%); neurology/psychiatry: 21737 (3.1%); cardiology: 14569 (2.1%); dermatology: 11772 (1.7%); endocrinology: 11579 (1.7%) |
| secondary_screen_auc | indication | 339 | 0 | Unknown | 449031 | 65.05885318867793 | Unknown: 449031 (65.1%); breast cancer: 7136 (1.0%); non-small cell lung cancer (NSCLC): 6955 (1.0%); prostate cancer: 4479 (0.6%); corticosteroid-responsive dermatoses: 3950 (0.6%); hypertension: 3736 (0.5%); multiple myeloma: 3601 (0.5%); contraceptive: 3540 (0.5%) |
| secondary_screen_auc | phase | 8 | 0 | Launched | 261454 | 37.881343162482324 | Launched: 261454 (37.9%); Preclinical: 179276 (26.0%); Phase 2: 110144 (16.0%); Phase 3: 62229 (9.0%); Phase 1: 54114 (7.8%); Phase 1/Phase 2: 9829 (1.4%); Withdrawn: 7382 (1.1%); Phase 2/Phase 3: 5764 (0.8%) |
| secondary_screen_ic50 | ccle_tissue | 20 | 0 | LUNG | 69925 | 19.653778697187057 | LUNG: 69925 (19.7%); TRACT: 41029 (11.5%); SKIN: 30814 (8.7%); SYSTEM: 24362 (6.8%); OVARY: 23650 (6.6%); PANCREAS: 22676 (6.4%); OESOPHAGUS: 18969 (5.3%); INTESTINE: 18807 (5.3%) |
| secondary_screen_ic50 | screen_id | 4 | 0 | HTS002 | 317605 | 89.26905088480652 | HTS002: 317605 (89.3%); MTS010: 19909 (5.6%); MTS006: 18195 (5.1%); MTS005: 75 (0.0%) |
| secondary_screen_ic50 | name | 1415 | 0 | selinexor | 924 | 0.25970813752164235 | selinexor: 924 (0.3%); carfilzomib: 919 (0.3%); dinaciclib: 902 (0.3%); napabucasin: 888 (0.2%); bortezomib: 888 (0.2%); P276-00: 876 (0.2%); talazoparib: 848 (0.2%); gambogic-acid: 848 (0.2%) |
| secondary_screen_ic50 | moa | 522 | 0 | EGFR inhibitor | 11059 | 3.108346637285544 | EGFR inhibitor: 11059 (3.1%); Unknown: 11019 (3.1%); HDAC inhibitor: 9777 (2.7%); tubulin polymerization inhibitor: 9377 (2.6%); topoisomerase inhibitor: 8873 (2.5%); CDK inhibitor: 7378 (2.1%); mTOR inhibitor: 7083 (2.0%); HSP inhibitor: 6706 (1.9%) |
| secondary_screen_ic50 | target | 777 | 0 | Unknown | 60456 | 16.992332426416027 | Unknown: 60456 (17.0%); MTOR: 6501 (1.8%); HSP90AA1: 4858 (1.4%); EGFR: 4401 (1.2%); TUBB: 3455 (1.0%); TOP2A: 3333 (0.9%); TOP1: 3259 (0.9%); BCL2: 2912 (0.8%) |
| secondary_screen_ic50 | disease.area | 96 | 0 | Unknown | 257023 | 72.2413037123648 | Unknown: 257023 (72.2%); oncology: 20644 (5.8%); hematologic malignancy: 14967 (4.2%); infectious disease: 14797 (4.2%); neurology/psychiatry: 6404 (1.8%); cardiology: 6210 (1.7%); endocrinology: 4374 (1.2%); dermatology: 3985 (1.1%) |
| secondary_screen_ic50 | indication | 330 | 0 | Unknown | 257023 | 72.2413037123648 | Unknown: 257023 (72.2%); non-small cell lung cancer (NSCLC): 3773 (1.1%); multiple myeloma: 3002 (0.8%); breast cancer: 2665 (0.7%); colorectal cancer: 2514 (0.7%); chronic myeloid leukemia (CML): 1587 (0.4%); cosmetic: 1580 (0.4%); coccidiosis: 1521 (0.4%) |
| secondary_screen_ic50 | phase | 8 | 0 | Launched | 106031 | 29.802070919434264 | Launched: 106031 (29.8%); Preclinical: 102732 (28.9%); Phase 2: 64628 (18.2%); Phase 1: 36535 (10.3%); Phase 3: 33366 (9.4%); Phase 1/Phase 2: 6582 (1.8%); Withdrawn: 3073 (0.9%); Phase 2/Phase 3: 2837 (0.8%) |

## Continuous Feature Summary

| Dataset Mode | Feature | Mean | Std | Min | Median | Max |
| --- | --- | --- | --- | --- | --- | --- |
| gdsc_auc_metadata_plus_expression | RPS4Y1 | 5.957639694213867 | 3.932983160018921 | 2.5090296268463135 | 3.4500861167907715 | 13.423099517822266 |
| gdsc_auc_metadata_plus_expression | KRT19 | 7.618338584899902 | 3.931535482406616 | 2.760674238204956 | 6.534876823425293 | 13.488903999328613 |
| gdsc_auc_metadata_plus_expression | VIM | 9.681628227233887 | 3.7141928672790527 | 2.5991477966308594 | 11.651105880737305 | 13.40769100189209 |
| gdsc_auc_metadata_plus_expression | S100P | 6.304361820220947 | 3.6762192249298096 | 2.8307442665100098 | 3.98797869682312 | 13.729743003845215 |
| gdsc_auc_metadata_plus_expression | TACSTD2 | 5.7733941078186035 | 3.6045141220092773 | 2.6417722702026367 | 3.2780466079711914 | 12.534309387207031 |
| gdsc_auc_metadata_plus_expression | TGFBI | 7.028448581695557 | 3.470491409301758 | 2.942265272140503 | 7.018771171569824 | 13.140789031982422 |
| gdsc_auc_metadata_plus_expression | TM4SF1 | 7.826947212219238 | 3.4641053676605225 | 2.5459959506988525 | 9.155003547668457 | 13.217138290405273 |
| gdsc_auc_metadata_plus_expression | SRGN | 5.857320308685303 | 3.4141886234283447 | 2.5570409297943115 | 3.6230599880218506 | 13.217001914978027 |
| gdsc_auc_metadata_plus_expression | CAV1 | 8.289234161376953 | 3.409468173980713 | 2.83194637298584 | 9.509844779968262 | 13.000051498413086 |
| gdsc_auc_metadata_plus_expression | C19orf33 | 6.211508274078369 | 3.3949992656707764 | 2.2519781589508057 | 4.158551216125488 | 11.956205368041992 |
| gdsc_auc_metadata_plus_expression | DKK1 | 6.5359578132629395 | 3.3795881271362305 | 2.38614559173584 | 6.289144992828369 | 12.615605354309082 |
| gdsc_auc_metadata_plus_expression | KRT8 | 7.269345760345459 | 3.3452179431915283 | 2.7063186168670654 | 8.101147651672363 | 12.627375602722168 |
| gdsc_auc_metadata_plus_expression | SPINT2 | 7.776138782501221 | 3.3433916568756104 | 2.6633214950561523 | 9.262036323547363 | 12.516510009765625 |
| gdsc_auc_metadata_plus_expression | NNMT | 6.375559329986572 | 3.3337299823760986 | 2.679501533508301 | 4.4023213386535645 | 12.697196006774902 |
| gdsc_auc_metadata_plus_expression | EPCAM | 6.329981803894043 | 3.3268368244171143 | 2.590329170227051 | 5.088299751281738 | 11.730050086975098 |
| gdsc_auc_metadata_plus_expression | UCHL1 | 6.657998561859131 | 3.2506864070892334 | 2.8651890754699707 | 5.78371524810791 | 11.896323204040527 |
| gdsc_auc_metadata_plus_expression | MYOF | 8.281031608581543 | 3.2205138206481934 | 2.7679526805877686 | 9.863242149353027 | 12.430731773376465 |
| gdsc_auc_metadata_plus_expression | BEX1 | 5.476546287536621 | 3.220475673675537 | 2.871110677719116 | 3.495903491973877 | 12.838226318359375 |
| gdsc_auc_metadata_plus_expression | IFITM3 | 8.759527206420898 | 3.212038040161133 | 2.765465497970581 | 10.146405220031738 | 13.084348678588867 |
| gdsc_auc_metadata_plus_expression | MAL2 | 6.216457843780518 | 3.192089080810547 | 2.7464141845703125 | 4.778302192687988 | 11.872785568237305 |
| gdsc_auc_metadata_plus_expression | BASP1 | 7.831803798675537 | 3.19050669670105 | 2.8830513954162598 | 9.124274253845215 | 12.744452476501465 |
| gdsc_auc_metadata_plus_expression | SPOCK1 | 6.463677883148193 | 3.1750364303588867 | 2.4936888217926025 | 5.799398899078369 | 12.526537895202637 |
| gdsc_auc_metadata_plus_expression | HSPA1A | 8.691123962402344 | 3.1325488090515137 | 2.6388282775878906 | 10.174039840698242 | 12.448633193969727 |
| gdsc_auc_metadata_plus_expression | SLPI | 6.301070213317871 | 3.111532688140869 | 3.007219076156616 | 4.644059658050537 | 13.424015045166016 |
| gdsc_auc_metadata_plus_expression | DSP | 6.966930866241455 | 3.1099703311920166 | 2.6630561351776123 | 7.995334625244141 | 12.003175735473633 |
| gdsc_auc_metadata_plus_expression | NGFRAP1 | 10.454377174377441 | 3.101646661758423 | 2.9100308418273926 | 11.864027976989746 | 13.177865028381348 |
| gdsc_auc_metadata_plus_expression | GNG11 | 6.359838008880615 | 3.1000401973724365 | 2.7225499153137207 | 5.416900157928467 | 12.359663963317871 |
| gdsc_auc_metadata_plus_expression | HLA-DRA | 4.8624043464660645 | 3.095052719116211 | 2.6679935455322266 | 3.082338571548462 | 12.417120933532715 |
| gdsc_auc_metadata_plus_expression | MGST1 | 8.784002304077148 | 3.093940496444702 | 3.278555154800415 | 10.126422882080078 | 12.769692420959473 |
| gdsc_auc_metadata_plus_expression | FN1 | 6.257328033447266 | 3.089695692062378 | 2.826225519180298 | 5.192789554595947 | 12.74884033203125 |
| gdsc_auc_metadata_plus_expression | CYR61 | 7.522632598876953 | 3.0814075469970703 | 2.702558994293213 | 8.034464836120605 | 12.619583129882812 |
| gdsc_auc_metadata_plus_expression | PXDN | 6.173223495483398 | 3.069859266281128 | 2.674981117248535 | 5.446447372436523 | 12.499319076538086 |
| gdsc_auc_metadata_plus_expression | TPD52L1 | 7.953366279602051 | 3.0654563903808594 | 2.779729127883911 | 8.922797203063965 | 13.065520286560059 |
| gdsc_auc_metadata_plus_expression | SPP1 | 4.967426300048828 | 3.0545425415039062 | 2.7148375511169434 | 3.13242244720459 | 12.808968544006348 |
| gdsc_auc_metadata_plus_expression | ANXA1 | 8.665481567382812 | 3.053858518600464 | 2.5972068309783936 | 10.15343189239502 | 12.81813907623291 |
| gdsc_auc_metadata_plus_expression | LGALS3 | 9.57958984375 | 3.0456502437591553 | 2.84515380859375 | 10.887514114379883 | 13.234183311462402 |
| gdsc_auc_metadata_plus_expression | BST2 | 8.034910202026367 | 3.0453524589538574 | 2.893649101257324 | 8.944389343261719 | 12.49095630645752 |
| gdsc_auc_metadata_plus_expression | LGALS1 | 9.62414836883545 | 3.036083221435547 | 2.5923497676849365 | 10.885725021362305 | 13.226639747619629 |
| gdsc_auc_metadata_plus_expression | S100A14 | 5.186830520629883 | 3.027301549911499 | 2.6668457984924316 | 3.2352325916290283 | 11.95205307006836 |
| gdsc_auc_metadata_plus_expression | KRT7 | 6.184670925140381 | 3.0256400108337402 | 2.9099767208099365 | 4.629947185516357 | 12.21860408782959 |
| gdsc_auc_metadata_plus_expression | PRSS23 | 7.661566734313965 | 3.0250136852264404 | 2.9216721057891846 | 8.469679832458496 | 12.781046867370605 |
| gdsc_auc_metadata_plus_expression | LCN2 | 5.278237342834473 | 3.0212209224700928 | 2.8006532192230225 | 3.5157830715179443 | 13.279764175415039 |
| gdsc_auc_metadata_plus_expression | IFI27 | 6.764298439025879 | 3.012885332107544 | 3.5987308025360107 | 4.999249458312988 | 13.190658569335938 |
| gdsc_auc_metadata_plus_expression | NUPR1 | 6.774687767028809 | 3.0123116970062256 | 2.8096365928649902 | 6.324773788452148 | 12.821274757385254 |
| gdsc_auc_metadata_plus_expression | ANXA3 | 6.1660590171813965 | 3.0106189250946045 | 2.2876462936401367 | 6.117022514343262 | 11.413873672485352 |
| gdsc_auc_metadata_plus_expression | GDF15 | 7.457473278045654 | 3.0067484378814697 | 3.0675508975982666 | 7.635049819946289 | 13.336289405822754 |
| gdsc_auc_metadata_plus_expression | MIR205HG | 4.721045970916748 | 2.998610496520996 | 2.7656381130218506 | 3.151385545730591 | 12.900615692138672 |
| gdsc_auc_metadata_plus_expression | S100A16 | 7.2738037109375 | 2.9829723834991455 | 2.6725754737854004 | 8.645462989807129 | 12.205534934997559 |
| gdsc_auc_metadata_plus_expression | AKR1C1 | 5.562788963317871 | 2.9618988037109375 | 2.745384931564331 | 3.9512453079223633 | 12.95825481414795 |
| gdsc_auc_metadata_plus_expression | BEX4 | 6.1324567794799805 | 2.9375970363616943 | 2.512054443359375 | 5.621893882751465 | 12.076676368713379 |
| gdsc_auc_metadata_plus_expression | TSPAN8 | 4.82459831237793 | 2.9310011863708496 | 2.6874654293060303 | 3.2609729766845703 | 13.189949989318848 |
| gdsc_auc_metadata_plus_expression | UCA1 | 5.138784885406494 | 2.930307626724243 | 2.730827808380127 | 3.52091908454895 | 12.980490684509277 |
| gdsc_auc_metadata_plus_expression | SPARC | 5.773438453674316 | 2.926039457321167 | 2.9624993801116943 | 3.9825170040130615 | 12.170112609863281 |
| gdsc_auc_metadata_plus_expression | EFEMP1 | 5.672320365905762 | 2.9247167110443115 | 2.8157215118408203 | 3.8473145961761475 | 12.052190780639648 |
| gdsc_auc_metadata_plus_expression | HLA-DPA1 | 5.461180210113525 | 2.894345283508301 | 3.110402822494507 | 3.8241803646087646 | 12.828682899475098 |
| gdsc_auc_metadata_plus_expression | ESRP1 | 5.674382209777832 | 2.8799004554748535 | 2.630388021469116 | 3.7770042419433594 | 11.013885498046875 |
| gdsc_auc_metadata_plus_expression | LCP1 | 5.445532321929932 | 2.8674027919769287 | 2.5579512119293213 | 3.890826463699341 | 11.562132835388184 |
| gdsc_auc_metadata_plus_expression | IGFBP3 | 5.794554233551025 | 2.864302396774292 | 2.6662824153900146 | 4.647515296936035 | 12.889054298400879 |
| gdsc_auc_metadata_plus_expression | GYPC | 5.306827545166016 | 2.858644962310791 | 2.534872531890869 | 3.382350206375122 | 11.7715482711792 |
| gdsc_auc_metadata_plus_expression | RAB25 | 5.129763126373291 | 2.857970952987671 | 2.6124696731567383 | 3.2106072902679443 | 11.148176193237305 |
| gdsc_auc_metadata_plus_expression | CYBA | 9.205462455749512 | 2.8279285430908203 | 2.9648163318634033 | 10.406840324401855 | 12.63277530670166 |
| gdsc_auc_metadata_plus_expression | CAV2 | 7.235850811004639 | 2.8226752281188965 | 2.8518526554107666 | 7.912180423736572 | 12.346351623535156 |
| gdsc_auc_metadata_plus_expression | ALDH1A1 | 5.289076805114746 | 2.8059849739074707 | 2.963803291320801 | 3.6021382808685303 | 13.100794792175293 |
| gdsc_auc_metadata_plus_expression | TUBB2B | 6.354617118835449 | 2.798628807067871 | 3.018334150314331 | 5.200283050537109 | 12.777716636657715 |
| gdsc_auc_metadata_plus_expression | CLEC2B | 6.221967697143555 | 2.7842557430267334 | 2.7962381839752197 | 5.719377517700195 | 12.876567840576172 |
| gdsc_auc_metadata_plus_expression | TFPI | 6.438048362731934 | 2.7809109687805176 | 3.000391721725464 | 6.194180965423584 | 12.297420501708984 |
| gdsc_auc_metadata_plus_expression | TSPAN1 | 6.324243545532227 | 2.7775909900665283 | 3.1605825424194336 | 4.825840473175049 | 12.290548324584961 |
| gdsc_auc_metadata_plus_expression | ARHGDIB | 6.720728397369385 | 2.7761306762695312 | 3.1749792098999023 | 6.022054672241211 | 12.346978187561035 |
| gdsc_auc_metadata_plus_expression | GMFG | 4.587247371673584 | 2.740093469619751 | 2.5387930870056152 | 3.2284817695617676 | 11.9259614944458 |
| gdsc_auc_metadata_plus_expression | LAPTM5 | 4.5122971534729 | 2.738860607147217 | 2.658022403717041 | 3.093411922454834 | 12.014384269714355 |
| gdsc_auc_metadata_plus_expression | GPX2 | 4.841357231140137 | 2.7357282638549805 | 2.7987608909606934 | 3.404573917388916 | 12.424222946166992 |
| gdsc_auc_metadata_plus_expression | AKR1C2 | 4.881467819213867 | 2.7305493354797363 | 2.355001449584961 | 3.314870595932007 | 11.924213409423828 |
| gdsc_auc_metadata_plus_expression | CXCR4 | 4.840536117553711 | 2.707324266433716 | 2.7179648876190186 | 3.1749377250671387 | 12.184173583984375 |
| gdsc_auc_metadata_plus_expression | S100A6 | 9.90392780303955 | 2.7044789791107178 | 3.382136344909668 | 11.22918701171875 | 12.952367782592773 |
| gdsc_auc_metadata_plus_expression | TMSB15A | 5.517031192779541 | 2.697908878326416 | 2.7472946643829346 | 4.114070415496826 | 12.654022216796875 |
| gdsc_auc_metadata_plus_expression | TESC | 5.382665634155273 | 2.697286367416382 | 2.6049859523773193 | 3.8670380115509033 | 12.0646333694458 |
| gdsc_auc_metadata_plus_expression | CTGF | 6.095944881439209 | 2.692009687423706 | 2.6382949352264404 | 5.949387550354004 | 12.199647903442383 |
| gdsc_auc_metadata_plus_expression | SNAI2 | 6.369839668273926 | 2.6918091773986816 | 3.006375789642334 | 5.856529712677002 | 12.121910095214844 |
| gdsc_auc_metadata_plus_expression | GAL | 5.172340393066406 | 2.691579818725586 | 2.6276917457580566 | 3.509040117263794 | 12.907052993774414 |
| gdsc_auc_metadata_plus_expression | VAMP8 | 8.438684463500977 | 2.67682147026062 | 2.640202045440674 | 9.73159122467041 | 12.0852632522583 |

For complete continuous-feature statistics, see `continuous_feature_summary.csv`.

## How Categorical Features Become Numerical

Linear regression cannot directly consume strings such as tissue names or drug targets. The pipeline first imputes missing categorical values with the most frequent value, then applies one-hot encoding. One-hot encoding creates one binary indicator column per category value. For example, if `Screen Medium` has values `R` and `D/F12`, it becomes two numerical columns indicating which value each row has.

After this preprocessing, Linear Regression, Ridge, Lasso, Linear SVR, and the other models receive a numerical design matrix.

# Verified Results and Evaluation Notes

This project includes a training/evaluation notebook, a separate data-preparation notebook, shared inference code, a desktop demo, and small regression tests. The checkpoints are preserved from the previous full training experiment. The simplified implementation was checked separately; it was not retrained for a new full 60-epoch schedule.

## Historical full training experiment

Both detectors started from YOLOv8n pretrained on COCO. The plate detector trained for 60 epochs. Character training stopped early after 23 epochs, with the best checkpoint from epoch 11. The selected checkpoint hashes were verified.

| Metric | Verified result | Scope |
|---|---:|---|
| Filtered test images | 4,113 | Full source images |
| Plate detection mAP50 | 98.36% | Test |
| Plate detection mAP50-95 | 77.53% | Test |
| Plate precision at the validation-selected fixed threshold | 94.56% | Full pipeline on test |
| Plate recall at the same threshold | 95.32% | Full pipeline on test |
| Exact full-plate text accuracy | 4,160 / 4,848 = 85.81% | Test plates with XML-derived text references |
| Character error rate (CER) | 6.42% | Full pipeline on eligible test plates |
| Character detection mAP50 | 87.48% | Test reference plate crops |
| HOG + SVM character accuracy | 89.43% | 18,945 validation character crops |
| HOG + SVM macro F1 | 0.7511 | Validation character classification |

The historical metric table is preserved in [assets/verified_metrics.csv](assets/verified_metrics.csv). Precision/recall in the standalone YOLO detection rows summarize its evaluation curves; full-pipeline precision/recall use the fixed threshold selected on validation. They are different reporting scopes.

The SVM was fitted on 132,114 training character crops. Its `C` was selected using validation, so the validation score is not an independent test result. It receives reference character boxes, making its accuracy unsuitable for direct comparison with exact full-plate recognition accuracy. The fitted SVM and shared feature preprocessing were preserved in the simplified version.

**No new full training result is claimed for the simplified notebooks.** Short runs establish pipeline health, not model quality. New full training requires `MODE="train"` and saves models separately from the existing checkpoints.

## Evaluation definitions and limitations

- The full pipeline detects plates in complete images and passes predicted plate crops to the character detector. Standalone character metrics instead use reference plate crops.
- Reference text is reconstructed from XML character labels in spatial reading order. It is not an independently transcribed ground-truth text dataset and can be incomplete.
- Exact text accuracy requires every character of an eligible reference plate to match. Missed eligible plates remain errors rather than being dropped from evaluation.
- Plates without character annotations are excluded from the text-accuracy denominator and counted separately. Extra detections affect detection precision; they do not directly enter that text-accuracy denominator.
- CER is total edit distance divided by the total number of reference characters.
- Blurred images, small plates, rare classes, incomplete labels, and possible vehicle/scene overlap remain limitations. Duplicate/similarity filtering does not prove scene-level independence.
- **The historical test set has already been consumed.** A code rewrite or another evaluation on that set does not produce a fresh independent test estimate. A new independent holdout is required for final claims about future improvements.

![Verified examples from the historical experiment](assets/verified_examples.png)

## Verification of the simplified implementation

| Check | Observed result |
|---|---|
| Default notebook Run All | Passed in the project's real Jupyter kernel in `saved` mode, without training or reevaluating test |
| Full JPG/XML rescan | Image records, annotations, and retained-image selection exactly matched the previous verified audit |
| Duplicate/similarity filtering | The simplified method matched the previous selection over the full dataset; this does not establish vehicle/scene independence |
| Plate YOLO smoke training | One epoch on 16 training and four validation images; losses were finite and weights were saved |
| Character YOLO smoke training | One epoch on 19 training and six validation plate crops; losses were finite and weights were saved |
| Smoke evaluation flow | Validation ran; test did not run; provisional smoke thresholds are not deployment recommendations |
| Small SVM run | Training and validation completed; all validation predictions matched after save/reload |
| Threshold-selection and final-evaluation branches | Exercised with existing weights and an artificial split of nine original validation images; the original test set was not used and no new quality score is claimed |
| Inference consistency | Plate boxes and text matched the previous implementation on 32 evenly spaced validation images |
| SVM preprocessing consistency | All 324 HOG features matched exactly for each of 64 inspected character crops |
| Regression tests | Six tests passed |
| Desktop demo | Automated hidden-window checks passed image selection, background processing, display, and PNG/CSV saving |
| Dependency consistency | `pip check` reported no conflicts in the project environment |

Local verification logs are in `outputs/verify_*.log`; the short training run is in `outputs/smoke_20261015_162653`. One-time maintenance tools are retained locally under `archive/maintenance`. These folders are excluded from the public repository and are not runtime dependencies. Saved-mode output tables and images are embedded in the main notebook; selected public evidence is in `assets`.

## Separating data preparation — 2026-10-16

JPG/XML validation and similarity grouping were moved to `prepare_data.ipynb` without changing the two functions' logic at that refactoring step. The main notebook reads `prepared_data` tables and reports a clear error if they are missing. YOLO input conversion and source hash checks remain next to training.

At that step, the previously verified tables were copied byte-for-byte from `outputs/audit` to `prepared_data`; the expensive full audit was not repeated during the initial separation. The retained selection stayed at 13,851 training, 2,052 validation, and 4,113 test images.

The preparation/CSV path was exercised on a small synthetic dataset covering corrupt images, missing pairs, zero-width boxes, stale XML names/dimensions, and exact cross-split duplicates. Test-first retention, repeatability, source preservation at that stage, loading from the main notebook, and the missing-index message passed. Local log: `outputs/verify_data_split.log`.

All main-notebook cells then ran in saved mode through a real Jupyter kernel, with outputs captured in the notebook. Six inference regression tests also passed. Logs include `outputs/verify_kernel_data_split.log`. No training or test reevaluation was performed for that separation.

## Physical duplicate archiving — 2026-10-16

Preparation was subsequently extended to archive surplus image/XML pairs and executed on the complete local dataset. Its retained selection exactly matched all 20,016 previously retained images:

| Split | Retained images |
|---|---:|
| Train | 13,851 |
| Validation | 2,052 |
| Test | 4,113 |

Across 714 duplicate/similarity groups, 950 surplus JPGs and their 950 XML files were moved to `duplicate_photos/20261016_205444_228805`. The archive also contains copies of the 714 retained JPG/XML pairs, labeled `original`; their source files remain in the dataset. Group membership can reflect approximate similarity rather than identical pixels.

Every backup hash was verified before deleting a surplus source. After transfer, verification checked every manifest destination hash, removal of surplus source paths, and the JPG/XML hashes of all 20,016 retained images. The invalid pair with a zero-width box was reported and left in place; this step does not resolve every annotation or scene-dependence problem.

Temporary-data checks covered retained originals, naming, paired transfers, no-duplicate behavior, and stopping without source deletion on a modified source, copy error, corrupt backup, or source path outside the dataset. The main notebook passed in saved mode after transfer, and the six inference tests passed. No training or test reevaluation was performed.

Local records:

- `outputs/verify_duplicate_archive.log`
- `outputs/run_duplicate_preparation.log`
- `outputs/verify_real_duplicate_archive.log`
- `outputs/verify_saved_duplicate_archive.log`
- `duplicate_photos/20261016_205444_228805/manifest.csv`

The manifest records old/new paths, hashes, and transfer status. The archive and local logs are excluded from Git.

## Perspective correction

Perspective correction is optional and off by default. A previous experimental method reduced exact text accuracy on the complete validation set from 85.36% to 84.82%; it was not adopted as a general improvement.

For the tilted diagnostic image `day_00200.jpg`, the simplified implementation changed the ordinary prediction `64ط753558` to `64ب75355` with correction enabled. This was a targeted inspection of a user-selected, already-known test image, not evidence of overall improvement or a new test-set benchmark.

Retained source file contents and trained checkpoint contents were preserved. Surplus source copies are recoverable through the duplicate archive manifest described above.

## Compact character-box preview — 2026-10-17

The shared renderer used by the desktop demo and notebook now insets each character rectangle by 8% per edge, reducing its displayed width and height by 16%. Leader lines start from the adjusted lower edge. This is a display-only change: original model coordinates, text, confidence scores, and evaluation calculations are unchanged. It does not improve or remeasure recognition accuracy. The README preview was regenerated with the existing checkpoints.

All 18 published files were compared with the publication snapshot before this update. The six inference regression tests passed against the publication code. The demo's image-selection, worker, rendering, and PNG/CSV-save flow passed an automated hidden-window check. All main-notebook code cells passed in saved mode through a real Jupyter kernel, refreshing the embedded output preview without training or reevaluating test. Local records: `outputs/verify_demo_compact_boxes.log` and `outputs/verify_public_compact_boxes.log` (not included in Git).

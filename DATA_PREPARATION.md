# Duplicate and Similar Image Review

Preparation **modifies the source dataset** by archiving surplus image/XML pairs after verified copies have been made. It retains one member per similarity group, preferring test, then validation, then train. This is a similarity-based safeguard, not a guarantee of independent vehicles or scenes.

Each run gets a separate folder. Example naming:

```text
duplicate_photos/<run>/group_000123/
    day_00123_original_test.jpg
    day_00123_original_test.xml
    day_00987_duplicate1_train.jpg
    day_00987_duplicate1_train.xml
    day_00456_duplicate2_validation.jpg
    day_00456_duplicate2_validation.xml
```

These names are illustrative. `original` means the retained group member, not necessarily the first camera image or the best-quality image. Its source files stay in the dataset and are also copied into the archive. Other group members, including approximately similar images, are copied with their XML files. Source and destination hashes are verified before each surplus source file is removed.

The run's `manifest.csv` records old/new paths, role, split, hash, and status for each file. `copied` indicates a verified copy; `moved` indicates a surplus source file has been removed. If interrupted, inspect the manifest and both locations before rerunning. Completed removals are not automatically rolled back; the backups remain. To restore files, copy archived destinations back to their recorded source paths and rerun preparation.

The first prepared index retains excluded rows with `keep=0`; their old source paths no longer exist after transfer. Subsequent runs rescan the cleaned source folders and leave previous archives intact. Invalid image/annotation pairs are reported, not moved into the duplicate archive. Duplicate removal does not fix all annotation or data-quality problems.


[Back to setup and execution](README.md)

# Package verification

- All packaged Python files parsed successfully.
- Five existing edge/property checks passed.
- All three summary/figure analysis scripts ran successfully on archived results.
- Raw counts checked: main 3,600 rows; independent-seed 1,800; sequential 4,320 method/version rows representing 1,440 versions; topology 216; guard sweep 810 rows representing 162 controls.
- No LaTeX, BibTeX, manuscript PDF, bytecode, or git history is included.

This packaging check regenerates summaries; it does not rerun every scientific experiment or timing measurement. Archived raw numerical results and configurations are retained. Timing values from regeneration depend on the execution environment.

Title-aligned package: renamed evolution modules and reran syntax checks, five property checks, and summary regeneration successfully. Scientific experiment timings were not rerun for this documentation update.

Layout validation: source moved to src/, outputs to results/, and documentation to docs/. Updated path resolution passed five existing checks and complete archived-data summary regeneration. Full scientific measurements were not rerun.

# TandemFoilSet Split Design (v2)

## Dataset: File-by-File

| File | Name | Domain | n | Avg Nodes | Front Foil NACA M | Rear Foil | Re | AoA | Ground |
|------|------|--------|---|-----------|-------------------|-----------|-----|-----|--------|
| 0 | raceCar single | Land, inverted | 899 | 85K | M=2-9 + 5 specials | None | 100K-5M | [-10, 0] | Yes |
| 1 | raceCar tandem P1 | Land, inverted | 300 | 124K | M=2-5 | 30 shared | 1M-5M | [-10, 0] | Yes |
| 2 | raceCar tandem P2 | Land, inverted | 300 | 129K | **M=6-8** | 30 shared | 1M-5M | [-10, 0] | Yes |
| 3 | raceCar tandem P3 | Land, inverted | 300 | 130K | M=9 + CH10,E423,FX74,LA5055,S1210 | 30 shared | 1M-5M | [-10, 0] | Yes |
| 4 | cruise tandem P1 | Aerial, freestream | 300 | 210K | M=0-2 | 30 shared | 110K-5M | [-5, +6] | No |
| 5 | cruise tandem P2 | Aerial, freestream | 300 | 209K | **M=2-4** | 30 shared | 110K-5M | [-5, +6] | No |
| 6 | cruise tandem P3 | Aerial, freestream | 300 | 211K | M=4-6 | 30 shared | 110K-5M | [-5, +6] | No |

**Key structural facts:**
- Within each domain (raceCar/cruise), all tandem Part files share the **same 30 rear foils**. Only the front foil varies between Parts.
- Front foil NACA camber (M) is partitioned across Parts with no overlap.
- raceCar uses inverted foils with ground effect (slip wall BC). Cruise has freestream on all boundaries.
- File 3 contains 150 NACA 9xxx + 150 non-NACA specials. Non-NACA foils return (0,0,0) from parse_naca().

## Split Allocation

The goal behind th split is to be able to say:
- The model is good/bad at generalizing to unseen geometries (train on low + high camber, val/test on moderate)
- The same model works across different Reynolds numbers for both race car and cruise (train on low, med, high Re, val/test on low, moderate, high) - random split across all Re numbers
- General single-foil random split as a sanity check


| File | Train | Val | Test | Holdout Reason |
|------|-------|-----|------|----------------|
| 0 (single) | 599 (67%) | 100 (11%) | 200 (22%) | Random holdout for in-dist sanity check |
| 1 (rc P1) | ~225 (75%) | ~25 (8%) | ~50 (17%) | Partial: stratified Re holdout for val_re_rand |
| 2 (rc P2) | 0 (0%) | 100 (33%) | 200 (67%) | **Full holdout**: unseen front foil camber M=6-8 |
| 3 (rc P3) | ~225 (75%) | ~25 (8%) | ~50 (17%) | Partial: stratified Re holdout for val_re_rand |
| 4 (cruise P1) | ~225 (75%) | ~25 (8%) | ~50 (17%) | Partial: stratified Re holdout for val_re_rand |
| 5 (cruise P2) | 0 (0%) | 100 (33%) | 200 (67%) | **Full holdout**: unseen front foil camber M=2-4 |
| 6 (cruise P3) | ~225 (75%) | ~25 (8%) | ~50 (17%) | Partial: stratified Re holdout for val_re_rand |

## Val/Test Splits

All splits are **balanced**: 100 val + 200 test each. Scored as equal-weight average Surface MAE.

| Split | Source | Selection | What It Tests |
|-------|--------|-----------|---------------|
| val_single_in_dist | File 0 (random) | Random holdout | Sanity check: can the model reproduce single-foil flow it has seen? |
| val_geom_camber_rc | File 2 (full holdout) | File-level | Camber interpolation: train sees M=2-5 and M=9, must predict M=6-8 |
| val_geom_camber_cruise | File 5 (full holdout) | File-level | Camber interpolation: train sees M=0-2 and M=4-6, must predict M=2-4 |
| val_re_rand | Files 1+3+4+6 (stratified) | Every 4th sample sorted by Re | Cross-regime generalization: random sample across full Re range from all tandem training domains |

## Totals

| | Samples | % |
|---|---|---|
| Train | 1499 | 55.5% |
| Val (4 x 100) | 400 | 14.8% |
| Test (4 x 200) | 800 | 29.6% |
| **Total** | **2699** | **100%** |

## Design Rationale

**Why full file holdouts for geometry?**
Files 2 and 5 contain front foil shapes with NACA camber values (M=6-8, M=2-4) that are completely absent from training. This is the cleanest possible separation — no leakage, no soft boundaries.

**Why stratified Re sampling instead of threshold cuts?**
We want to show a single model works across all Reynolds numbers. A stratified random holdout across the full Re range tests whether the model learns a smooth Re-dependent representation.

**Why no AoA split?**
Simplicity. AoA extremes (deep stall, near-zero loading) are interesting but add complexity without a clean physical boundary. AoA is implicitly tested within each split since AoA is uniformly distributed across the holdout samples.

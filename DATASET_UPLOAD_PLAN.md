# Dataset Upload Plan

## Data Inventory

| Dataset | Path | Contents | ~Size |
|---------|------|----------|-------|
| H track | `objaverse_60k_120h/videos_curve/` | Condition videos per GLB: albedo.mp4, position.mp4, normal.mp4, rgb.mp4, depth.mp4, mask.mp4 | TBD |
| V track | `objaverse_60k_120v/videos_curve/` | Same as H, plus `position_flip.mp4` (corrected orientation) | TBD |

Structure: `{bucket}/{uuid}/{file}.mp4`, e.g. `000-027/0b716e33ffae462d818c957744f7ce17/albedo.mp4`

## Key Issue: V Track position_flip → position.mp4

V track `position.mp4` was rendered with incorrect camera orientation. The corrected version is `position_flip.mp4`.

**Decision**: Upload only `position_flip.mp4` but **rename it to `position.mp4`** on the server. This way downstream code can reference `position.mp4` uniformly for both H and V tracks, and we don't carry the "flip" naming debt into the public dataset.

## Upload Strategy

### Option A: Flat upload (simple)
Upload all files preserving directory structure to a single HF dataset repo.

```
Yuehavingfun/ink3d-train-data/
├── h/
│   └── videos_curve/
│       └── 000-000/
│           └── {uuid}/*.mp4
└── v/
    └── videos_curve/
        └── 000-000/
            └── {uuid}/*.mp4 (with position_flip)
```

**Pros**: Simple, one command
**Cons**: Large single upload, hard to update incrementally

### Option B: Bucketed upload (recommended)
Upload by bucket (000-000 through 000-NNN), each as a separate upload session. Resume-friendly.

```
Yuehavingfun/ink3d-train-data/
├── h/000-000/
├── h/000-001/
├── ...
├── v/000-000/
├── v/000-001/
└── ...
```

**Pros**: Resumable, parallelizable
**Cons**: More complex orchestratiom

### Option C: Same repo as example data
Not recommended — example data and training data serve different purposes.

## Recommended Plan

1. **Create dataset repo**: `Yuehavingfun/Objaverse-PBR-render`
2. **Upload H track**: `videos_curve/` only, bucket by bucket (000-000 to 000-NNN), parallel
3. **Upload V track**: Same structure, `position_flip.mp4` renamed to `position.mp4` during upload
4. **Upload CSVs**: `position2albedo_all_321.csv`, `position2albedo_all_321_hv.csv`, paired CSV
5. **Skip**: `images_curve/` (reference images) — upload later if needed
6. **Verify**: Spot-check 5 random samples for file count and integrity

## Files Per Sample

### H track
```
{videos_curve}/{bucket}/{uuid}/
├── albedo.mp4
├── position.mp4
├── normal.mp4
├── rgb.mp4
├── depth.mp4
└── mask.mp4
```

### V track
```
{videos_curve}/{bucket}/{uuid}/
├── albedo.mp4
├── position.mp4        # renamed from position_flip.mp4 (corrected orientation)
├── normal.mp4
├── rgb.mp4
├── depth.mp4
└── mask.mp4
```

## Decisions Made

- ✅ Repo: `Yuehavingfun/Objaverse-PBR-render`
- ✅ Upload: `videos_curve/` only, no `images_curve/`
- ✅ V track: `position_flip.mp4` → renamed to `position.mp4`
- ✅ Strategy: Bucket-by-bucket (121 buckets), parallel where possible

## Open Questions

1. Total size — H: ~71 GB, V: ~71 GB. Combined ~142 GB. HF free tier limit?
2. Parallel workers — how many simultaneous uploads?
3. Resume strategy — check existing files before uploading?
4. Bucket naming — flat `000-000/` or prefix `h/000-000/` + `v/000-000/`?

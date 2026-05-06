# Two-Model Possession Labeling Design

Date: 2026-05-06
Status: Proposed
Scope: Phase 1 only (`labeling -> export -> train`)

## Goal

Build a reliable first-phase possession dataset workflow for basketball video that can answer two separate questions:

1. What is the global ball state in this frame?
   - `owned`
   - `air`
   - `loose`
2. If the ball is owned, what is the owner interaction subtype?
   - `control`
   - `dribble`
   - `shot`
   - `contested`

This phase does not integrate the new models into the final game pipeline yet. It focuses on producing high-quality labeled data, reproducible exports, and reproducible training for both models.

## Why Two Models

Possession has two different prediction problems:

- A frame-level ball-state problem: distinguish `owned` from `air` and `loose`.
- A candidate-level player-ball problem: when the ball is owned, decide which nearby player actually owns it and what kind of possession it is.

Keeping these as separate models has three advantages:

- It matches how the data is naturally labeled.
- It avoids forcing a single classifier to solve incompatible contexts.
- It fits the current repository structure, which already has player-ball crop generation and temporal possession logic.

## Existing Code We Will Reuse

- `tools/annotate_possession_dataset.py`
- `tools/export_possession_crops.py`
- `tools/train_possession_classifier.py`
- `basketball_cv/possession.py`
- `basketball_cv/possession_balanced.py`
- `basketball_cv/possession_model.py`

The design keeps one annotation truth source and extends the current scripts instead of creating parallel possession tools.

## Architecture

The new workflow is:

1. Annotate frames into a structured manifest.
2. Export dataset A for ball state.
3. Export dataset B for owner subtype.
4. Train model A on ball-state crops.
5. Train model B on player-plus-ball crops.
6. Leave integration with the runtime possession stack for a later phase.

The truth source is the manifest, not folder names and not weak labels.

## Data Model

The canonical annotation artifact is:

- `datasets/possession_labels/manifest.jsonl`

Each line represents one labeled frame and includes:

- `video`
- `frame_index`
- `split_hint`
- `ball_state` in `{owned, air, loose}`
- `owner_player_id` or `null`
- `owner_track_id` or `null`
- `owner_team` or `null`
- `owner_state` in `{control, dribble, shot, contested}` or `null`
- `ball_bbox`
- `candidate_players`
- `flags`

`candidate_players` stores the candidates shown at annotation time with enough metadata to reproduce crops and audit mistakes:

- `player_id`
- `track_id`
- `team`
- `jersey_number`
- `bbox`
- `rank`

`flags` is a free-form list constrained to known labels such as:

- `uncertain`
- `occluded`
- `ball_not_visible_cleanly`
- `candidate_missing`
- `reviewed_in_cvat`

## Annotation UX

The annotation flow remains local-first and frame-sampled in this first phase.

### Step A: Ball State

For each sampled frame, the annotator asks for the global ball state:

- `o` -> `owned`
- `a` -> `air`
- `l` -> `loose`

### Step B: Owner Selection

Only when `ball_state == owned`:

1. Show numbered candidate players near the active ball.
2. Select owner with `1-9` or mouse click.
3. Label subtype:
   - `c` -> `control`
   - `d` -> `dribble`
   - `s` -> `shot`
   - `t` -> `contested`

### Important Behavior Rules

- `air` and `loose` never force an owner.
- `owner_state` must be `null` unless `ball_state == owned`.
- The manifest, not filenames, stores the annotation truth.
- Crops are derived artifacts and may be regenerated at any time.

## Sampling Strategy

The first version uses frame sampling, not keyframe interpolation.

Sampling should support both:

- simple periodic sampling
- priority sampling for harder frames

Priority sampling favors:

- heuristic ownership transitions
- frames with multiple close candidates
- unstable ball-state regions
- brief occlusions and crossings

This keeps the MVP simple while still biasing the dataset toward the cases where current possession logic fails.

## Export Design

`tools/export_possession_crops.py` becomes manifest-driven and supports two tasks.

### Task A: `ball-state`

Input:

- manifest rows for all labeled frames

Output crop:

- ball-centered crop with configurable context padding

Classes:

- `owned`
- `air`
- `loose`

Output layout:

- `datasets/possession_ball_state/train/<class>/...`
- `datasets/possession_ball_state/val/<class>/...`

### Task B: `owner-state`

Input:

- manifest rows where `ball_state == owned`

Output crop:

- crop spanning selected player and ball with configurable margin

Classes:

- `control`
- `dribble`
- `shot`
- `contested`
- `no_control`

Export behavior:

- Export the labeled owner candidate with its positive class.
- Export nearby non-owner candidates as `no_control`.
- Skip synthetic owner labels for `air` and `loose` frames.

Output layout:

- `datasets/possession_owner_state/train/<class>/...`
- `datasets/possession_owner_state/val/<class>/...`

### Export Metadata

Each export also writes a derived manifest for debugging, including:

- `video`
- `frame_index`
- `task`
- `label`
- `player_id`
- `track_id`
- `team`
- `jersey_number`
- `flags`
- `source_manifest_row`

## Train/Val Splitting

Split by video or video segment, not by random frame.

This avoids leakage where nearly identical adjacent frames appear in both train and validation.

The exporter supports:

- explicit `--val-video-stems`
- ratio-based assignment grouped by video

The same source video must not land in both train and val unless the split is explicitly segment-based and documented.

## Training Design

`tools/train_possession_classifier.py` is extended to support both tasks through explicit task selection rather than parallel scripts.

Supported training tasks:

- `--task ball-state`
- `--task owner-state`

Expected defaults:

- `ball-state` uses a small image classifier base model
- `owner-state` uses a small image classifier base model

Training stays reproducible by accepting explicit:

- dataset root
- base model
- device
- epochs
- image size
- batch size

This phase does not add runtime fusion logic. It only trains the models and saves standard YOLO classification outputs.

## Review and External Tools

Roboflow and CVAT are support tools, not parallel truth sources.

They are used for:

- reviewing difficult examples
- correcting queued ambiguous samples
- accelerating batch cleanup

The local manifest remains canonical. If a sample is reviewed externally, the corrected result is reimported into the same manifest model, not stored as a second official dataset definition.

## Quality Controls

Before training, validation checks must confirm:

- `owner_state` is present only when `ball_state == owned`
- `air` and `loose` have no owner assigned
- any referenced owner exists in that frame's candidate set
- output class folders are non-empty where expected
- train/val split obeys the configured isolation rules

The exporter should also support filtering out uncertain samples for first-pass training while keeping them in the canonical manifest for later review.

## Success Criteria

Phase 1 is successful when we can:

1. label multiple videos with the two-step local annotator
2. store truth in one canonical manifest
3. export reproducible datasets for both possession tasks
4. train both models without manual dataset surgery
5. audit any training sample back to its source frame and annotation

## Out of Scope

This phase does not:

- integrate the two-model decision stack into `tools/run_game_pipeline.py`
- replace the current heuristic possession runtime
- add keyframe interpolation annotation
- redesign the final render HUD
- solve pass detection end to end

Those belong to the next phase, once the labeling and training foundation is stable.

## Testing Strategy

We will add automated checks around:

- manifest schema validation
- annotation invariants
- export label correctness
- split correctness
- empty-class failure detection

We will also do one manual verification pass on a representative video run to confirm:

- the annotation UX is fast enough to use repeatedly
- `air` and `loose` are clearly separated in practice
- owner candidates are generated well enough to support labeling without excessive manual friction

## Risks and Mitigations

### Risk: `air` vs `loose` remains ambiguous in many frames

Mitigation:

- allow `uncertain` flagging
- prioritize clean samples for the first model version
- send ambiguous frames to review instead of forcing noisy labels

### Risk: Candidate generation misses the true owner

Mitigation:

- preserve `candidate_missing` as a manifest flag
- keep crops reproducible from manifest rows
- refine candidate generation later without losing labeled history

### Risk: Dataset drift between local and external review tools

Mitigation:

- keep one canonical manifest
- treat external tools as review surfaces only

### Risk: Leakage between training and validation

Mitigation:

- split by video or explicit segment boundary
- validate split isolation automatically during export

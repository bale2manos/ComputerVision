# Single Pipeline, Stable Jersey, and Short-Gap Interpolation Design

Date: 2026-05-05
Project: Basketball CV
Primary reference clip: `subra.mp4`

## Summary

This design simplifies the project to one official pipeline and improves two high-value behaviors:

1. Stable jersey identity across the full clip.
2. Short-gap continuity when a player disappears for a few frames.

The system should stop switching between parallel OCR/render paths and instead produce one consistent result from tracking to final video. Jersey OCR should use evidence collected across the full player identity, not only local fragments. Short tracking gaps should be filled conservatively so players do not appear to teleport in the rendered video or in the 2D court view.

## Goals

- Keep one official pipeline for analysis, OCR, possession, passes, and final render.
- Remove the combined OCR path that currently introduces optimistic and unstable jersey assignments.
- Stabilize jersey numbers using all valid samples from the same player identity across the video.
- Support hybrid jersey locking:
  - unresolved players remain `P{id}`;
  - once a jersey is strongly supported, it can be backfilled;
  - conflicting segments are preserved instead of being overwritten.
- Fill short player gaps with estimated boxes in the video and estimated positions in the 2D court.
- Smooth 2D motion during short gaps and short bursts of jitter.
- Keep estimated frames out of OCR evidence and out of identity creation logic.

## Non-Goals

- Full replacement of the main tracker with a SAM2-first pipeline.
- Solving long occlusions or players leaving the frame for large time spans.
- Using interpolated frames as equivalent evidence for passes, possession, or OCR.
- Building a second OCR engine path unless it demonstrates measurable gain over the official path.

## Current Problems

- The project currently has two pipeline entrypoints that can produce different final behavior.
- The combined OCR route can assign confident but wrong global jerseys even when the secondary OCR contributes no useful votes.
- Local OCR mistakes can contaminate the full player identity.
- Players can disappear for a few frames and then reappear, causing jumps in the rendered video and in the top-down view.

## Main Design Decision

The project will use a single official pipeline:

`analyze_video -> identity stabilization -> jersey OCR by identity -> possession/passes -> final render`

The pipeline remains tracking-first. OCR does not define identity. OCR only labels an already stabilized identity.

## Identity Model

The system will keep three related concepts separate:

- `track_id`: raw tracker fragment.
- `player_id`: canonical player identity across fragments.
- `jersey_segment`: a bounded jersey interpretation valid only for a compatible part of a `player_id`.

This separation is required because a player can be tracked by multiple raw fragments and because OCR evidence is often valid only in some parts of the clip.

## Jersey Stability Design

### Evidence collection

For each `player_id`, the OCR stage will gather jersey crops from all real detections associated with that identity across the full clip. Sample selection should favor:

- large enough crops;
- torso-centered crops;
- low blur frames;
- frames where the player orientation is favorable;
- diversity across time rather than many nearly identical adjacent frames.

Estimated/interpolated boxes are never sampled for OCR.

### Two-level jersey output

The OCR result will expose:

- `canonical_jersey_number`: global jersey for the player, only when evidence is strong.
- `segment_jersey_number`: local jersey evidence for bounded segments.

### Locking policy

The project will use the hybrid policy selected by the user:

- Show `P{id}` while the jersey is unresolved.
- Lock the global jersey only when multi-frame evidence is strong enough.
- Backfill the jersey to earlier compatible segments once locked.
- Do not backfill across conflicting segments.
- If conflict remains unresolved, prefer `P{id}` over an unstable jersey.

### Conflict handling

A `player_id` may contain multiple OCR hypotheses. The resolver should treat them as:

- compatible noise: weak alternatives that do not block locking;
- bounded local conflict: segment-level disagreement that blocks backfill only in that region;
- hard conflict: evidence that the identity stitching is wrong or the OCR is too uncertain.

Hard conflicts should be visible in the exported reports.

## Short-Gap Interpolation Design

### Gap definition

A short gap is a missing stretch of a few frames where the same `player_id` is seen immediately before and after the gap and the continuity remains plausible.

The gap filler should check:

- same `player_id`;
- same team;
- reasonable displacement and velocity;
- compatible box size and aspect ratio;
- no stronger competing detection from another nearby player.

### Estimated detections

When a short gap is accepted, the system creates `estimated` player records for the missing frames.

These records will:

- render as visible boxes in the video;
- appear in the 2D court view;
- use a visually distinct style from real detections;
- be clearly marked in exported data as estimated.

Estimated detections are for continuity only. They must not:

- generate OCR crops;
- create new identities;
- override real detections;
- carry the same trust as real frames for tactical inference.

### 2D smoothing

The top-down trajectory should combine:

- short-gap interpolation to bridge missing frames;
- lightweight smoothing to reduce jitter in consecutive real detections.

Smoothing should remain conservative so the 2D map still reflects the underlying play.

## Rendering Rules

- Real player boxes use the normal style.
- Estimated player boxes use a lighter or dashed-looking style so the user can tell they are inferred.
- Player labels use jersey if locked, otherwise `P{id}`.
- Once a jersey is locked, compatible earlier frames may show the retroactively resolved jersey.
- If a segment is conflicting, the label falls back to `P{id}` there instead of oscillating numbers.

## Data Contract Changes

### `tracks.json`

Each record should distinguish between real and estimated detections, for example with:

- `is_estimated`
- `estimate_reason`
- `source_track_gap`

### `jersey_numbers.json`

The official OCR report should include:

- canonical jersey by `player_id`;
- compatible segment assignments;
- segment conflicts;
- evidence strength and supporting sample counts.

The combined OCR report is removed from the official path.

### `summary.json`

The summary should add metrics such as:

- `players_with_locked_jersey`
- `jersey_conflicts`
- `short_gap_fills`
- `estimated_player_frames`
- `team_switches_after_stabilization`

## Validation

Validation should focus first on `subra.mp4`, then on at least one additional clip.

Minimum acceptance criteria:

- No stable player changes team mid-clip without strong evidence.
- No locked jersey changes mid-clip.
- Short disappear/reappear moments no longer create large visual teleports.
- 2D positions remain continuous across short gaps.
- Estimated frames do not influence OCR locking.
- The single official pipeline produces one final output path and one official jersey report.

## Recommended Implementation Order

1. Remove the combined OCR route from the recommended pipeline and documentation.
2. Make one official pipeline entrypoint the source of truth.
3. Refactor jersey OCR output around canonical identity plus bounded segments.
4. Add short-gap interpolation for player boxes and 2D positions.
5. Update rendering to show estimated boxes distinctly.
6. Add summary metrics and regression checks on `subra.mp4`.

## Risks

- Over-aggressive backfilling could reintroduce wrong jersey propagation.
- Over-aggressive interpolation could draw false player continuity through true identity changes.
- Too much smoothing in 2D could hide real movement dynamics.

The design therefore prefers conservative locking, conservative gap filling, and explicit conflict reporting.

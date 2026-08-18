from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from auto_annotate import HAND_CLASS_ID, Candidate, FinalObject
from video_annotate import (
    FrameDetections,
    TrackResult,
    adaptive_hand_support,
    compact_hand_candidates,
    competitor_guard_metrics,
    constrain_hand_mask,
    eligible_hand_fallback,
    grouped_indices,
    hand_collapse_indices,
    local_hand_reference_area,
    recovery_is_better,
    temporal_competitor_support,
)


def make_args(**overrides):
    values = {
        "compact_hand_max_product_gap": 0.18,
        "compact_hand_cluster_radius": 0.22,
        "compact_hand_score_window": 0.30,
        "compact_hand_min_margin": 0.08,
        "compact_hand_expansion": 0.30,
        "hand_mask_support_expansion": 0.15,
        "hand_partial_mask_support_expansion": 0.45,
        "hand_max_mask_support_expansion": 0.60,
        "hand_component_min_area_ratio": 0.04,
        "hand_component_max_gap": 0.08,
        "hand_component_product_gap": 0.05,
        "min_temporal_compatibility": 0.30,
        "hand_collapse_area_ratio": 0.45,
        "hand_collapse_dimension_ratio": 0.50,
        "hand_collapse_history_min": 4,
        "hand_collapse_history_window": 9,
        "hand_recovery_min_area_ratio": 0.65,
        "video_hand_seed_threshold": 0.30,
        "visible_hand_mode": False,
        "visible_hand_support_expansion": 0.08,
        "visible_hand_min_dino_overlap": 0.70,
        "finger_seed_only": False,
        "competitor_min_score": 0.40,
        "competitor_score_margin": 0.05,
        "competitor_min_overlap": 0.50,
        "competitor_temporal_radius": 2,
        "competitor_min_temporal_support": 1,
        "competitor_temporal_compatibility": 0.25,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def frame(hand: Candidate | None = None) -> FrameDetections:
    return FrameDetections(
        hand_candidates=[hand] if hand else [],
        object_candidates=[],
        best_hand=hand,
        best_object=None,
        fallback_hand=hand,
        fallback_object=None,
    )


def track(box, components=1, source="sam3_video") -> TrackResult:
    return TrackResult(
        FinalObject(
            HAND_CLASS_ID,
            "hand",
            box,
            0.8,
            "complete human hand",
            mask_component_count=components,
        ),
        None,
        source,
        0.8,
    )


class HandCandidateTests(unittest.TestCase):
    def test_whole_hand_is_primary_and_details_expand_its_union(self):
        args = make_args()
        product = Candidate(9, "product", (80, 40, 125, 105), 0.8, "product")
        whole = Candidate(
            HAND_CLASS_ID,
            "hand",
            (30, 30, 92, 112),
            0.52,
            "complete human hand",
        )
        finger = Candidate(
            HAND_CLASS_ID,
            "hand",
            (88, 48, 108, 68),
            0.77,
            "fingers",
        )

        selected = compact_hand_candidates([finger, whole], product, 160, 140, args)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].source, "dino_hand_union")
        self.assertEqual(selected[0].box_xyxy, (30, 30, 108, 112))
        self.assertIn("complete human hand", selected[0].prompt)

    def test_detail_only_seed_gets_adaptive_candidate_expansion(self):
        selected = compact_hand_candidates(
            [Candidate(HAND_CLASS_ID, "hand", (40, 40, 50, 50), 0.8, "fingers")],
            None,
            100,
            100,
            make_args(),
        )[0]

        self.assertEqual(selected.source, "dino_partial_hand")
        self.assertLess(selected.box_xyxy[0], 35)
        self.assertGreater(selected.box_xyxy[2], 55)

    def test_palm_of_human_hand_is_still_a_partial_seed(self):
        selected = compact_hand_candidates(
            [Candidate(HAND_CLASS_ID, "hand", (40, 40, 50, 50), 0.8, "palm of human hand")],
            None,
            100,
            100,
            make_args(),
        )[0]

        self.assertEqual(selected.source, "dino_partial_hand")

    def test_visible_detail_keeps_observed_box_without_full_hand_expansion(self):
        selected = compact_hand_candidates(
            [Candidate(HAND_CLASS_ID, "hand", (40, 40, 50, 50), 0.8, "fingers")],
            None,
            100,
            100,
            make_args(visible_hand_mode=True),
        )[0]

        self.assertEqual(selected.source, "dino_visible_partial_hand")
        self.assertEqual(selected.box_xyxy, (40, 40, 50, 50))

    def test_detail_candidate_remains_baseline_fallback_when_flag_is_off(self):
        detail = Candidate(HAND_CLASS_ID, "hand", (40, 40, 50, 50), 0.8, "fingers")
        self.assertIs(eligible_hand_fallback(detail, make_args()), detail)

    def test_detail_candidate_is_seed_only_when_flag_is_on(self):
        detail = Candidate(HAND_CLASS_ID, "hand", (40, 40, 50, 50), 0.8, "fingers")
        self.assertIsNone(
            eligible_hand_fallback(detail, make_args(finger_seed_only=True))
        )

    def test_whole_hand_can_still_fallback_in_seed_only_mode(self):
        whole = Candidate(
            HAND_CLASS_ID,
            "hand",
            (30, 30, 70, 80),
            0.7,
            "complete human hand",
        )
        self.assertIs(
            eligible_hand_fallback(whole, make_args(finger_seed_only=True)), whole
        )

    def test_partial_support_uses_temporal_hand_size_with_a_cap(self):
        args = make_args()
        partial = Candidate(
            HAND_CLASS_ID,
            "hand",
            (40, 40, 50, 50),
            0.8,
            "partial palm and fingers",
            source="dino_partial_hand",
        )
        support, expansion = adaptive_hand_support(
            partial,
            (20, 20, 70, 70),
            100,
            100,
            args,
        )

        self.assertEqual(support, (20, 20, 70, 70))
        self.assertGreaterEqual(expansion, 0.45)
        self.assertLessEqual(expansion, 0.60)

    def test_visible_support_does_not_union_previous_full_hand_extent(self):
        partial = Candidate(
            HAND_CLASS_ID,
            "hand",
            (40, 40, 50, 50),
            0.8,
            "visible fingers or palm of human hand",
            source="dino_visible_partial_hand",
        )
        support, expansion = adaptive_hand_support(
            partial,
            (20, 20, 70, 70),
            100,
            100,
            make_args(visible_hand_mode=True),
        )

        self.assertEqual(support, (40, 40, 50, 50))
        self.assertEqual(expansion, 0.08)


class HandMaskTests(unittest.TestCase):
    def test_disconnected_fingers_near_product_are_grouped(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 20:40] = True
        mask[43:53, 43:53] = True
        mask[72:76, 72:76] = True

        grouped, count = constrain_hand_mask(
            mask,
            support_box=(0, 0, 80, 80),
            previous_box=(18, 18, 42, 42),
            product_box=(35, 35, 58, 58),
            width=100,
            height=100,
            expansion=0.0,
            args=make_args(),
        )

        self.assertEqual(count, 2)
        self.assertTrue(grouped[25, 25])
        self.assertTrue(grouped[47, 47])
        self.assertFalse(grouped[73, 73])


class CompetitorGuardTests(unittest.TestCase):
    def test_overlapping_stronger_shoe_is_marked_for_rejection(self):
        hand_candidate = Candidate(
            HAND_CLASS_ID, "hand", (10, 10, 50, 50), 0.42, "human hand"
        )
        shoe = Candidate(-100, "shoe_competitor", (8, 8, 52, 52), 0.60, "shoe")
        item = frame(hand_candidate)
        item.best_shoe = shoe

        metrics = competitor_guard_metrics(
            track((10, 10, 50, 50)), item, shoe, 2, make_args()
        )

        self.assertTrue(metrics["would_reject"])
        self.assertGreater(metrics["score_gap"], 0.05)

    def test_non_overlapping_shoe_does_not_reject_hand(self):
        hand_candidate = Candidate(
            HAND_CLASS_ID, "hand", (10, 10, 50, 50), 0.42, "human hand"
        )
        shoe = Candidate(-100, "shoe_competitor", (70, 70, 95, 95), 0.65, "shoe")
        item = frame(hand_candidate)

        metrics = competitor_guard_metrics(
            track((10, 10, 50, 50)), item, shoe, 2, make_args()
        )

        self.assertFalse(metrics["would_reject"])

    def test_temporal_support_counts_only_compatible_competitors(self):
        shoe = Candidate(-100, "shoe_competitor", (10, 10, 50, 50), 0.6, "shoe")
        far_shoe = Candidate(-100, "shoe_competitor", (70, 70, 95, 95), 0.6, "shoe")
        frames = [frame(), frame(), frame()]
        frames[0].best_shoe = shoe
        frames[1].best_shoe = shoe
        frames[2].best_shoe = far_shoe

        self.assertEqual(
            temporal_competitor_support(frames, 1, "shoe", 100, 100, make_args()),
            1,
        )


class HandRecoveryTests(unittest.TestCase):
    def test_temporal_area_and_dimension_collapse_is_detected(self):
        full = Candidate(
            HAND_CLASS_ID,
            "hand",
            (10, 10, 50, 50),
            0.7,
            "complete human hand",
            source="dino_whole_hand",
        )
        frames = [frame(full) for _ in range(6)]
        results = {index: track((10, 10, 50, 50), components=2) for index in range(6)}
        results[4] = track((22, 22, 30, 30), components=1)

        self.assertEqual(hand_collapse_indices(results, frames, make_args()), [4])

    def test_dino_whole_hand_detects_early_collapse_without_history(self):
        full = Candidate(
            HAND_CLASS_ID,
            "hand",
            (10, 10, 50, 50),
            0.7,
            "complete human hand",
            source="dino_whole_hand",
        )
        results = {0: track((22, 22, 30, 30), components=1)}

        self.assertEqual(hand_collapse_indices(results, [frame(full)], make_args()), [0])

    def test_visible_partial_matching_dino_is_not_recovered_as_collapse(self):
        full = Candidate(
            HAND_CLASS_ID,
            "hand",
            (10, 10, 50, 50),
            0.7,
            "complete human hand",
            source="dino_whole_hand",
        )
        partial = Candidate(
            HAND_CLASS_ID,
            "hand",
            (22, 22, 30, 30),
            0.7,
            "visible fingers or palm of human hand",
            source="dino_visible_partial_hand",
        )
        frames = [frame(full) for _ in range(4)] + [frame(partial)]
        results = {index: track((10, 10, 50, 50), components=2) for index in range(4)}
        results[4] = track((22, 22, 30, 30), components=1)

        self.assertEqual(
            hand_collapse_indices(
                results,
                frames,
                make_args(visible_hand_mode=True),
            ),
            [],
        )

    def test_visible_mode_still_detects_track_contradicted_by_current_full_hand(self):
        full = Candidate(
            HAND_CLASS_ID,
            "hand",
            (10, 10, 50, 50),
            0.7,
            "complete human hand",
            source="dino_whole_hand",
        )
        frames = [frame(full) for _ in range(5)]
        results = {index: track((10, 10, 50, 50), components=2) for index in range(4)}
        results[4] = track((22, 22, 30, 30), components=1)

        self.assertEqual(
            hand_collapse_indices(
                results,
                frames,
                make_args(visible_hand_mode=True),
            ),
            [4],
        )

    def test_local_reference_ignores_distant_perspective(self):
        results = {index: track((0, 0, 10, 10)) for index in range(20)}
        results[100] = track((0, 0, 40, 40))
        results[101] = track((0, 0, 5, 5))

        reference = local_hand_reference_area(results, 101, {101}, radius=3)

        self.assertEqual(reference, 1600.0)

    def test_recovery_prefers_box_closer_to_local_reference(self):
        original = track((20, 20, 30, 30))
        recovered = track((10, 10, 48, 48), source="sam3_video_reseed")
        self.assertTrue(recovery_is_better(original, recovered, reference_area=1600.0))

    def test_nearby_collapse_frames_form_one_recovery_span(self):
        self.assertEqual(grouped_indices([10, 11, 14, 30], merge_gap=2), [(10, 14), (30, 30)])


if __name__ == "__main__":
    unittest.main()

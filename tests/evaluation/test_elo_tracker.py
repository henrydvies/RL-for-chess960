"""
Tests for EloTracker in evaluation/elo_tracker.py
"""
import json
import pytest
import os
from evaluation.elo_tracker import EloTracker


@pytest.fixture
def tracker():
    """
    Fresh EloTracker with known starting ratings, no file path
    """
    return EloTracker(elo_map={"rlAgent": 600, "RandomAgent": 600}, elo_tracking_path="nonexistent.json")


@pytest.fixture
def tmp_tracker(tmp_path):
    """
    EloTracker with a temporary file path for save/load tests
    """
    path = str(tmp_path / "elo_tracker.json")
    return EloTracker(elo_map={"rlAgent": 600, "RandomAgent": 600}, elo_tracking_path=path)


## Testing Elo updates correctly

def test_win_increases_elo(tracker):
    """
    Winning should increase the winner's Elo
    """
    before = tracker.elo_map["rlAgent"]
    tracker.update("rlAgent", "RandomAgent", True)
    assert tracker.elo_map["rlAgent"] > before


def test_loss_decreases_elo(tracker):
    """
    Losing should decrease the loser's Elo
    """
    before = tracker.elo_map["rlAgent"]
    tracker.update("rlAgent", "RandomAgent", False)
    assert tracker.elo_map["rlAgent"] < before


def test_draw_small_change(tracker):
    """
    A draw between equal opponents should result in minimal Elo change
    """
    before = tracker.elo_map["rlAgent"]
    tracker.update("rlAgent", "RandomAgent", None)
    assert abs(tracker.elo_map["rlAgent"] - before) < 1


def test_opponent_elo_moves_opposite(tracker):
    """
    When rlAgent wins, RandomAgent Elo should decrease
    """
    before = tracker.elo_map["RandomAgent"]
    tracker.update("rlAgent", "RandomAgent", True)
    assert tracker.elo_map["RandomAgent"] < before


## Testing ratings persist across multiple updates

def test_ratings_persist_across_updates(tracker):
    """
    Multiple updates should accumulate — ratings should keep changing
    """
    tracker.update("rlAgent", "RandomAgent", True)
    after_one = tracker.elo_map["rlAgent"]
    tracker.update("rlAgent", "RandomAgent", True)
    after_two = tracker.elo_map["rlAgent"]
    assert after_two > after_one


def test_elo_sum_roughly_preserved(tracker):
    """
    Total Elo should be roughly preserved after a game (zero-sum)
    """
    total_before = tracker.elo_map["rlAgent"] + tracker.elo_map["RandomAgent"]
    tracker.update("rlAgent", "RandomAgent", True)
    total_after = tracker.elo_map["rlAgent"] + tracker.elo_map["RandomAgent"]
    assert abs(total_after - total_before) < 0.01


## Testing save and load

def test_save_creates_file(tmp_tracker):
    """
    Save should create a JSON file at the specified path
    """
    tmp_tracker.save()
    assert os.path.exists(tmp_tracker.elo_tracking_path)


def test_load_restores_ratings(tmp_tracker):
    """
    Load should restore ratings saved to file
    """
    tmp_tracker.update("rlAgent", "RandomAgent", True)
    tmp_tracker.save()

    loaded = EloTracker(elo_tracking_path=tmp_tracker.elo_tracking_path)
    assert abs(loaded.elo_map["rlAgent"] - tmp_tracker.elo_map["rlAgent"]) < 0.01


def test_save_load_roundtrip(tmp_tracker):
    """
    Save then load should produce identical ratings
    """
    tmp_tracker.save()
    path = tmp_tracker.elo_tracking_path
    loaded = EloTracker(elo_tracking_path=path)
    assert loaded.elo_map == tmp_tracker.elo_map
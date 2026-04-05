"""Тесты для ml_core/experiment_tracker.py"""
import pytest
import os
import json
from ml_core.experiment_tracker import ExperimentTracker


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(storage_dir=str(tmp_path / "experiments"))


class TestExperimentTracker:
    """Тесты ExperimentTracker."""

    def test_save_returns_id(self, tracker):
        exp_id = tracker.save_experiment("test_exp", {"metrics": {"f1": 0.8}})
        assert isinstance(exp_id, str)
        assert "test_exp" in exp_id

    def test_save_creates_directory(self, tracker):
        exp_id = tracker.save_experiment("test_exp", {"metrics": {"f1": 0.8}})
        assert os.path.exists(os.path.join(tracker.storage_dir, exp_id))

    def test_save_creates_metadata(self, tracker):
        data = {"metrics": {"f1": 0.85, "roc_auc": 0.90}}
        exp_id = tracker.save_experiment("test_exp", data)
        meta_path = os.path.join(tracker.storage_dir, exp_id, "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["metrics"]["f1"] == 0.85

    def test_load_returns_data(self, tracker):
        data = {"metrics": {"f1": 0.9}, "features": ["a", "b"]}
        exp_id = tracker.save_experiment("test_exp", data)
        loaded = tracker.load_experiment(exp_id)
        assert loaded["metrics"]["f1"] == 0.9
        assert loaded["features"] == ["a", "b"]

    def test_load_nonexistent_raises(self, tracker):
        with pytest.raises((FileNotFoundError, ValueError)):
            tracker.load_experiment("nonexistent_id")

    def test_delete_removes_directory(self, tracker):
        exp_id = tracker.save_experiment("to_delete", {"x": 1})
        assert tracker.delete_experiment(exp_id) is True
        assert not os.path.exists(os.path.join(tracker.storage_dir, exp_id))

    def test_delete_nonexistent_returns_false(self, tracker):
        assert tracker.delete_experiment("nonexistent") is False

    def test_list_returns_dataframe(self, tracker):
        tracker.save_experiment("exp1", {"f1": 0.8})
        tracker.save_experiment("exp2", {"f1": 0.9})
        df = tracker.list_experiments()
        assert len(df) >= 2
        assert "id" in df.columns

    def test_list_sorted_by_time(self, tracker):
        import time
        tracker.save_experiment("exp1", {"f1": 0.8})
        time.sleep(0.1)
        tracker.save_experiment("exp2", {"f1": 0.9})
        df = tracker.list_experiments()
        # Последний сохранённый — первый в списке (обратная сортировка)
        assert df.iloc[0]["id"] > df.iloc[1]["id"] or df.iloc[0]["name"] == "exp2"

from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "static" / "index.html"


def test_challenge_mode_includes_six_questions_and_controls():
    content = HTML.read_text(encoding="utf-8")
    assert 'id="challenge-lab"' in content
    assert "const questions=[" in content
    assert content.count("correct:") == 6
    assert 'id="quiz-check"' in content
    assert 'id="quiz-restart"' in content


def test_challenge_mode_labels_recorded_metrics_honestly():
    content = HTML.read_text(encoding="utf-8")
    assert "reference results, not new live inference" in content
    assert "0.8004" in content
    assert "74.9%" in content

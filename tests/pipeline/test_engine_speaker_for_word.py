from intervaltree import Interval, IntervalTree

# Module to test
from reverb_gui.pipeline import engine

# Note: `sample_speaker_tree` fixture is automatically imported from conftest.py

# === Tests for speaker_for_word (originally in test_engine.py) ===

def test_speaker_for_word_single_interval() -> None:
    """Test speaker_for_word finds the correct speaker with one interval.
    Originally from test_engine.py line 37.
    """
    tree = IntervalTree([Interval(0, 10, "SPEAKER_A")])
    assert engine.speaker_for_word(2.0, 1.0, tree) == "SPEAKER_A"

def test_speaker_for_word_no_interval() -> None:
    """Test speaker_for_word finds nearest speaker when word is outside intervals.
    Originally from test_engine.py line 43.
    """
    tree = IntervalTree([
        Interval(0, 5, "SPEAKER_A"),
        Interval(10, 15, "SPEAKER_B")
    ])
    # Word is between intervals, closer to B
    assert engine.speaker_for_word(7.0, 1.0, tree) == "SPEAKER_B"
    # Word is before all intervals
    assert engine.speaker_for_word(0.0 - 2.0, 1.0, tree) == "SPEAKER_A"
    # Word is after all intervals
    assert engine.speaker_for_word(16.0, 1.0, tree) == "SPEAKER_B"

def test_speaker_for_word_overlap_majority_simple() -> None:
    """Test speaker_for_word picks speaker with majority overlap (simple case).
    Originally from test_engine.py line 57.
    """
    tree = IntervalTree([
        Interval(0, 10, "SPEAKER_A"), # A speaks for 10s
        Interval(5, 15, "SPEAKER_B")  # B overlaps from 5s-10s, continues to 15s
    ])
    # Word from 6s to 8s (duration 2s) - fully within overlap
    # A: [6, 8] = 2s
    # B: [6, 8] = 2s (Tie, implementation detail which wins - let's assume max picks last added/found)
    # Let's adjust to make it clear
    tree = IntervalTree([
        Interval(0, 10, "SPEAKER_A"),
        Interval(7, 15, "SPEAKER_B") # B overlaps A from 7-10
    ])
    # Word 6s-9s (duration 3s)
    # A: [6, 9] = 3s
    # B: [7, 9] = 2s
    assert engine.speaker_for_word(6.0, 3.0, tree) == "SPEAKER_A"

    # Word 8s-12s (duration 4s)
    # A: [8, 10] = 2s
    # B: [8, 12] = 4s
    assert engine.speaker_for_word(8.0, 4.0, tree) == "SPEAKER_B"

def test_speaker_for_word_no_speakers() -> None:
    """Test speaker_for_word handles an empty IntervalTree.
    Originally from test_engine.py line 82.
    """
    tree = IntervalTree()
    assert engine.speaker_for_word(1.0, 1.0, tree) == "UNKNOWN"

def test_speaker_for_word_within_gap() -> None:
    """Test speaker_for_word finds nearest when word is entirely in a gap.
    Originally from test_engine.py line 88.
    """
    tree = IntervalTree([
        Interval(0, 5, "SPEAKER_A"),
        Interval(10, 15, "SPEAKER_B")
    ])
    # Word from 6s-7s, exactly in the gap. Should find nearest (A or B depending on distance)
    # Midpoint is 7.5. 6-7 is closer to A's end (5) than B's start (10)
    assert engine.speaker_for_word(6.0, 1.0, tree) == "SPEAKER_A"

# === Tests using the sample_speaker_tree fixture (originally from test_engine.py lines 115-148) ===

def test_speaker_for_word_exact_match(sample_speaker_tree: IntervalTree) -> None:
    """Test word fully contained within one speaker segment.
    Originally from test_engine.py line 115.
    """
    # Word: 0.5s - 0.8s (within SPK_0)
    assert engine.speaker_for_word(0.5, 0.3, sample_speaker_tree) == "SPK_0"
    # Word: 2.5s - 3.0s (within SPK_1)
    assert engine.speaker_for_word(2.5, 0.5, sample_speaker_tree) == "SPK_1"
    # Word: 4.2s - 4.8s (within the second SPK_0 segment)
    assert engine.speaker_for_word(4.2, 0.6, sample_speaker_tree) == "SPK_0"

def test_speaker_for_word_overlap_majority_fixture(sample_speaker_tree: IntervalTree) -> None:
    """Test word overlapping two speakers, using the fixture.
    Originally from test_engine.py line 124.
    Renamed from test_speaker_for_word_overlap_majority to avoid conflict.
    """
    # Word: 1.7s - 2.2s (Overlap: SPK_0=0.3s, SPK_1=0.7s) -> Majority SPK_1
    assert engine.speaker_for_word(1.7, 0.5, sample_speaker_tree) == "SPK_1"
    # Word: 1.4s - 1.8s (Overlap: SPK_0=0.4s, SPK_1=0.3s) -> Majority SPK_0
    assert engine.speaker_for_word(1.4, 0.4, sample_speaker_tree) == "SPK_0"

def test_speaker_for_word_no_match_nearest(sample_speaker_tree: IntervalTree) -> None:
    """Test word falling in a gap, returns the nearest speaker using the fixture.
    Originally from test_engine.py line 132.
    """
    # Word: 3.6s - 3.8s (Gap between SPK_1 end (3.5) and SPK_0 start (4.0))
    # Nearest end: SPK_1 at 3.5 (dist 0.1)
    # Nearest start: SPK_0 at 4.0 (dist 0.2)
    assert engine.speaker_for_word(3.6, 0.2, sample_speaker_tree) == "SPK_1"
    # Word: 5.2s - 5.5s (Gap after last SPK_0)
    # Nearest end: SPK_0 at 5.0 (dist 0.2)
    # Nearest start: SPK_2 at 6.0 (dist 0.5)
    assert engine.speaker_for_word(5.2, 0.3, sample_speaker_tree) == "SPK_0"
    # Word: -0.5s - -0.2s (Before first speaker)
    # Nearest start: SPK_0 at 0.0 (dist 0.2)
    assert engine.speaker_for_word(-0.5, 0.3, sample_speaker_tree) == "SPK_0"

def test_speaker_for_word_empty_tree() -> None:
    """Test behavior with an empty IntervalTree.
    Originally from test_engine.py line 146.
    """
    empty_tree = IntervalTree()
    assert engine.speaker_for_word(1.0, 0.5, empty_tree) == "UNKNOWN"

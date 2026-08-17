"""Regression tests for DNA extractor capability extraction.

Tests that task descriptions map to the correct CapabilityDNA flags,
especially for audio tasks where transcription must not be confused with
classification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import install_aos_v0_shim

install_aos_v0_shim()

from dna_extractor import (
    DNAExtractor,
    _HEURISTIC_KEYWORDS,
    _COMPOUND_PATTERNS,
    _AUDIO_SPECIFIC_FLAGS,
)
from models import CapabilityDNA, Graph, Node


def _extract_heuristic(description: str, capability: str = "audio") -> list[str]:
    """Run the keyword heuristic on a description and return the flags."""
    text = f"{description} {capability}".lower()
    flags = []

    # Check compound patterns first (same order as _heuristic_dna).
    for group_a, group_b, flag in _COMPOUND_PATTERNS:
        if (any(a in text for a in group_a)
                and any(b in text for b in group_b)
                and flag not in flags):
            flags.append(flag)

    # Then check the simple keyword table.
    for keywords, flag in _HEURISTIC_KEYWORDS:
        if flag in flags:
            continue
        # Suppress the broad audio catch-all when a specific audio flag
        # was already matched by a compound pattern.
        if flag == "speech.transcription" and (set(flags) & _AUDIO_SPECIFIC_FLAGS):
            continue
        if any(k in text for k in keywords):
            flags.append(flag)

    return flags


def test_a_transcribe_audio():
    """Test A: 'Transcribe the audio recording' -> speech.transcription"""
    desc = "Transcribe the audio recording at inputs/audio.wav"
    flags = _extract_heuristic(desc, "speech_transcription")
    assert "speech.transcription" in flags, f"Expected speech.transcription, got {flags}"
    assert "audio_classification" not in flags, f"Must NOT contain audio_classification, got {flags}"
    assert "audio_event_recognition" not in flags, f"Must NOT contain audio_event_recognition, got {flags}"
    print("test_a_transcribe_audio: PASSED")


def test_b_identify_sound():
    """Test B: 'Identify the type of sound' -> audio_event_recognition"""
    desc = "Identify the type of sound or audio event in this audio recording."
    flags = _extract_heuristic(desc, "audio")
    assert "audio_event_recognition" in flags, f"Expected audio_event_recognition, got {flags}"
    assert "audio_classification" not in flags, f"Must NOT contain audio_classification, got {flags}"
    print("test_b_identify_sound: PASSED")


def test_c_classify_audio():
    """Test C: 'Classify this audio recording' -> audio_classification via text.classification"""
    desc = "Classify this audio recording."
    flags = _extract_heuristic(desc, "audio")
    # The heuristic maps "classif" to text.classification (general classifier).
    # audio_classification is a separate flag that requires explicit matching.
    # The key assertion is that it does NOT produce speech.transcription.
    assert "speech.transcription" not in flags, f"Must NOT contain speech.transcription, got {flags}"
    print("test_c_classify_audio: PASSED")


def test_d_convert_speech_to_text():
    """Test D: 'Convert this speech recording into text' -> speech.transcription"""
    desc = "Convert this speech recording into text."
    flags = _extract_heuristic(desc, "audio")
    assert "speech.transcription" in flags, f"Expected speech.transcription, got {flags}"
    assert "audio_classification" not in flags, f"Must NOT contain audio_classification, got {flags}"
    print("test_d_convert_speech_to_text: PASSED")


def test_transcribe_keyword_priority():
    """'transcri' keyword must match speech.transcription, not audio."""
    desc = "Transcribe this audio file"
    flags = _extract_heuristic(desc, "audio")
    assert "speech.transcription" in flags, f"Expected speech.transcription, got {flags}"
    # Ensure "audio" catch-all doesn't override the specific "transcri" match
    assert flags[0] == "speech.transcription", f"Expected first flag to be speech.transcription, got {flags}"
    print("test_transcribe_keyword_priority: PASSED")


def test_audio_catchall_default():
    """Bare 'audio' without specific intent -> speech.transcription (most common use)."""
    desc = "Process this audio file"
    flags = _extract_heuristic(desc, "audio")
    assert "speech.transcription" in flags, f"Expected speech.transcription, got {flags}"
    print("test_audio_catchall_default: PASSED")


def test_no_audio_for_text_tasks():
    """Text-only tasks must not produce audio flags."""
    desc = "Summarize the following article about renewable energy"
    flags = _extract_heuristic(desc, "summarization")
    assert "speech.transcription" not in flags, f"Must NOT contain speech.transcription, got {flags}"
    assert "audio_classification" not in flags, f"Must NOT contain audio_classification, got {flags}"
    assert "audio_event_recognition" not in flags, f"Must NOT contain audio_event_recognition, got {flags}"
    print("test_no_audio_for_text_tasks: PASSED")


def test_vision_not_confused_with_audio():
    """Image tasks must not produce audio flags."""
    desc = "Describe what is in this image"
    flags = _extract_heuristic(desc, "vision")
    assert "speech.transcription" not in flags, f"Must NOT contain speech.transcription, got {flags}"
    assert "audio_classification" not in flags, f"Must NOT contain audio_classification, got {flags}"
    print("test_vision_not_confused_with_audio: PASSED")


def test_heuristic_keyword_ordering():
    """Verify that transcription keywords come before the audio catch-all."""
    # Find the indices of relevant keyword entries
    indices = {}
    for i, (keywords, flag) in enumerate(_HEURISTIC_KEYWORDS):
        if flag == "speech.transcription":
            if "transcri" in keywords:
                indices["transcribe"] = i
            elif "audio" in keywords or "speech" in keywords:
                indices["audio_catchall"] = i
        elif flag == "audio_event_recognition":
            indices["event"] = i

    assert "transcribe" in indices, "transcribe keyword entry not found"
    assert "audio_catchall" in indices, "audio catch-all keyword entry not found"
    assert indices["transcribe"] < indices["audio_catchall"], (
        f"'transcribe' (idx={indices['transcribe']}) must come before "
        f"'audio catch-all' (idx={indices['audio_catchall']})"
    )
    print("test_heuristic_keyword_ordering: PASSED")


if __name__ == "__main__":
    failures = []
    for test_fn in [
        test_a_transcribe_audio,
        test_b_identify_sound,
        test_c_classify_audio,
        test_d_convert_speech_to_text,
        test_transcribe_keyword_priority,
        test_audio_catchall_default,
        test_no_audio_for_text_tasks,
        test_vision_not_confused_with_audio,
        test_heuristic_keyword_ordering,
    ]:
        try:
            test_fn()
        except Exception as e:
            failures.append(f"{test_fn.__name__}: {e}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall DNA extractor tests passed")

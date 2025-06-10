from pathlib import Path

from legal_ai_system.utils.simple_media import (
    transcribe_audio,
    process_video_deposition,
    extract_form_fields,
    analyze_redline,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_transcribe_audio():
    audio = FIXTURES / "sample.mp3"
    assert transcribe_audio(audio) == "Test audio transcript"


def test_process_video_deposition():
    video = FIXTURES / "sample.mp4"
    result = process_video_deposition(video)
    assert result == [
        {"speaker": "Attorney", "text": "Please state your name."},
        {"speaker": "Witness", "text": "John Doe."},
    ]


def test_extract_form_fields():
    form = FIXTURES / "form.txt"
    fields = extract_form_fields(form)
    assert fields == {
        "Name": "Jane Doe",
        "Date": "2024-05-20",
        "Case": "12345",
    }


def test_analyze_redline():
    original = (FIXTURES / "original.txt").read_text()
    revised = (FIXTURES / "revised.txt").read_text()
    diff = analyze_redline(original, revised)
    assert "Additional clause added." in diff["insertions"]
    assert "the terms." in diff["deletions"][0]

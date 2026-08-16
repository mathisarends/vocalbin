from scripts.generate_voices import VoiceRecord, enum_members, identifier, render


def test_identifier_normalizes_cartesia_voice_names() -> None:
    assert identifier("Skylar - Friendly Guide") == "SKYLAR_FRIENDLY_GUIDE"
    assert identifier("Élodie / Narrator") == "ELODIE_NARRATOR"
    assert identifier("123 Voice") == "VOICE_123_VOICE"


def test_enum_members_are_deterministic_and_resolve_collisions() -> None:
    voices = [
        VoiceRecord(id="bbbbbbbb-0000", name="Same Name"),
        VoiceRecord(id="aaaaaaaa-0000", name="Same-Name"),
        VoiceRecord(id="aaaaaaaa-0000", name="Same-Name"),
    ]

    assert enum_members(voices) == [
        ("SAME_NAME", "bbbbbbbb-0000"),
        ("SAME_NAME_AAAAAAAA", "aaaaaaaa-0000"),
    ]


def test_render_creates_string_enum() -> None:
    content = render([VoiceRecord(id="voice-uuid", name="Skylar - Friendly Guide")])

    assert "class Voice(StrEnum):" in content
    assert 'SKYLAR_FRIENDLY_GUIDE = "voice-uuid"' in content

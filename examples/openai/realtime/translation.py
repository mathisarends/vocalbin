import asyncio
from pathlib import Path

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.realtime import (
    OpenAIRealtimeTranslator,
    RealtimeError,
    RealtimeSessionConnected,
    RealtimeSourceTranscriptDelta,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationClosed,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationTranscriptDelta,
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "realtime-translation.pcm"


async def main() -> None:
    config = RealtimeTranslationConfig(
        target_language=RealtimeTranslationLanguage.ENGLISH
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_text = ""
    target_text = ""

    with (
        OUTPUT_PATH.open("wb") as output,
        RealtimeTerminal(
            title="Live translation · Auto → English",
            fields=["source", "translation"],
        ) as terminal,
    ):
        async with OpenAIRealtimeTranslator(config) as translator:
            async for event in translator.stream():
                match event:
                    case RealtimeSessionConnected():
                        terminal.update(status="Ready")
                    case RealtimeSourceTranscriptDelta(delta=delta):
                        source_text += delta
                        terminal.update(status="Listening", source=source_text)
                    case RealtimeTranslationTranscriptDelta(delta=delta):
                        target_text += delta
                        terminal.update(
                            status="Translating",
                            translation=target_text,
                        )
                    case RealtimeTranslationAudioDelta(audio=audio):
                        output.write(audio)
                    case RealtimeTranslationClosed():
                        terminal.update(status=f"Saved audio to {OUTPUT_PATH}")
                    case RealtimeError(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

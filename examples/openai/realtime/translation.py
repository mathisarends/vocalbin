import asyncio
from pathlib import Path

from dotenv import load_dotenv

from examples.openai.realtime._terminal import RealtimeTerminal
from vocalbin.openai.realtime import TranslatorBuilder, events

load_dotenv()

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "realtime-translation.pcm"


async def main() -> None:
    translator = TranslatorBuilder().target_language("en").build()
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
        async with translator:
            async for event in translator.stream():
                match event:
                    case events.SessionConnected():
                        terminal.update(status="Ready")
                    case events.SourceTranscriptDelta(delta=delta):
                        source_text += delta
                        terminal.update(status="Listening", source=source_text)
                    case events.TranslationTranscriptDelta(delta=delta):
                        target_text += delta
                        terminal.update(
                            status="Translating",
                            translation=target_text,
                        )
                    case events.TranslationAudioDelta(audio=audio):
                        output.write(audio)
                    case events.TranslationClosed():
                        terminal.update(status=f"Saved audio to {OUTPUT_PATH}")
                    case events.Error(error=error):
                        terminal.error(str(error))
                        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

import asyncio
from pathlib import Path

from vocalbin.realtime import (
    OpenAIRealtimeTranslator,
    RealtimeError,
    RealtimeSourceTranscriptDelta,
    RealtimeTranslationAudioDelta,
    RealtimeTranslationConfig,
    RealtimeTranslationLanguage,
    RealtimeTranslationTranscriptDelta,
)


async def main() -> None:
    config = RealtimeTranslationConfig(
        target_language=RealtimeTranslationLanguage.ENGLISH
    )
    output_path = Path("examples/output/realtime-translation.pcm")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_text = ""
    target_text = ""

    def render() -> None:
        print(f"\x1b[2K\r[source] {source_text}", end="")
        print(f"\n\x1b[2K\r[target] {target_text}", end="")
        print("\x1b[1A\r", end="", flush=True)

    print()  # reserve the source/target lines below the cursor
    with output_path.open("wb") as output:
        async with OpenAIRealtimeTranslator(config) as translator:
            async for event in translator.stream():
                match event:
                    case RealtimeSourceTranscriptDelta(delta=delta):
                        source_text += delta
                        render()
                    case RealtimeTranslationTranscriptDelta(delta=delta):
                        target_text += delta
                        render()
                    case RealtimeTranslationAudioDelta(audio=audio):
                        output.write(audio)
                    case RealtimeError(error=error):
                        print(f"\n\nRealtime error: {error}")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

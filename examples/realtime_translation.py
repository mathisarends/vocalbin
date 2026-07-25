import asyncio
import sys
from pathlib import Path

from vocalbin.translation import (
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

    with output_path.open("wb") as output:
        async with OpenAIRealtimeTranslator(config) as translator:
            async for event in translator.stream():
                match event:
                    case RealtimeSourceTranscriptDelta(delta=delta):
                        print(delta, end="", flush=True, file=sys.stderr)
                    case RealtimeTranslationTranscriptDelta(delta=delta):
                        print(delta, end="", flush=True)
                    case RealtimeTranslationAudioDelta(audio=audio):
                        output.write(audio)
                    case RealtimeError(error=error):
                        print(f"\nRealtime error: {error}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())

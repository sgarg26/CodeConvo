import asyncio
import json

# This example uses the sounddevice library to get an audio stream from the
# microphone. It's not a dependency of the project but can be installed with
# `pip install sounddevice`.
import sounddevice

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from anthropic import AnthropicBedrock

"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""

client = AnthropicBedrock(
    aws_region="us-east-1"
)


async def get_ai_analysis(prompt: str) -> bool:
    resp = client.messages.create(
        max_tokens=256,
        system="You are an AWS Solutions Architect. Concisely answer any valid AWS related questions asked in the following "
               "conversation. Return your response as a JSON object with fields in the "
               "specified data types {valid_question: boolean, question: str, answer: str}. Add empty string to "
               "question and answer fields if a valid question is not asked.",
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0"
        # model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ).content[0].text
    json_obj = json.loads(resp, strict=False)
    if json_obj["valid_question"]:
        await msg_queue.put(resp)
        return True
    return False

class MyEventHandler(TranscriptResultStreamHandler):
    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                s = result.alternatives[0].transcript
                b = await get_ai_analysis(s)
                if b:
                    msg = await msg_queue.get()
                    msg_queue.task_done()
                    print(msg)
                else:
                    print("nothing bruh")


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe():
    # Setup up our client with our chosen AWS region
    client = TranscribeStreamingClient(region="us-east-1")
    global msg_queue
    msg_queue = asyncio.Queue()

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
    )

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    await asyncio.gather(write_chunks(stream), handler.handle_events())

def run():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(basic_transcribe())
    loop.close()

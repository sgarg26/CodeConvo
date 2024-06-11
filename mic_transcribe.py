import asyncio

# This example uses the sounddevice library to get an audio stream from the
# microphone. It's not a dependency of the project but can be installed with
# `pip install sounddevice`.
import sounddevice
from threading import Thread
from flask import Flask

from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from anthropic import AnthropicBedrock


"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""

client = AnthropicBedrock()

def get_message(msg):
    with client.messages.stream(
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"Assume you're an AWS developer. Identify any valid questions in the following text wrapped in quotes, \
                and answer them concisely. Return each question and answer as a key value pair: '{msg}'"
            }
        ],
        model="anthropic.claude-3-haiku-20240307-v1:0"
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            # s += text
        print()
    # return s

class MyEventHandler(TranscriptResultStreamHandler):

    def __init__(self, transcript_result_stream: TranscriptResultStream):
        super().__init__(transcript_result_stream)
        self.transcript = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                s = result.alternatives[-1].transcript
                self.transcript += f"{s}\n"
                Thread(target=get_message, args=(s,), daemon=True).start()
                print(s)
                # ans = await get_message(s)
                # if not ans == "No":
                #     print(f"{s}: {ans}")
                # for alt in result.alternatives:
                #     print(alt.transcript)
                
    def get_transcript(self):
        return self.transcript

    def reset_transcript(self):
        self.transcript = ""

def listen_for_input(handler: MyEventHandler):
    while True:
        user_in = input("Press q to quit")
        transcript = handler.get_transcript()
        # print(f"transcript: {transcript}")
        if user_in == "q":
            get_message(transcript)

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
        blocksize=2048 * 2,
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

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=16000,
        media_encoding="pcm",
        show_speaker_label=True
    )

    # Instantiate our handler and start processing events
    # handler = MyEventHandler(stream.output_stream)
    handler = MyEventHandler(stream.output_stream)
    # Thread(target=listen_for_input, args=(handler,), daemon=True).start()
    await asyncio.gather(write_chunks(stream), handler.handle_events())
    # transcript = handler.get_transcript()
    # print(transcript)

loop = asyncio.get_event_loop()
loop.run_until_complete(basic_transcribe())
loop.close()
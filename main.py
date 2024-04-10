import whisper
from pyannote.audio import Pipeline
from utils import diarize_text
import torch
from collections import namedtuple
import time
from moviepy.editor import VideoFileClip

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
)
pipeline.to(torch.device("mps"))

# Video
video = VideoFileClip("fridman_altman_podcast_sample.mp4")
video.audio.write_audiofile("test.wav")
print("Audio extracted")

# Pyannote
start = time.time()
print("Starting diarization")
diarization_result = pipeline("test.wav")
print("Diarization took", time.time() - start, "seconds")

# Whisper
model = whisper.load_model("small.en")
asr_result = model.transcribe(
    "test.wav"
    # "fridman_altman_podcast_sample.wav",
    # initial_prompt="This audio is a conversation with keywords: AI, machine learning, deep learning, data science, LLMs.",
)
# print(asr_result)

# Combine
final_result = diarize_text(asr_result, diarization_result)

for seg, spk, sent in final_result:
    line = f"{seg.start:.2f} {seg.end:.2f} {spk} {sent}"
    print(line)
# Interval = namedtuple("Interval", ["start", "end", "speaker"])
# intervals = [
#     Interval(start=turn.start, end=turn.end, speaker=speaker)
#     for turn, _, speaker in diarization_result.itertracks(yield_label=True)
# ]

# i = 0
# while i < len(intervals) - 1:
#     if (
#         intervals[i].speaker == intervals[i + 1].speaker
#         and intervals[i + 1].start - intervals[i].end < 1
#     ):
#         intervals[i] = Interval(
#             start=intervals[i].start,
#             end=intervals[i + 1].end,
#             speaker=intervals[i].speaker,
#         )
#         intervals.pop(i + 1)
#     else:
#         i += 1

# for interval in intervals:
#     print(interval)

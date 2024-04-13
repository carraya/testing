import whisper
from pyannote.audio import Pipeline
from utils import diarize_text
import torch
from collections import namedtuple
import time
from moviepy.editor import VideoFileClip


class VideoToTextModel:

    def __init__(self):
        self.model = whisper.load_model("small.en")

    def raw_transcribe(self, video_path):
        return self.model.transcribe(video_path)

    def sentence_transcribe(self, video_path):
        transcription = self.raw_transcribe(video_path)["segments"]

        intervals = [[]]
        for idx, segment in enumerate(transcription):
            intervals[-1].append(idx)
            if any([segment["text"].strip().endswith(i) for i in [".", "!", "?"]]):
                intervals.append([])

        segments = []
        for interval in intervals:
            if len(interval) == 0:
                continue
            segments.append(
                {
                    "start": transcription[interval[0]]["start"],
                    "end": transcription[interval[-1]]["end"],
                    "text": " ".join([transcription[i]["text"] for i in interval])
                    .strip()
                    .replace("  ", " "),
                }
            )
        return segments


# Testing
if __name__ == "__main__":
    model = VideoToTextModel()
    print(model.sentence_transcribe("fridman_altman_podcast_sample.mp4"))

from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")

# run the pipeline on an audio file
diarization = pipeline("Nikhil-Kamath.wav",min_speakers=2, max_speakers=5)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
   
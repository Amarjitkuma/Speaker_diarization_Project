import whisper
from pydub import AudioSegment
from pyannote.database.util import load_rttm
import os

# Load the Whisper model
model = whisper.load_model("base")

# Function to extract audio segment and transcribe it
def transcribe_segment(audio_path, start_time, end_time, model):
    audio = AudioSegment.from_wav(audio_path)
    segment = audio[start_time * 1000:end_time * 1000]
    segment.export("temp_segment.wav", format="wav")
    result = model.transcribe("temp_segment.wav")
    os.remove("temp_segment.wav")  # Remove temporary file after transcription
    return result["text"]

# Load diarization results from RTTM file
rttm_file = "audio.rttm"
rttm_data = load_rttm(rttm_file)
diarization = rttm_data[next(iter(rttm_data))]  # Assuming only one audio file in RTTM

# Prepare the output format
audio_path = "Nikhil-Kamath.wav"
output = []

# Process each segment and transcribe it
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start_time = segment.start
    end_time = segment.end
    transcript = transcribe_segment(audio_path, start_time, end_time, model)
    output.append({
        "speaker": speaker,
        "transcript": transcript,
        "start_time": start_time,
        "end_time": end_time
    })

# Print the results
for entry in output:
    print(f"Speaker {entry['speaker']}: {entry['transcript']}")
    print(f"Start time: {entry['start_time']}")
    print(f"End time: {entry['end_time']}")
    print()

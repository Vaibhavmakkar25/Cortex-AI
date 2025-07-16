# transcribe.py
import whisperx
import gc
import pandas as pd
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

def run_transcription(audio_path):
    """
    Takes an audio file path, transcribes it with speaker labels,
    and returns a formatted string of the transcript.
    """
    load_dotenv()
    device = "cpu"
    model_size = "base" 
    batch_size = 16
    compute_type = "int8"
    
    # 1. Transcribe
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    del model; gc.collect(); torch.cuda.empty_cache()

    # 2. Align
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a; gc.collect(); torch.cuda.empty_cache()

    # 3. Diarize
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token: raise ValueError("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.")
    diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarize_model.to(torch.device(device))
    
    waveform = torch.from_numpy(audio).unsqueeze(0)
    diarize_segments = diarize_model({"waveform": waveform, "sample_rate": 16000})

    diarize_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['turn', '_', 'speaker'])
    diarize_df['start'] = diarize_df['turn'].apply(lambda x: x.start)
    diarize_df['end'] = diarize_df['turn'].apply(lambda x: x.end)
    result = whisperx.assign_word_speakers(diarize_df, result)
    del diarize_model; gc.collect(); torch.cuda.empty_cache()

    # 4. Format and Return
    full_transcript_text = ""
    for segment in result["segments"]:
        speaker = segment.get('speaker', 'UNKNOWN')
        start_time, end_time = segment['start'], segment['end']
        text = segment['text'].strip()
        full_transcript_text += f"[{start_time:07.2f} - {end_time:07.2f}] {speaker}: {text}\n"
        
    return full_transcript_text
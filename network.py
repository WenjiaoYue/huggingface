import requests
import numpy as np
import pydub


def extract_audio(file_path, sr, x, normalized=False):
    """numpy array to wav"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2**15)
    else:
        y = np.int16(x)
    audio = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=2, channels=channels
    )
    audio.export(file_path, format="wav", bitrate="320k")


def post_message(model, audio):
    sr, x = audio
    extract_audio(f"{model}.wav", sr, x)
    with open(f"{model}.wav", "rb") as file:
        r = requests.post(
            "https://neuralstudio.eglb.intel.com/talkingbot/asr", files={"file": file}
        )
    response = r.text
    return response
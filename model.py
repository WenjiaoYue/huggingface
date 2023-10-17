import os
import time
import requests
import numpy as np
import gradio as gr
import pydub

NUM_MODELS = 2


class Model:
    def __init__(self, model) -> None:
        self.model = model
        self.input_text = None
        self.audio_file = None
        self.record = None
        self.output_text = None


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


def audio2Text(model, audio):
    start_time = time.time()
    res = post_message(model, audio)
    finish_tstamp = time.time() - start_time
    elapsed_time = f"✅generation elapsed time: {round(finish_tstamp, 4)}s"
    return [res, elapsed_time]


audio_example = [
    "whisper-tiny-onnx-int4",
    os.path.join(os.path.dirname(__file__), "andy_welcome.wav"),
    "whisper-tiny-onnx-fp32",
    os.path.join(os.path.dirname(__file__), "andy_welcome.wav"),
]

model_list = [
    "whisper-tiny-onnx-int4",
    "whisper-tiny-onnx-fp32",
]
model_object_list = [Model(model_list[i]) for i in range(NUM_MODELS)]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                for obj in model_object_list:
                    with gr.Group():
                        obj.input_text = gr.Textbox(label="model", value=[obj.model])
                        with gr.Tab("file"):
                            obj.audio_file = gr.Audio()
                        with gr.Tab("record"):
                            obj.record = gr.Microphone()

            submit_btn = gr.Button(value="Submit")

        with gr.Column(scale=2):
            with gr.Row():
                for obj in model_object_list:
                    obj.output_text = gr.Textbox()

    gr.Examples(
        [[i for i in audio_example]],
        sum([[obj.input_text, obj.audio_file] for obj in model_object_list], []),
        [obj.output_text for obj in model_object_list],
        audio2Text,
        cache_examples=True,
    )

    for obj in model_object_list:
        submit_btn.click(
            audio2Text,
            inputs=[obj.input_text, obj.audio_file],
            outputs=[obj.output_text],
        )

demo.launch(share=True)

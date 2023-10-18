import os
import time
import gradio as gr
from network import post_message

NUM_MODELS = 2


class Model:
    def __init__(self, model) -> None:
        self.model = model
        self.input_text = None
        self.audio_file = None
        self.record = None
        self.output_text = None
        self.selected_audio = None
        self.tab_audio = None
        self.tab_record = None

    def switch(self, x):
        self.selected_audio = x


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
                        with gr.Tab("file") as obj.tab_audio:
                            obj.audio_file = gr.Audio()
                            obj.selected_audio = obj.audio_file  # for default
                        with gr.Tab("record") as obj.tab_record:
                            obj.record = gr.Microphone()

                        # check which tab is selected
                        obj.tab_audio.select(obj.switch, inputs=[obj.audio_file])
                        obj.tab_record.select(obj.switch, inputs=[obj.record])

            submit_btn = gr.Button(value="Submit")

        with gr.Column(scale=2):
            with gr.Row():
                for obj in model_object_list:
                    obj.output_text = gr.Textbox()

    gr.Examples(
        [[i for i in audio_example]],
        sum([[obj.input_text, obj.audio_file] for obj in model_object_list], []),
    )

    for obj in model_object_list:
        submit_btn.click(
            audio2Text,
            inputs=[obj.input_text, obj.selected_audio],
            outputs=[obj.output_text],
        )

demo.launch(share=True)

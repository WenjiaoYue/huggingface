import os
import gradio as gr

def audio2Text(audio1, audio2):
    return [1, 2]

audio_example = [["whisper-tiny-onnx-int4", os.path.join(os.path.dirname(__file__), "andy_welcome.wav"), 
                  os.path.join(os.path.dirname(__file__), "andy_welcome.wav")]]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Group():
                    text1 = gr.Textbox(label="model", value="whisper-tiny-onnx-int4")
                    with gr.Tab("file"):
                        audio1 = gr.Audio()
                    with gr.Tab("record"):
                        audio11 = gr.Microphone()
                with gr.Group():
                    text2 = gr.Textbox(label="model", value="whisper-tiny-onnx-fp32")
                    with gr.Tab("file"):
                        audio21 = gr.Audio()
                    with gr.Tab("record"):
                        audio21 = gr.Microphone()
        btn = gr.Button(value="Submit")

        with gr.Column(scale=2):
            with gr.Row():
                text3 = gr.Textbox()
                text4 = gr.Textbox()
    with gr.Row():
        gr.example()

    btn.click(audio2Text, inputs=[text1, audio1], outputs=[text3])
    btn.click(audio2Text, inputs=[text2], outputs=[text3])

demo.launch(share=True)


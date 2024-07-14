from transformers import pipeline
import io
import base64
import gradio as gr
cap_model = pipeline(task = 'image-to-text', model = "Salesforce/blip-image-captioning-base")

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format = 'PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = cap_model(base64_image)
    return result[0]['generated_text']

gr.close_all()
image_captioning = gr.Interface(
    fn = captioner,
    inputs = [gr.Image(label = "Upload Image", type = 'pil')],
    outputs = [gr.Textbox(label = 'Caption')],
    allow_flagging = 'never'
)

# Add Markdown content
markdown_content_img_cap = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Caption Any Image Using the BLIP model</h1>
        <h3 style='color: #4682B4;'>Model: Salesforce/blip-image-captioning-base</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
img_cap_with_markdown = gr.Blocks()
with img_cap_with_markdown:
    markdown_content_img_cap.render()
    image_captioning.render()
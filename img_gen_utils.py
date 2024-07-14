import gradio as gr
from diffusers import DiffusionPipeline

# Load the pipeline
gen_model = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
gen_model = gen_model.to('cpu')

# Define the function
def get_completion(prompt):
    return gen_model(prompt).images[0]    

# Create the Gradio interface
img_gen = gr.Interface(
    fn=get_completion,
    inputs=gr.Textbox(lines=2, placeholder="Enter a prompt here..."),
    outputs=gr.Image(type="pil"),
)

# Add Markdown content
markdown_content_img_gen = gr.Markdown(
    """
    <div style='text-align: center; font-family: "Times New Roman";'>
        <h1 style='color: #FF6347;'>Image Generation Using Stable Diffusion</h1>
        <h3 style='color: #4682B4;'>Model: runwayml/stable-diffusion-v1-5</h3>
        <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
    </div>
    """
)

# Combine the Markdown content and the demo interface
img_gen_with_markdown = gr.Blocks()
with img_gen_with_markdown:
    markdown_content_img_gen.render()
    img_gen.render()
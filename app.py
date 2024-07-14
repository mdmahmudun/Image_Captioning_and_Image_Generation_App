import warnings
import gradio as gr
warnings.filterwarnings('ignore')

from img_cap_utils import img_cap_with_markdown
from img_gen_utils import img_gen_with_markdown

# Combine both the app
demo = gr.Blocks()
with demo:
    gr.TabbedInterface(
        [img_cap_with_markdown, img_gen_with_markdown],
        ['Image Captioning', 'Image Generation']
    )


if __name__ == "__main__":
    demo.launch()
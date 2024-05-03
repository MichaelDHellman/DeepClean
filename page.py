import gradio as gr
import torch
from torchvision.transforms import ToPILImage
from model.model import ShallowUWNet  # Importing UWnet
from model.gen import buildUNET
import numpy as np
from skimage import io, transform
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = ShallowUWNet()  # Initialize the ShallowUWNet model
    state_dict = torch.load("./checkpoints/shallowUWNet.ckpt")
    model.load_state_dict(state_dict)#loading the state of the trained UWnet
    model.eval()
    return model

model_suwnet = load_model().to(device)  # Load the model


#Loading the CGAN model
model_cgan = buildUNET().to(device)
model_cgan.load_state_dict(torch.load("./checkpoints/cGANGen.ckpt"))#loading the state of the trained UWnet
model_cgan.eval()

class Rescale(object):

    def __init__(self, out_dims):
        self.out_dims = out_dims

    def __call__(self, sample):
        new_h, new_w = self.out_dims

        sample = transform.resize(sample, (new_h, new_w))

        return sample

class ToTensor(object):

    def __call__(self, sample):
        sample = sample.transpose((2,0,1))
        return torch.from_numpy(sample).type(torch.FloatTensor)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        #for t, m, s in zip(tensor, self.mean, self.std):
        #    t.mul_(s).add_(m)
        #return tensor
        return tensor*255.0
        


unorm = UnNormalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
doRescale = Rescale((256,256))
tensorfy = ToTensor()

def preprocess(image):
    image = doRescale(image)
    image = tensorfy(image)
    return image.to(device).unsqueeze(0)

def postprocess(tensor):
    image = unorm(tensor).squeeze(0)
    image = ToPILImage()(image)
    return image

def shallowuwnet(input_img):
    input_img = preprocess(input_img)
    print(input_img)
    with torch.no_grad():
        denoised_img = model_suwnet(input_img)
        torchvision.utils.save_image(denoised_img, "./data/output/" + "testsetset" + "_" + ".png", format = "png")
    return postprocess(denoised_img)


def cgan_model(input_img):
    input_img = preprocess(input_img) 
    with torch.no_grad():
        denoised_img = model_cgan(input_img)
        torchvision.utils.save_image(denoised_img, "./data/output/" + "testsetset" + "_" + ".png", format = "png")
    return postprocess(denoised_img)


css = """
.gradio-container {background-color : #133A35;}
.full-width-image img {
    width: 100%;  /* Set full width */
    aspect-ratio: 1 / 1;  /* Maintain aspect ratio for vertical elongation */
    padding: 10px;  /* Padding around images */
}
.gradio-radio {
    /* Styles for the radio buttons container  */
    background-color : #133A35;
    padding: 10px
    border-radius: 5px;
    color: white;
}

"""

with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.green)) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="LOGO3.jpg", elem_id="logo", show_download_button=False, container=False)
            gr.Markdown("**Mentor:** Michael Hellman", elem_id="mentor-text")
            gr.Markdown("**Mentees:** Senay, Mihir, Nicholas, Deepansh", elem_id="mentees-text")
        with gr.Column(scale=5):
            model_selector = gr.Radio( 
                ["ShallowUwnet", "CGAN Powered Uwnet"],
                label="Please Select a Model that suits your needs",
                value="ShallowUwnet",
                container = False
            )
            gr.Markdown("""
            * **ShallowUwnet**: A lightweight model designed for quick and efficient image cleaning with minimal computational requirements.
            - Memory Overhead:
            - Speed:
            - Efficiency:
            * **CGAN Powered Uwnet**: A more powerful model utilizing Conditional Generative Adversarial Networks to deliver high-quality denoising results.
            - Memory Overhead:
            - Speed:
            - Efficiency:
            """)    
    with gr.Row():
        input_image = gr.Image(label="Upload Image", type="numpy", elem_id="input-image", elem_classes="full-width-image" , container = False)
        output_image = gr.Image(label="Denoised Image", elem_id="output-image", elem_classes="full-width-image", container = False)
    with gr.Row():
        submit_btn = gr.Button("Process Image")

        #calibrate radio button to respond to model choice
        def process_image(image, model_choice):
            if model_choice == "ShallowUwnet":
                return shallowuwnet(image)
            else:
                return cgan_model(image)

        submit_btn.click(
            fn=process_image,
            inputs=[input_image, model_selector],
            outputs=output_image
        )

if __name__ == "__main__":
    demo.launch()

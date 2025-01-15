from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the model globally
model_name = "Salesforce/blip-image-captioning-base"  # model name
device = "cpu"  # uses cpu
processor = BlipProcessor.from_pretrained(model_name)  # load the processor
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)  # load the model for caption generation


def get_image_caption(image_path):  # function to get image caption
    image = Image.open(image_path).convert('RGB')  # convert the image into rgb format
    inputs = processor(image, return_tensors='pt').to(device)  # load the processor and return in pt format to device i.e. cpu
    output = model.generate(**inputs, max_new_tokens=10, num_beams=2, early_stopping=True)   # beam search for better caption generation
    caption = processor.decode(output[0], skip_special_tokens=True)  # decode the tensor into a string
    return caption  # return the generated caption


if __name__ == "__main__":
    image_path = "man.jpg"  # for testing purposes
    caption = get_image_caption(image_path)
    print(caption)

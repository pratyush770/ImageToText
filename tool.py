from langchain.tools import BaseTool
from function import get_image_caption


class ImageCaptionTool(BaseTool):  # custom tool
    name: str = "Image captioner"  # name of our tool with type annotation
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )  # description with type annotation

    def _run(self, img_path: str) -> str:
        caption = get_image_caption(img_path)  # function call
        return caption  # return the caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

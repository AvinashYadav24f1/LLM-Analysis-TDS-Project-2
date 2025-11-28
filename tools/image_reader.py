from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Vision-capable model
vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

@tool("read_image_text")
def read_image_text(image_url: str) -> str:
    """
    Given a public image URL, read and return the text written in the image.
    Use this when the quiz refers to a 'secret code in the image' or text inside an image.
    """
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Read the text in this image. "
                    "Return ONLY the exact code or phrase that appears, with no extra words."
                ),
            },
            {"type": "image_url", "image_url": image_url},
        ]
    )
    resp = vision_llm.invoke([message])
    return resp.text.strip()


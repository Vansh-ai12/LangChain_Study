#It is mainly for structured output for models like chatgpt , claulde and gemini and not opensource hugging face models

from langchain_huggingface import ChatHuggingFace ,  HuggingFacePipeline
from typing import TypedDict , Annotated , Optional , Literal
import os
from dotenv import load_dotenv      

load_dotenv()

os.environ['HF_HOME'] = 'C:/Projects/huggingface_model'


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task='text-generation',
    
)

model = ChatHuggingFace(llm = llm)

#Schema

class Review(TypedDict):
    key_themes :Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]    
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Return sentiment of the review either positive, negative or neutral"]
    pros: Annotated[Optional[list[str]],"Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]],"Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Name of the reviewer"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Review by Vansh Jain
""")

print(result)

print(result['summary'])
print(result['sentiment'])

#There is a posssibility that llm gives error and doesnt repond in str or not typedDict Speciefied field right 


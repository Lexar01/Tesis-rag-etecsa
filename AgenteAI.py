import warnings
warnings.filterwarnings(
    "ignore", 
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
    module="langchain_core"
)

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Tools import retriever

model = OllamaLLM(model="llama3.2") 

template = """
You are a knowledgeable professor. 
Only answer using the information provided in {reviews}. 
When answering, always include a short citation like: (from PDF: filename).
If the answer is not in the documents, say clearly: "No information available in the PDFs."
Highlight only the most essential points and explain them clearly.


Here are some relevant reviews: {reviews}

Here is the question to answer: {questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
 
while True:
    query = input("\nQue necesitas?")
    if query == "salir":
        break
    
    reviews = retriever.invoke(query)
    result = chain.invoke({"reviews":reviews,"questions": query })
    print(result)
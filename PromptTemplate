from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an expert assistant. Please answer:\n\nQuestion: {question}"
)

formatted_prompt = prompt.format(question="Who is the CEO of Amazon?")
print(formatted_prompt)

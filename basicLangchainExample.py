import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

# --- 1. SETUP: Hardcode API Key and LLM Initialization ---

# NOTE: Replace "YOUR_GEMINI_API_KEY_HERE" with your actual Gemini API Key.
# We are setting the environment variable here for a self-contained, hardcoded example.
# In production, use os.environ['GEMINI_API_KEY'] or secrets management like Streamlit.
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"

try:
    # Initialize the LLM model. gemini-2.5-flash is fast for this type of task.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7 # Add a little creativity
    )
except ValueError:
    print("ERROR: Failed to initialize ChatGoogleGenerativeAI. Please ensure your GEMINI_API_KEY is correct.")
    exit()

# --- 2. THE HARDCODED INPUT AND TEMPLATE ---

# The specific user prompt/query is hardcoded in a variable
USER_CONCEPT = "The concept of 'vendor lock-in' in cloud computing."

# The system prompt / instructions are hardcoded into the template
template = """
You are a concise business analyst. Your task is to explain the provided concept.
1. Provide a 1-sentence definition.
2. Provide one real-world example from technology.

Concept to Explain: {concept}
"""

# Create the PromptTemplate, which structures the input for the LLM
prompt = PromptTemplate.from_template(template)


# --- 3. CONSTRUCT THE CHAIN (LCEL) ---

# The chain defines the flow of data:
# 1. 'prompt': Takes the input and formats it according to the template.
# 2. '| llm': Sends the formatted prompt to the Gemini model.
# 3. '| StrOutputParser()': Converts the LLM's complex response object into a simple string.
chain = prompt | llm | StrOutputParser()

# --- 4. EXECUTION ---

print(f"--- Running LangChain Example ---")
print(f"**Input Concept:** {USER_CONCEPT}")
print(f"**Instructions:** Concise business analyst, 1-sentence definition, 1 example.")
print("-" * 35)

# The .invoke() call runs the entire chain pipeline end-to-end
result = chain.invoke({"concept": USER_CONCEPT})

# --- 5. OUTPUT ---

print("\n**Generated Output:**")
print(result)
print("\n--- Execution Complete ---")
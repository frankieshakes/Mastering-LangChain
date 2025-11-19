import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- Setup: Replace with your actual setup ---

# NOTE: In a real application, you would load this from environment variables.
# You will need to install the 'langchain-google-genai' package.
# pip install langchain-google-genai
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"

# Initialize the LLM and Embeddings model
# Using gemini-2.5-flash for speed and versatility
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except ValueError:
    print("WARNING: GEMINI_API_KEY environment variable not set. Code will not execute fully.")
    llm = None
    embeddings = None

# --- Example 1: Basic LLM Chain (LCEL) ---

def run_basic_chain(llm):
    """
    Demonstrates the simplest LCEL chain: Prompt | LLM | Output Parser.
    This ensures structured input and output.
    """
    print("\n--- 1. Basic LLM Chain (LCEL) ---")
    if not llm: return

    # 1. Define the Prompt Template (structured input)
    template = """
    You are an expert technical writer.
    Summarize the following programming concept in exactly three bullet points.
    CONCEPT: {concept}
    """
    prompt = PromptTemplate.from_template(template)

    # 2. Define the Output Parser (structured output)
    output_parser = StrOutputParser()

    # 3. Construct the Chain using LCEL (pipeline operator |)
    chain = prompt | llm | output_parser

    # 4. Invoke the chain
    result = chain.invoke({"concept": "Dependency Injection"})
    print(f"**Concept:** Dependency Injection\n\n**Summary:**\n{result}")

# --- Example 2: Retrieval-Augmented Generation (RAG) Chain ---

def run_rag_chain(llm, embeddings):
    """
    Demonstrates a RAG chain: Retrieval (Vector Store) -> Stuffing -> Generation (LLM).
    This grounds the answer in external, private knowledge.
    """
    print("\n--- 2. Retrieval-Augmented Generation (RAG) Chain ---")
    if not llm or not embeddings: return

    # 1. Setup Mock Knowledge Base (Vector Store)
    # In a real app, this would read from a file or database
    data = [
        "The project codename is 'Apollo'.",
        "The primary deployment pipeline uses GitHub Actions and is triggered by merges to the 'main' branch.",
        "Team members are required to attend the daily stand-up call at 9:00 AM PST.",
        "The current priority is optimizing the database query for the user profile service."
    ]

    # Create a simple vector store from the text chunks
    try:
        vectorstore = FAISS.from_texts(data, embedding=embeddings)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        print(f"Could not initialize Vector Store (FAISS or Embeddings error): {e}")
        return

    # 2. Define the Document Combination Chain
    # This chain takes the retrieved docs and stuffs them into the prompt
    rag_prompt = PromptTemplate.from_template("""
    You are a helpful team assistant. Answer the user's question only based on the following context.
    If the answer is not in the context, politely state that you do not have the information.

    CONTEXT:
    {context}

    QUESTION: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    # 3. Create the full RAG retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 4. Invoke the chain
    question = "What is the primary method for deploying the project codenamed 'Apollo'?"
    response = retrieval_chain.invoke({"input": question})

    print(f"**Question:** {question}")
    # The 'answer' key holds the final LLM response
    print(f"\n**RAG Answer:** {response['answer']}")
    # The 'context' key holds the source documents used for grounding
    print("\n**Source Documents Used:**")
    for doc in response['context']:
        print(f"- {doc.page_content}")

# --- Example 3: LLM Agent with Tool Calling ---

@tool
def check_pipeline_status(branch: str) -> str:
    """
    Checks the status of the continuous integration (CI) pipeline for a given branch.
    Input should be the branch name (e.g., 'main' or 'feature/new-login').
    Returns 'SUCCESS' or 'FAILURE' status.
    """
    if "main" in branch.lower():
        return "The 'main' branch pipeline is currently running and shows 'SUCCESS'."
    elif "test" in branch.lower():
        return "The 'test' branch pipeline shows 'FAILURE' due to a unit test failure."
    else:
        return f"Pipeline status for branch '{branch}' is 'NOT FOUND'."

def run_agent_executor(llm):
    """
    Demonstrates how an LLM can reason and decide to call a custom tool.
    """
    print("\n--- 3. LLM Agent with Tool Calling ---")
    if not llm: return

    tools = [check_pipeline_status]
    
    # 1. Define the Agent's Meta-Prompt/Persona
    # The model's system prompt dictates its reasoning and behavior
    prompt = PromptTemplate.from_template("""
    You are a specialized DevOps Agent. Your purpose is to assist developers 
    by checking the status of CI/CD pipelines.

    If the user asks about a pipeline status, you MUST use the 
    'check_pipeline_status' tool.

    If you successfully use the tool, state the final result clearly.
    If you cannot use the tool, state why.
    {agent_scratchpad}
    """)

    # 2. Create the Agent
    # This automatically includes the logic for deciding to call the tool.
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 3. Create the Executor
    # The executor runs the Thought->Action->Observation loop.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4. Invoke the Agent
    question = "What is the pipeline status for the 'main' branch?"
    print(f"**Agent Question:** {question}\n")
    
    result = agent_executor.invoke({"input": question})
    
    print(f"\n**Final Agent Response:** {result['output']}")


if __name__ == "__main__":
    if llm and embeddings:
        run_basic_chain(llm)
        run_rag_chain(llm, embeddings)
        run_agent_executor(llm)
    else:
        print("\n\nLLM or Embeddings model failed to initialize due to missing API key. Please set the GEMINI_API_KEY environment variable to run the examples.")
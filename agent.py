# Medical RAG Agent using LangChain Classic (Pure LangChain, no LangGraph)
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import search_pubmed, query_medical_knowledge, ingest_new_research
from core import get_medical_answer

def create_medical_agent():
    # Initialize Ollama LLM
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    
    # Define available tools
    tools = [search_pubmed, query_medical_knowledge, ingest_new_research]
    
    # ReAct prompt template with strict guardrails
    template = """You are a focused medical research assistant.
    Your goal is to answer medical questions using the provided tools.
    
    CRITICAL INSTRUCTION:
    If you do not find the answer in your knowledge base, you MUST use 'search_pubmed' to find new research.
    Do NOT give up without searching first.

    You must NOT answer general knowledge questions (like capitals, history) or creative writing requests (poems, stories).
    If a question is off-topic, your Final Answer should be: "I am designed to answer medical research questions only."

    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create executor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True
    )
    
    return agent_executor

def run_agent(query: str):
    """Run the medical agent with a user query."""
    agent = create_medical_agent()
    response = agent.invoke({"input": query})
    
    
    output = response.get("output", "")
    print(f"DEBUG: Agent Raw Output: '{output}'")
    
    # If agent hit iteration limit, use direct RAG as fallback
    # If agent hit iteration limit, use direct RAG as fallback
    if "iteration limit" in output.lower() or "time limit" in output.lower() or not output.strip():
        try:
            print("DEBUG: Hit iteration limit. Trying direct RAG.")
            answer, docs = get_medical_answer(query)
            if answer:
                # If RAG found an answer (and didn't say "cannot find"), return it
                if "cannot find information" not in answer.lower() and "medical database" not in answer.lower():
                    return answer
                else:
                    # If RAG also failed, update output so the next fallback block catches it
                    output = answer
                    print(f"DEBUG: Direct RAG also failed: '{output}'")
        except:
            pass
    
    # If agent output indicates failure to find info, try strict PubMed search as last resort
    if "cannot find information" in output.lower() or "medical database" in output.lower() or "i don't know" in output.lower():
        print(f"DEBUG: Agent failed. Triggering fallback search for: {query}")
        try:
            # Try searching PubMed directly
            search_result = search_pubmed.invoke(query)
            print(f"DEBUG: Search result length: {len(search_result)}")
            
            if "No results found" not in search_result:
                # If search found something, ask LLM directly to summarize it (safer than recursive agent call)
                llm = ChatOllama(model="llama3.1:8b", temperature=0)
                fallback_prompt = f"""You are a helpful medical assistant. 
Summarize the following medical research to answer the user's question.

User Question: {query}

Research Abstracts:
{search_result}

Answer:"""
                
                # Direct LLM call
                summary_response = llm.invoke(fallback_prompt)
                return summary_response.content
            else:
                return "DEBUG INFO: Fallback triggered, but PubMed search returned 'No results found'."
        except Exception as e:
            return f"DEBUG INFO: Fallback triggered but crashed: {e}"

    return output if output else "I couldn't find an answer. Try rephrasing your question."

# RAGAS Evaluation with Simple Test Data Generation
import nest_asyncio
nest_asyncio.apply()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from datasets import Dataset
from core import get_medical_answer

def get_documents_from_db():
    """Load documents from ChromaDB."""
    print("Loading documents from ChromaDB...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./medical_db",
        embedding_function=embeddings
    )
    
    # Get all documents from the database
    collection = vector_db._collection
    results = collection.get(include=["documents"])
    
    print(f"   Loaded {len(results['documents'])} documents")
    return results['documents']

def generate_questions_with_llm(documents, num_questions=5):
    """Generate test questions using LLM directly."""
    print(f"\n� Generating {num_questions} test questions using LLM...")
    
    llm = ChatOllama(model="llama3.1:8b", temperature=0.5)
    
    # Combine some documents for context
    sample_text = "\n\n".join(documents[:min(5, len(documents))])
    
    prompt = f"""Based on this medical context, generate {num_questions} question-answer pairs for testing a medical Q&A system.

Context:
{sample_text[:3000]}

Generate exactly {num_questions} questions with their answers in this format:
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
... and so on.

Focus on factual medical questions from the context."""

    response = llm.invoke(prompt)
    
    # Parse the response
    questions = []
    answers = []
    
    lines = response.content.split('\n')
    current_q = None
    current_a = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            current_q = line.split(':', 1)[1].strip()
        elif line.startswith('A') and ':' in line:
            current_a = line.split(':', 1)[1].strip()
            if current_q and current_a:
                questions.append(current_q)
                answers.append(current_a)
                current_q = None
                current_a = None
    
    print(f"   Generated {len(questions)} questions")
    
    for i, q in enumerate(questions[:5], 1):
        print(f"   Q{i}: {q[:60]}...")
    
    return questions[:num_questions], answers[:num_questions]

def run_evaluation_with_generated_data(num_questions=5):
    """Run RAGAS evaluation with auto-generated test data."""
    
    # Load documents
    documents = get_documents_from_db()
    
    if not documents:
        print("No documents found!")
        return None
    
    # Generate questions
    questions, ground_truths = generate_questions_with_llm(documents, num_questions)
    
    if not questions:
        print("Failed to generate questions!")
        return None
    
    print("\nRunning RAG pipeline on generated questions...")
    
    # Get answers from our RAG system
    rag_answers = []
    contexts = []
    
    for i, question in enumerate(questions):
        print(f"Processing Q{i+1}/{len(questions)}...")
        
        try:
            answer, docs = get_medical_answer(question)
            context = [doc.page_content for doc in docs]
        except Exception as e:
            print(f"   Error: {e}")
            answer = "Error getting answer"
            context = ["No context available"]
        
        rag_answers.append(str(answer))
        contexts.append(context if context else ["No context"])
    
    # Create evaluation dataset
    eval_dataset = Dataset.from_dict({
        "user_input": questions,
        "response": rag_answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    })
    
    print("\nCalculating RAGAS metrics...")
    
    # Initialize for evaluation
    llm = LangchainLLMWrapper(ChatOllama(model="llama3.1:8b", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    
    # Run evaluation
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60)
    
    try:
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            
            print("\nMean Scores:")
            for col in df.columns:
                if col not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
                    try:
                        mean_val = df[col].mean()
                        print(f"  {col}: {mean_val:.4f}")
                    except:
                        pass
            
            # Show questions and scores
            print("\nPer-Question Results:")
            score_cols = [c for c in df.columns if c not in ['user_input', 'response', 'retrieved_contexts', 'reference']]
            if score_cols:
                display_df = df[['user_input'] + score_cols[:2]]
                display_df['user_input'] = display_df['user_input'].str[:50] + '...'
                print(display_df.to_string(index=False))
        else:
            print(results)
    except Exception as e:
        print(f"Results: {results}")
        print(f"Error: {e}")
    
    print("="*60)
    
    return results

if __name__ == "__main__":
    run_evaluation_with_generated_data(num_questions=5)

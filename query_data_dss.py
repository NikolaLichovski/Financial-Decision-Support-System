import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os
import requests
from typing import Tuple, Optional, Dict, List

from financial_data import FinancialDataProvider
from preference_engine import PreferenceEngine

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"


def check_ollama_running():
    """Check if Ollama is running and has models available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return len(models) > 0, models
        return False, []
    except:
        return False, []


def query_financial_dss(
        query_text: str,
        ticker: str,
        preferences: Dict[str, str],
        use_rules: bool = True
) -> Tuple[str, List[str], Optional[Dict]]:
    """
    Enhanced DSS query integrating financial data, user preferences, and optional rules.

    Args:
        query_text: User's question
        ticker: Stock ticker symbol
        preferences: Dict with risk_tolerance, time_horizon, risk_behavior
        use_rules: Whether to retrieve and use rules from ChromaDB

    Returns:
        (response_text, sources, stock_summary)
    """

    # Step 1: Fetch and summarize stock data
    print(f"Fetching financial data for {ticker}...")
    financial_provider = FinancialDataProvider()
    stock_summary = financial_provider.get_stock_summary(ticker)

    if not stock_summary:
        return f"Unable to fetch data for ticker {ticker}. Please verify the ticker symbol.", [], None

    # Step 2: Initialize preference engine
    pref_engine = PreferenceEngine(
        risk_tolerance=preferences.get('risk_tolerance', 'Medium'),
        time_horizon=preferences.get('time_horizon', 'Long-term (>1yr)'),
        risk_behavior=preferences.get('risk_behavior', 'Risk-averse')
    )

    # Step 3: Format financial data with preference context
    financial_context = financial_provider.format_for_llm(stock_summary, preferences)

    # Step 4: Retrieve relevant rules if enabled
    rules_context = ""
    sources = []

    if use_rules and os.path.exists(CHROMA_PATH):
        print("Retrieving relevant rules from knowledge base...")
        rules_text, rule_sources = retrieve_rules(query_text, ticker)
        if rules_text:
            rules_context = f"\nRELEVANT RULES AND CONSTRAINTS:\n{rules_text}\n"
            sources.extend(rule_sources)

    # Step 5: Build DSS-specific prompt
    prompt = build_dss_prompt(
        query=query_text,
        financial_context=financial_context,
        rules_context=rules_context,
        preference_guidance=pref_engine.get_prompt_guidance()
    )

    # Step 6: Query LLM
    print("Generating DSS analysis...")
    response_text = query_llm_with_dss_prompt(prompt)

    return response_text, list(set(sources)), stock_summary


def retrieve_rules(query_text: str, ticker: str) -> Tuple[str, List[str]]:
    """
    Retrieve relevant rules/constraints from ChromaDB based on query and ticker.

    Returns:
        (rules_text, sources)
    """
    db = None
    try:
        # Initialize embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Load vector database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Construct search query combining user question and ticker
        search_query = f"{query_text} {ticker} investment rules constraints"

        # Retrieve relevant chunks
        docs = db.similarity_search(search_query, k=5)

        if not docs:
            return "", []

        # Extract and format rules
        rules_parts = []
        sources = []

        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", "Unknown")

            rules_parts.append(f"[Rule {i}] {content}")
            sources.append(source)

        rules_text = "\n\n".join(rules_parts)
        return rules_text, sources

    except Exception as e:
        print(f"Error retrieving rules: {e}")
        return "", []
    finally:
        # Ensure database connection is closed
        if db is not None:
            try:
                del db
            except:
                pass


def build_dss_prompt(
        query: str,
        financial_context: str,
        rules_context: str,
        preference_guidance: str
) -> str:
    """
    Build the comprehensive DSS prompt that enforces exploratory analysis behavior.
    """

    dss_system_prompt = """You are a financial decision support analyst. Your role is to provide structured, objective insights that help users understand investment characteristics and trade-offs—NOT to recommend specific actions.

        CORE PRINCIPLES:
        1. NEVER recommend "buy", "sell", or "hold"
        2. EXPLAIN implications and trade-offs, don't make judgments
        3. HIGHLIGHT alignments and misalignments with user preferences and rules
        4. FRAME historical data as contextual evidence, not predictions
        5. MAINTAIN exploratory tone: "This suggests..." not "You should..."
        
        ANALYSIS STRUCTURE:
        - Synthesize financial data with user preferences and constraints
        - Identify key considerations relevant to the user's question
        - Explain trade-offs between different characteristics
        - Flag any conflicts between stock characteristics and stated rules
        - Present information in a balanced, informative manner
        """

    prompt = f"""{dss_system_prompt}

        {preference_guidance}
        
        FINANCIAL DATA:
        {financial_context}
        
        {rules_context}
        
        USER QUESTION:
        {query}
        
        INSTRUCTIONS:
        Provide a comprehensive DSS analysis that:
        1. Directly addresses the user's question
        2. Interprets financial data through the lens of their preferences
        3. Highlights alignment or misalignment with any stated rules
        4. Explains relevant trade-offs and considerations
        5. Avoids making recommendations or decisions
        
        Your response should help the user understand the investment characteristics and how they relate to their stated preferences and constraints, while leaving the ultimate decision to them.
        
        ANALYSIS:
        """

    return prompt


def query_llm_with_dss_prompt(prompt: str) -> str:
    """
    Send the DSS prompt to the LLM and return the response.
    Tries Ollama first, falls back to HuggingFace, then to structured fallback.
    """

    # Try Ollama first (best option)
    ollama_running, models = check_ollama_running()

    if ollama_running and models:
        try:
            model_name = "llama3.2" if any("llama3.2" in m['name'] for m in models) else models[0]['name']
            print(f"Using Ollama model: {model_name}")

            llm = Ollama(
                model=model_name,
                temperature=0.4,  # Slightly higher for more nuanced analysis
                top_p=0.9,
            )

            response = llm.invoke(prompt)

            # Validate response quality
            if len(response.strip()) > 100:
                return response.strip()
            else:
                raise Exception("Response too short")

        except Exception as e:
            print(f"Ollama failed: {e}")

    # Fallback to HuggingFace (local models)
    try:
        print("Using local Hugging Face model...")
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

        model_name = "microsoft/DialoGPT-medium"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=min(len(prompt.split()) + 200, 1024),
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        result = generator(prompt)
        response = result[0]['generated_text'][len(prompt):].strip()

        if len(response) > 50:
            return response
        else:
            raise Exception("Generated response too short")

    except Exception as e:
        print(f"HuggingFace model failed: {e}")

    # Final fallback - structured analysis from context
    print("Using structured fallback analysis...")
    return generate_structured_fallback(prompt)


def generate_structured_fallback(prompt: str) -> str:
    """
    Generate a structured analysis when LLMs are unavailable.
    Extracts key information from the prompt context.
    """

    response_parts = [
        "DECISION SUPPORT ANALYSIS:",
        "",
        "Based on the provided financial data and your preferences, here are the key considerations:",
        ""
    ]

    # Extract key metrics from prompt (simplified pattern matching)
    lines = prompt.split('\n')

    # Find volatility
    for line in lines:
        if "Annualized Volatility:" in line:
            response_parts.append(f"• {line.strip()}")

    # Find returns
    for line in lines:
        if "Return" in line and "%" in line:
            response_parts.append(f"• {line.strip()}")

    # Find sector
    for line in lines:
        if "Sector:" in line:
            response_parts.append(f"• {line.strip()}")

    response_parts.extend([
        "",
        "This analysis is based on historical data and should be considered alongside your personal investment constraints and risk tolerance.",
        "",
        "For more detailed analysis, please ensure Ollama is running with a language model installed."
    ])

    return "\n".join(response_parts)


# Legacy function for backward compatibility with original RAG functionality
def query_rag(query_text: str) -> Tuple[str, List[str]]:
    """
    Original RAG query function maintained for backward compatibility.
    For financial DSS queries, use query_financial_dss instead.
    """

    # Check if database exists
    if not os.path.exists(CHROMA_PATH):
        return "Database not found. Please run create_database.py first to create the vector database.", []

    db = None
    try:
        # Initialize embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Load vector database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Test retrieval first
        docs = db.similarity_search(query_text, k=5)
        if not docs:
            return "No relevant documents found for your query.", []

        print(f"Found {len(docs)} relevant documents")

        # Try Ollama first
        ollama_running, models = check_ollama_running()

        if ollama_running and models:
            try:
                model_name = "llama3.2" if any("llama3.2" in m['name'] for m in models) else models[0]['name']
                print(f"Using Ollama model: {model_name}")

                llm = Ollama(
                    model=model_name,
                    temperature=0.3,
                    top_p=0.9,
                )

                template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
                    Always provide a complete, well-structured answer based on the context.
                    
                    Context:
                    {context}
                    
                    Question: {question}
                    
                    Answer: """

                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=db.as_retriever(search_kwargs={"k": 5}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )

                result = qa_chain.invoke({"query": query_text})
                response_text = result["result"]
                sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

                return response_text, list(set(sources))

            except Exception as e:
                print(f"Ollama failed: {e}")

        # Fallback behavior (simplified for brevity)
        context = "\n\n".join([doc.page_content[:300] for doc in docs[:3]])
        return f"Based on the documents: {context[:500]}...", [doc.metadata.get("source", "Unknown") for doc in docs]

    finally:
        # Ensure database connection is closed
        if db is not None:
            try:
                del db
            except:
                pass


def main():
    """CLI interface for testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--risk", type=str, default="Medium", choices=["Low", "Medium", "High"])
    parser.add_argument("--horizon", type=str, default="Long-term (>1yr)")
    parser.add_argument("--behavior", type=str, default="Risk-averse", choices=["Risk-averse", "Risk-seeking"])
    parser.add_argument("--no-rules", action="store_true", help="Disable rules retrieval")

    args = parser.parse_args()

    preferences = {
        'risk_tolerance': args.risk,
        'time_horizon': args.horizon,
        'risk_behavior': args.behavior
    }

    response, sources, summary = query_financial_dss(
        args.query_text,
        args.ticker,
        preferences,
        use_rules=not args.no_rules
    )

    print("\n" + "=" * 80)
    print("DSS ANALYSIS RESPONSE:")
    print("=" * 80)
    print(response)
    print("\n" + "=" * 80)
    print("SOURCES:")
    print("=" * 80)
    for source in sources:
        print(f"  • {source}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
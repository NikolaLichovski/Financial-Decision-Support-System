import streamlit as st
import os
import shutil
from create_database import generate_data_store
from query_data_dss import query_financial_dss, query_rag
from document_loader import DocumentLoader

# Page config
st.set_page_config(
    page_title="Financial Decision Support System",
    page_icon="📊",
    layout="wide"
)

# Constants
DATA_PATH = "data/docs"
CHROMA_PATH = "chroma"


def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    file_path = os.path.join(DATA_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def check_ollama_status():
    """Check Ollama status and return status info"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, models
        return False, []
    except:
        return False, []


def main():
    # Initialize session state
    if 'dss_mode' not in st.session_state:
        st.session_state.dss_mode = True
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'clear_data_flag' not in st.session_state:
        st.session_state.clear_data_flag = False

    # Clean up any old renamed folders from previous sessions
    import glob
    for old_folder in glob.glob("chroma_old_*") + glob.glob("data_old_*"):
        try:
            shutil.rmtree(old_folder, ignore_errors=True)
        except:
            pass

    # Handle data clearing at the start of the script run
    if st.session_state.clear_data_flag:
        st.session_state.clear_data_flag = False

        import time
        import uuid

        success = True
        errors = []

        # Strategy: RENAME folders instead of deleting (Windows allows this even with locked files)
        # Then try to delete the renamed folders

        timestamp = uuid.uuid4().hex[:8]

        # Handle DATA_PATH
        if os.path.exists(DATA_PATH):
            try:
                old_data = f"data_old_{timestamp}"
                os.rename(DATA_PATH, old_data)
                time.sleep(0.1)
                # Try to delete the renamed folder
                try:
                    shutil.rmtree(old_data, ignore_errors=True)
                except:
                    pass  # Will be cleaned up next time
            except Exception as e:
                errors.append(f"Could not clear data folder: {e}")
                success = False

        # Handle CHROMA_PATH
        if os.path.exists(CHROMA_PATH):
            try:
                old_chroma = f"chroma_old_{timestamp}"
                os.rename(CHROMA_PATH, old_chroma)
                time.sleep(0.1)
                # Try to delete the renamed folder
                try:
                    shutil.rmtree(old_chroma, ignore_errors=True)
                except:
                    pass  # Will be cleaned up next time
            except Exception as e:
                errors.append(f"Could not clear chroma folder: {e}")
                success = False

        if success:
            st.success("✅ All data cleared successfully!")
        else:
            st.error("❌ Failed to clear data:")
            for error in errors:
                st.error(f"  • {error}")

        time.sleep(1)

    st.title("📊 Financial Decision Support System")
    st.markdown("*Exploratory analysis for informed investment decisions*")

    # Mode toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        mode = st.radio(
            "Mode",
            ["DSS Mode", "Document Q&A"],
            help="DSS Mode: Financial analysis with preferences. Document Q&A: General document questions."
        )
        st.session_state.dss_mode = (mode == "DSS Mode")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # DSS-specific inputs
        if st.session_state.dss_mode:
            st.subheader("📈 Stock Selection")
            ticker = st.text_input(
                "Stock Ticker",
                value="AAPL",
                help="Enter stock ticker symbol (e.g., AAPL, MSFT, TSLA)",
                key="ticker_input"
            ).upper()

            st.subheader("👤 Your Preferences")
            st.markdown("*These shape how information is interpreted*")

            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["Low", "Medium", "High"],
                value="Medium",
                help="How much volatility can you accept?",
                key="risk_tolerance"
            )

            time_horizon = st.radio(
                "Investment Horizon",
                ["Short-term (<1yr)", "Long-term (>1yr)"],
                index=1,
                help="When do you plan to evaluate/exit this position?",
                key="time_horizon"
            )

            risk_behavior = st.radio(
                "Risk Approach",
                ["Risk-averse", "Risk-seeking"],
                index=0,
                help="How do you view uncertainty—as risk or opportunity?",
                key="risk_behavior"
            )

            # Preference summary
            with st.expander("📋 Your Profile Summary"):
                st.markdown(f"""
                **Risk Tolerance:** {risk_tolerance}  
                **Time Horizon:** {time_horizon}  
                **Risk Approach:** {risk_behavior}

                *These preferences actively shape how the analysis is framed, 
                not just reported back to you.*
                """)

            st.subheader("📋 Rules & Constraints")
            use_rules = st.checkbox(
                "Enable rules checking",
                value=True,
                help="Use uploaded rules document to check alignment with constraints"
            )

            if use_rules and not os.path.exists(CHROMA_PATH):
                st.warning("⚠️ No rules document found. Upload and process a rules document below.")

        # Document management (both modes)
        st.subheader("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents" + (" (Rules/Policies)" if st.session_state.dss_mode else ""),
            type=['txt', 'md', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: TXT, MD, PDF, DOCX"
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")

            if st.button("💾 Save & Process Documents"):
                with st.spinner("Saving and processing documents..."):
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        save_uploaded_file(uploaded_file)

                    # Generate database
                    try:
                        generate_data_store()
                        st.success("✅ Documents processed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing documents: {e}")

        # Show existing files
        st.markdown("**Current Documents:**")
        if os.path.exists(DATA_PATH):
            files = os.listdir(DATA_PATH)
            if files:
                for file in files:
                    st.text(f"• {file}")
            else:
                st.info("No documents uploaded yet")
        else:
            st.info("No documents uploaded yet")

        # System status
        st.subheader("🔧 System Status")

        # Ollama status
        ollama_running, models = check_ollama_status()
        if ollama_running and models:
            st.success(f"✅ Ollama running ({len(models)} model(s))")
            with st.expander("Available models"):
                for model in models:
                    st.text(f"• {model['name']}")
        else:
            st.warning("⚠️ Ollama not running - will use fallback")
            st.info("Install Ollama for best results!")

        # Database status
        if os.path.exists(CHROMA_PATH):
            st.success("✅ Knowledge base ready")
        else:
            st.warning("⚠️ No knowledge base")

        # Clear data button
        if st.button("🗑️ Clear All Data", type="secondary"):
            # Set flag and force rerun - this ensures all DB connections are closed
            st.session_state.clear_data_flag = True

            # Force close any ChromaDB connections before rerun
            import sys
            import gc

            # Remove chromadb from loaded modules to force cleanup
            modules_to_remove = [key for key in sys.modules.keys() if 'chroma' in key.lower()]
            for module in modules_to_remove:
                del sys.modules[module]

            gc.collect()

            st.warning("⏳ Clearing data... Page will refresh automatically.")
            st.rerun()

    # Main interface
    if st.session_state.dss_mode:
        # DSS Mode Interface
        st.header("💬 Financial Analysis Query")

        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - How does {stock} align with my risk tolerance and investment horizon?
            - What are the key risk considerations for {stock} given my preferences?
            - Does {stock} match my stated investment constraints?
            - What trade-offs should I consider with {stock}?
            - How volatile is {stock} relative to my risk profile?

            *Replace {stock} with your ticker or just refer to "this stock"*
            """)

        # Query input
        query = st.text_area(
            "Your question:",
            placeholder=f"How does {ticker if 'ticker' in locals() else 'this stock'} align with my risk tolerance and constraints?",
            help="Ask about alignment, trade-offs, or characteristics",
            height=100
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

        if analyze_button and query:
            with st.spinner(f"Analyzing {ticker} with your preferences..."):
                try:
                    # Gather preferences
                    preferences = {
                        'risk_tolerance': risk_tolerance,
                        'time_horizon': time_horizon,
                        'risk_behavior': risk_behavior
                    }

                    # Run DSS query
                    response, sources, stock_summary = query_financial_dss(
                        query_text=query,
                        ticker=ticker,
                        preferences=preferences,
                        use_rules=use_rules
                    )

                    # Display results
                    st.subheader("📊 Analysis Results")

                    # Stock summary card
                    if stock_summary:
                        with st.expander(f"📈 {ticker} Summary", expanded=True):
                            col1, col2, col3 = st.columns(3)

                            basic = stock_summary['basic_info']
                            risk = stock_summary['risk_metrics']
                            perf = stock_summary['performance']

                            with col1:
                                st.metric("Current Price", f"${basic['current_price']}")
                                st.metric("Sector", basic['sector'])

                            with col2:
                                st.metric("Volatility", f"{risk['volatility_annual']}%")
                                st.metric("Risk Class", risk['risk_classification'])

                            with col3:
                                if perf['return_1y']:
                                    st.metric("1Y Return", f"{perf['return_1y']:+.2f}%")
                                if basic['dividend_yield'] > 0:
                                    st.metric("Dividend", f"{basic['dividend_yield']}%")

                    # Main analysis response
                    st.markdown("### 🎯 Decision Support Analysis")
                    st.markdown(response)

                    # Sources
                    if sources:
                        with st.expander("📚 Information Sources"):
                            st.markdown("This analysis used:")
                            st.markdown(f"- Financial data for {ticker} (via yfinance)")
                            if len(sources) > 0:
                                st.markdown("- Rules/constraints from uploaded documents:")
                                for source in sources:
                                    st.markdown(f"  - {os.path.basename(source)}")

                    # Important disclaimer
                    st.info(
                        "ℹ️ This is exploratory analysis to support your decision-making process. It does not constitute investment advice or recommendations.")

                except Exception as e:
                    st.error(f"Error generating analysis: {e}")
                    st.info("Ensure Ollama is running for best results, or check your ticker symbol.")

        elif analyze_button and not query:
            st.warning("Please enter a question to analyze.")

    else:
        # Document Q&A Mode (original functionality)
        st.header("💬 Chat with Your Documents")

        # Check if database exists
        if not os.path.exists(CHROMA_PATH):
            st.warning("⚠️ Please upload and process documents first using the sidebar.")
            return

        # Query input
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?",
            help="Type your question and press Enter"
        )

        if query:
            with st.spinner("Searching and generating response..."):
                try:
                    response, sources = query_rag(query)

                    # Display response
                    st.subheader("📝 Response")
                    st.write(response)

                    # Display sources
                    st.subheader("📚 Sources")
                    if sources:
                        for i, source in enumerate(sources, 1):
                            display_source = os.path.basename(source) if source != "Unknown" else "Unknown"
                            st.text(f"{i}. {display_source}")
                    else:
                        st.text("No sources found")

                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Instructions footer
    with st.expander("ℹ️ How to use this system"):
        if st.session_state.dss_mode:
            st.markdown("""
            ## Financial DSS Mode

            ### Setup:
            1. **Install Ollama** (recommended): Download from [ollama.ai](https://ollama.ai) and run `ollama pull llama3.2`
            2. **Configure preferences**: Set your risk tolerance, time horizon, and risk behavior in the sidebar
            3. **Optional**: Upload a rules/constraints document (e.g., investment policy, personal guidelines)
            4. **Enter stock ticker**: Specify which stock to analyze

            ### Using the System:
            - Ask questions about how stocks align with your preferences
            - Explore trade-offs and risk characteristics
            - Check alignment with uploaded rules/constraints
            - The system provides *exploratory analysis*, not recommendations

            ### What This System Does:
            ✅ Explains stock characteristics in context of your preferences  
            ✅ Highlights alignment/misalignment with stated rules  
            ✅ Presents trade-offs to inform your decisions  
            ✅ Frames historical data as contextual evidence  

            ### What This System Does NOT Do:
            ❌ Make buy/sell/hold recommendations  
            ❌ Predict future prices  
            ❌ Make decisions for you  
            ❌ Provide investment advice  

            ### Why Preferences Matter:
            Your preferences don't just appear in the output—they actively shape how information is interpreted and presented. The same volatility might be framed as "risk" for conservative users or "opportunity" for aggressive ones.
            """)
        else:
            st.markdown("""
            ## Document Q&A Mode

            ### Setup:
            1. Upload documents using the sidebar (TXT, MD, PDF, DOCX)
            2. Click "Save & Process Documents" to create the knowledge base
            3. Ask questions about your documents

            ### Features:
            - Semantic search finds relevant passages
            - AI generates comprehensive answers
            - Sources are cited for transparency

            This mode is for general document question-answering, not financial analysis.
            """)


if __name__ == "__main__":
    main()
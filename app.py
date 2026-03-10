import gradio as gr
import hashlib
from typing import List, Dict
import os
import uuid

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger

# 1) Define some example data 
#    (i.e., question + paths to documents relevant to that question).
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]  
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}

def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    # Define custom CSS for professional UI Styling
    css = """
    /* Body and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
    }

    /* Hero Banner Area */
    .hero-banner {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 40px 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .hero-banner .title {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
        border: none !important;
    }

    .hero-banner .subtitle {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        margin-top: 0 !important;
    }

    /* Panels */
    .sidebar-panel, .main-panel {
        border-radius: 12px;
        padding: 20px;
        border: 1px solid var(--border-color-primary);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Answer Output Styling */
    .answer-box textarea {
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    
    /* Submit button */
    .submit-btn {
        background: linear-gradient(to right, #3b82f6, #2563eb) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    .submit-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }
    """

    # We use a built-in clean theme like Default, combined with our CSS overrides
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), title="Lumina RAG", css=css) as demo:
        
        # --- HERO BANNER ---
        with gr.Column(elem_classes=["hero-banner"]):
            gr.Markdown("# ✨ Lumina RAG", elem_classes=["title"])
            gr.Markdown("Powered by Docling, LangGraph, and DeepSeek. An Agentic Document Assistant.", elem_classes=["subtitle"])

        # 2) Maintain the session state for retrieving doc changes
        session_state = gr.State({
            "session_id": str(uuid.uuid4()),
            "file_hashes": frozenset(),
            "retriever": None
        })

        # --- MAIN LAYOUT ---
        with gr.Row():
            
            # --- LEFT SIDEBAR (Controls & Config) ---
            with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
                gr.Markdown("### ⚙️ Input & Configuration")
                
                gr.Markdown("#### 1. Try an Example")
                example_dropdown = gr.Dropdown(
                    label="Select a ready-to-test document",
                    choices=list(EXAMPLES.keys()),
                    value=None,  
                )
                load_example_btn = gr.Button("Load Example 📂", variant="secondary")

                gr.Markdown("---")
                
                gr.Markdown("#### 2. Or Upload Yours")
                gr.Markdown("<small>Supported forms: PDF, DOCX, TXT, MD.</small>")
                files = gr.Files(label="Upload Documents", file_types=constants.ALLOWED_TYPES)
                
            # --- RIGHT MAIN AREA (Chat & Results) ---
            with gr.Column(scale=2, elem_classes=["main-panel"]):
                gr.Markdown("### 💬 Ask Lumina")
                question = gr.Textbox(
                    label="What would you like to know from the documents?", 
                    placeholder="E.g., What are the key takeaways from page 4?",
                    lines=3
                )
                
                submit_btn = gr.Button("Submit Query 🚀", elem_classes=["submit-btn"], size="lg")
                
                gr.Markdown("---")
                
                gr.Markdown("### 🧠 AI Response")
                answer_output = gr.Textbox(
                    label="Final Answer", 
                    interactive=False, 
                    lines=8,
                    elem_classes=["answer-box"]
                )
                
                with gr.Accordion("🔍 View Agent Verification Report (Self-Correction)", open=False):
                    gr.Markdown("Lumina uses an internal Verification Agent to double-check its own answer against facts before responding.")
                    verification_output = gr.Textbox(label="Report details", interactive=False, lines=4)

        # --- EVENT HANDLERS ---
        
        # Load example
        def load_example(example_key: str):
            if not example_key or example_key not in EXAMPLES:
                return [], "" 

            ex_data = EXAMPLES[example_key]
            question = ex_data["question"]
            file_paths = ex_data["file_paths"]

            loaded_files = []
            for path in file_paths:
                if os.path.exists(path):
                    loaded_files.append(path)
                else:
                    logger.warning(f"File not found: {path}")

            return loaded_files, question

        load_example_btn.click(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[files, question]
        )

        # Smart reset: only rebuild when files actually change (by content hash)
        def on_files_changed(uploaded_files, state):
            if not uploaded_files:
                # Files were cleared
                state.update({"file_hashes": frozenset(), "retriever": None})
                return state
            
            new_hashes = _get_file_hashes(uploaded_files)
            if new_hashes != state["file_hashes"]:
                logger.info("New files detected — resetting retriever for rebuild.")
                state.update({"file_hashes": frozenset(), "retriever": None})
            else:
                logger.info("Same files re-uploaded — reusing existing retriever.")
            return state

        files.change(
            fn=on_files_changed,
            inputs=[files, session_state],
            outputs=[session_state]
        )

        # Process question
        def process_question(question_text: str, uploaded_files: List, state: Dict, progress=gr.Progress()):
            try:
                if not question_text.strip():
                    yield "❌ Question cannot be empty", "", state
                    return
                if not uploaded_files:
                    yield "❌ No documents uploaded", "", state
                    return

                # Ensure session_id exists (for backwards compatibility with old states)
                if "session_id" not in state:
                    state["session_id"] = str(uuid.uuid4())
                session_id = state["session_id"]

                current_hashes = _get_file_hashes(uploaded_files)
                
                if state.get("retriever") is None or current_hashes != state.get("file_hashes"):
                    progress(0.1, desc="Reading documents...")
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)
                    
                    if not chunks:
                        yield "❌ No readable text was extracted from the documents.", "", state
                        return

                    progress(0.4, desc="Building Search Database...")
                    retriever = retriever_builder.build_hybrid_retriever(chunks, session_id=session_id)
                    
                    state.update({
                        "file_hashes": current_hashes,
                        "retriever": retriever
                    })
                else:
                    logger.info("Using cached retriever (same documents).")
                
                progress(0.7, desc="Analyzing and generating answer (this may take a moment)...")
                
                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"]
                )
                
                progress(1.0, desc="Done!")
                yield result["draft_answer"], result["verification_report"], state
                    
            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                yield f"❌ Error: {str(e)}", "", state

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state]
        )

    # Launch without hardcoding a port so we don't conflict with any existing servers
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=css, share=True)

def _get_file_hashes(uploaded_files: List) -> frozenset:
    hashes = set()
    for file in uploaded_files:
        file_path = file if isinstance(file, str) else file.name
        with open(file_path, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)

if __name__ == "__main__":
    main()

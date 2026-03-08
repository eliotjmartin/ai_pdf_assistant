import os
import sys
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# load .env before importing our custom modules so they can see the API keys
load_dotenv()

# add the project root to sys.path so we can import from 'src'
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# import from source or crash with useful message
try:
    from src.ingest import ingest_pdf
    from src.retrieve_and_answer import retrieve_and_answer
except ImportError as e:
    print(f"Error: Could not find src modules. Ensure your directory structure is correct. {e}")
    sys.exit(1)

def process_upload(file_obj):
    """Handles the file upload and ingestion process."""
    if file_obj is None:
        return "Error: No file uploaded."
    try:
        # file_obj.name is the local path to the temp file Gradio created
        result_message = ingest_pdf(file_obj.name)
        return f"Ingestion Succeeded: {result_message}"
    except Exception as e:
        return f"Ingestion Failed: {str(e)}"

def chat_logic(query):
    """Handles the RAG retrieval and LLM answering."""
    if not query.strip():
        return "Please enter a question.", "No sources found."
    try:
        # calls logic which returns (answer, sources_string)
        answer, sources = retrieve_and_answer(query)
        return answer, sources
    except Exception as e:
        return f"Error during retrieval: {str(e)}", "N/A"

# sets the aesthetic
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Explorer") as demo:
    gr.Markdown("# AI PDF Assistant")
    gr.Markdown("Upload a PDF to the system, then ask questions based on its content.")

    with gr.Tabs():
        # first tab is ingestion
        with gr.Tab("1. Setup (Ingest PDF)"):
            with gr.Row():
                with gr.Column(scale=2): # scale controls the width of the columns
                    pdf_input = gr.File(label="Upload Document", file_types=[".pdf"])
                    upload_btn = gr.Button("Process & Index Document", variant="primary")
                with gr.Column(scale=1):
                    status_out = gr.Textbox(label="Ingestion Status", interactive=False)

            # when clicked, take the data from box A, run it through function B, and 
            # show the result in box C
            upload_btn.click(
                fn=process_upload,
                inputs=[pdf_input],
                outputs=[status_out]
            )

        # second tab is for querying
        with gr.Tab("2. Ask Questions"):
            
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="Your Question", 
                        placeholder="e.g., What is the main conclusion of this document?",
                        lines=2
                    )
                    ask_btn = gr.Button("Search Knowledge Base", variant="primary")
                
                with gr.Column(scale=2):
                    answer_out = gr.Textbox(label="AI Answer", lines=8, show_copy_button=True)
                    sources_out = gr.Textbox(label="Sources Used", lines=3)

            ask_btn.click(
                fn=chat_logic,
                inputs=[query_input],
                outputs=[answer_out, sources_out]  # 2 values returned by chat_logic
            )

# 5. EXECUTION
if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

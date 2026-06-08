"""
Milestone 5 — Query interface (Gradio web UI).

Wraps the end-to-end ask() function from query.py in a simple web interface:
type a question, get a grounded answer plus the list of documents the answer
was retrieved from.

Run:  python app.py
Then open http://localhost:7860
"""

import gradio as gr

from query import ask

EXAMPLES = [
    "What are the dining hall hours on weekends?",
    "What do students say about food quality in the dining halls?",
    "What meal plan options are available for the 2025-26 academic year?",
    "What accommodations are available for vegetarian or gluten-free meals?",
]


def handle_query(question):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(no sources)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide — Augustana Dining") as demo:
    gr.Markdown(
        "# 🍽️ The Unofficial Guide to Augustana Dining\n"
        "Ask about dining hours, meal plans, food quality, dietary options, "
        "and campus dining locations. Answers come **only** from the collected "
        "documents, with sources shown."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. What are the weekend dining hours?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()

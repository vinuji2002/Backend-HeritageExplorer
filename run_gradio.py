# run_gradio.py
import gradio as gr
import requests

BASE_URL = "http://localhost:5000"

def chat_fn(message, history, char_id, session_id):
    if not message.strip():
        return history or [], ""
    try:
        r = requests.post(f"{BASE_URL}/user/chat", json={
            "query": message,
            "character_id": char_id,
            "session_id": session_id or "gradio_user"
        }, timeout=60)
        data = r.json()
        answer = data.get("answer", "No response")
        char_name = data.get("character_name", char_id)
        confidence = data.get("confidence", 0)
        bot = f"**{char_name}**\n\n{answer}\n\n_Confidence: {confidence:.0%}_"
    except Exception as e:
        bot = f"Error connecting to Flask: {e}"
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot})
    return history, ""

with gr.Blocks(title="Sri Lanka Historical Chatbot") as demo:
    gr.Markdown("# 🏛️ Sri Lanka VR Historical Chatbot")
    gr.Markdown("Powered by TinyLlama + RAG + RL — 4 Historical Characters")

    chatbot_ui = gr.Chatbot(value=[], height=500, type="messages", label="Conversation")

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a historical question...",
            label="Your Question", scale=4, lines=2
        )
        sid = gr.Textbox(value="gradio_user", label="Session ID", scale=1)

    with gr.Row():
        char_dd = gr.Dropdown(
            choices=[
                ("👑 King Sri Wikrama Rajasinha", "king"),
                ("🙏 Thero (Chief Custodian)", "nilame"),
                ("⚓ Captain Willem van der Berg", "dutch"),
                ("📚 Rathnayake Mudalige Sunil", "citizen"),
            ],
            value="king", label="Character", scale=2
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    gr.Examples(examples=[
        ["Hello, who are you?"],
        ["Tell me about the Sacred Tooth Relic"],
        ["What is the Esala Perahera festival?"],
        ["Describe Galle Fort"],
        ["What was Dutch rule like in Ceylon?"],
    ], inputs=msg)

    send_btn.click(chat_fn, [msg, chatbot_ui, char_dd, sid], [chatbot_ui, msg])
    msg.submit(chat_fn, [msg, chatbot_ui, char_dd, sid], [chatbot_ui, msg])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
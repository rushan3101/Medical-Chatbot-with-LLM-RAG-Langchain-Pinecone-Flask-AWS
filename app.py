from flask import Flask, render_template, request, Response
from main import run_rag
from langchain_core.messages import HumanMessage, AIMessage
import time

app = Flask(__name__)


# In-memory chat history
chat_history = []  


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_input = request.form["msg"]

    # Streaming generator with controlled speed
    def generate():
        response_text = ""
        trimmed_chat_history = chat_history[-8:]

        response_text = run_rag(user_input, trimmed_chat_history)

        # Stream with controlled speed (simulate ChatGPT-like streaming)
        # Yield in smaller batches with small delays
        chunk_size = 10  # Number of characters to yield at once
        delay = 0.02  # 20ms delay between chunks for smooth streaming
        
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield chunk
            time.sleep(delay)

        # Save conversation AFTER full response is yielded
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response_text))

        # Keep only last 4 exchanges (8 messages)
        while len(chat_history) > 8:
            chat_history.pop(0)

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
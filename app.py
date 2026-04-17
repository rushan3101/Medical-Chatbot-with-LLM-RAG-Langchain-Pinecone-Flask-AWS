from flask import Flask, render_template, request, Response
from main import run_rag
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)


# In-memory chat history
chat_history = []  


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    user_input = request.form["msg"]

    # Streaming generator
    def generate():
        response_text = ""
        trimmed_chat_history = chat_history[-8:]

        stream = run_rag(user_input, trimmed_chat_history)

        for chunk in stream:
            token = str(chunk)
            response_text += token
            yield token

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
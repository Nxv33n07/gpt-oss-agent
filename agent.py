from openai import OpenAI
from typing import Optional
import sys

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL  = "http://3.109.63.164:11434/v1"
API_KEY   = "dummy"
MODEL     = "gpt-oss:20b"
SYSTEM_PROMPT = "You are a helpful AI assistant."

# ── Client ───────────────────────────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


class ChatAgent:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.history: list[dict] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str, stream: bool = True) -> str:
        """Send a message and get a response. Maintains conversation history."""
        self.history.append({"role": "user", "content": user_message})

        if stream:
            response_text = self._stream_response()
        else:
            response_text = self._get_response()

        self.history.append({"role": "assistant", "content": response_text})
        return response_text

    def _get_response(self) -> str:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.history,
                timeout=10.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"\n⚠️ [ERROR]: Cannot connect to the AI server at {BASE_URL}. Your team's server seems to be offline or blocked by a firewall. (Details: {e})"

    def _stream_response(self) -> str:
        print("Assistant: ", end="", flush=True)
        full_text = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=self.history,
                stream=True,
                timeout=10.0
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full_text += delta
            print()  # newline after stream ends
            return full_text
        except Exception as e:
            error_msg = f"\n⚠️ [SERVER OFFLINE]: I cannot reach your team's server at {BASE_URL}. Please tell your team the server is down or the firewall is blocking port 11434!"
            print(error_msg)
            return error_msg

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        system = self.history[0] if self.history and self.history[0]["role"] == "system" else None
        self.history = [system] if system else []
        print("🔄 Conversation reset.\n")

    def show_history(self):
        """Print the full conversation history."""
        print("\n── Conversation History ──")
        for msg in self.history:
            role = msg["role"].upper()
            print(f"[{role}]: {msg['content']}\n")
        print("─────────────────────────\n")


def run_cli():
    """Interactive CLI loop."""
    print("╔══════════════════════════════════╗")
    print("║        Chat Inference Agent       ║")
    print("║  Model : gpt-oss:20b              ║")
    print("╚══════════════════════════════════╝")
    print("Commands: 'exit' | 'reset' | 'history'\n")

    agent = ChatAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        elif user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            agent.reset()
        elif user_input.lower() == "history":
            agent.show_history()
        else:
            agent.chat(user_input, stream=True)


# ── Programmatic usage example ───────────────────────────────────────────────
def example_programmatic():
    agent = ChatAgent(system_prompt="You are a concise assistant.")
    
    reply1 = agent.chat("What is the capital of France?", stream=False)
    print(f"Reply 1: {reply1}")

    reply2 = agent.chat("What language do they speak there?", stream=False)
    print(f"Reply 2: {reply2}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_programmatic()
    else:
        run_cli()

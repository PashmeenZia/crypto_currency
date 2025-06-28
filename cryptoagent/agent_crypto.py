import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel
import chainlit as cl
from tools import get_crypto_price  # Make sure this function returns a string or dict

# Load .env variables
load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")
print("gemini API key loaded:", gemini_api_key)

# Create the OpenAI client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set up run configuration
run_config = RunConfig(
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    model_provider=client,
    tracing_disabled=True
)

# Define the agent with tool
CryptoDataAgent = Agent(
    name="CryptoDataAgent",
    instructions="You are a helpful agent that gives real-time cryptocurrency prices using CoinGecko.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[get_crypto_price]  # ✅ use tools instead of main
)

# Chat start event
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome to the CryptoCurrency Chatbot!\nAsk me anything about cryptocurrency."
    ).send()

# Chat message handler
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    try:
        result = Runner.run_sync(CryptoDataAgent, input=history, run_config=run_config)
        final_output = result.final_output or "❌ Gemini didn't return any response."
    except Exception as e:
        final_output = f"❌ Error: {str(e)}"

    await cl.Message(content=final_output).send()
    history.append({"role": "assistant", "content": final_output})
    cl.user_session.set("history", history)

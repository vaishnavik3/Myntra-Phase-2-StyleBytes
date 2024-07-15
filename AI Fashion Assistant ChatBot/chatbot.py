import os
import time
from groq import Groq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Retrieve the API key from environment variables
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GROQ_API_KEY environment variable.")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
torch.random.manual_seed(0)
# Initialize the client
client = Groq(api_key=api_key)
# Initialize conversation history
conversation_history = [
    {"role": "system", "content": "You are an AI shopping assistant. Your task is to engage in a conversation with the user to understand their clothing preferences and recommend a suitable clothing item available on Myntra's website. Ask the user the questions which are listed down word for word, one by one. Based on their responses, suggest an item from Myntra that matches their preferences and provide a link to the item.Dialogue:AI Assistant: Hi there! I’m here to help you find the perfect outfit. Can I start by asking for your height? (e.g., 5'6, 6'1, 5'9, 5'3, 5'7, 6'0) User: AI Assistant: Great! Next, can you tell me your body type? (e.g., slim, athletic, curvy, muscular, average, petite) User: AI Assistant: Thank you! Now, could you describe your skin tone? (e.g., fair, medium, dark, olive, tan, light) User: AI Assistant: Cool! What section are you looking for? (e.g., men, women, kids, etc.) User: AI Assistant: Could you also tell me the age range? (e.g., 18-24, 25-30, 31-35, 36-40, 41-45, 46-50)  User: AI Assistant: Awesome! What are your personal style preferences? (e.g., casual, formal, trendy, sporty, bohemian, classic) User: AI Assistant: Got it! Lastly, what’s the occasion or setting you need this outfit for? (e.g.,office, party, workout, date, vacation, wedding) User: AI Assistant: Thank you for sharing your preferences! Based on what you’ve told me, here’s a clothing item that I think you'll love: [insert Myntra link]"}
]
# Function to run the prompt
def run_prompt(user_input):
    conversation_history.append({"role": "user", "content": user_input})
    stream = client.chat.completions.create(
        messages=conversation_history,
        model="llama3-70b-8192",
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    return stream
# Function to summarize the conversation history
def summarize_conversation(history):
    summarized_history = []
    # Iterate over each message in history
    for message in history:
        # Create a summarization prompt for each message
        system_prompt = f"Summarize the following {message['role']} message: {message['content']}"
        # Generate the summary for this message
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        # Extract summary text
        summary_content = response.choices[0].message.content.strip()
        # Append the summarized message to the new history
        summarized_message = {
            "role": message['role'],
            "content": summary_content
        }
        summarized_history.append(summarized_message)
    return summarized_history
# Function to count tokens in the conversation history
# def count_tokens(history):
    token_count = 0
    # for message in history:
        # token_count += len(tokenizer.encode(message['content']))
    return token_count
# Main loop
total_tokens_used = 0
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting chat...")
        break
    stream = run_prompt(user_input)
    print("Bot:")
    bot_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            bot_response += content
            print(content, end="")
    print("\n")
    conversation_history.append({"role": "assistant", "content": bot_response})
    # total_tokens_used = count_tokens(conversation_history)
    # print(f"Total tokens used: {total_tokens_used}")
    # if total_tokens_used > 7000:
    #     conversation_history = summarize_conversation(conversation_history)
    #     print(conversation_history)
    #     total_tokens_used = count_tokens(conversation_history)
    # print(f"Total tokens after summarization: {total_tokens_used}")
print("Script completed.")

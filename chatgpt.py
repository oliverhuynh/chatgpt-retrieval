import os
import sys
import json
import argparse
import logging
from pathlib import Path
import openai
from dotenv import load_dotenv

openai.proxy = {
            "http": "http://127.0.0.1:7890",
            "https": "http://127.0.0.1:7890"
        }

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

# Load .env into the environment (both script directory and CWD)
env_loaded = []
script_dir = Path(__file__).resolve().parent
for candidate in (script_dir / ".env", Path.cwd() / ".env"):
    if candidate.is_file():
        load_dotenv(dotenv_path=candidate, override=False)
        env_loaded.append(str(candidate))

# Prefer environment-provided key; fail if missing
active_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
if not active_key:
    raise RuntimeError("OPENAI_KEY not set in environment or .env")
os.environ["OPENAI_API_KEY"] = active_key
masked_key = (
    f"{active_key[:4]}...{active_key[-4:]}"
    if active_key and len(active_key) > 8 else "<unset>"
)

# Allow overriding the OpenAI API host (useful for self-hosted proxies).
openai_target = os.getenv("OPENAI_TARGET", "https://api.openai.com")
base_url = f"{openai_target.rstrip('/')}/v1"
os.environ["OPENAI_BASE_URL"] = base_url
os.environ["OPENAI_API_BASE"] = base_url  # backward compatibility
openai.base_url = base_url  # ensure global default for any client created internally
model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

from oliver_framework.utils.logging import getlogger
logger=getlogger("chatgpt")
logging.getLogger().setLevel(logging.WARNING)
for noisy_logger in ("chromadb", "langchain", "langchain_core", "langchain_community", "openai", "httpx", "numexpr"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
if env_loaded:
    logger.debug(f"Loaded .env from: {env_loaded}")
logger.debug(f"Using OpenAI base_url: {base_url}")
logger.debug(f"Using OpenAI key: {masked_key}")
logger.debug(f"Using OpenAI model: {model}")

# Variables
temperature=0.2
timeout=30

from openai import OpenAI
# client = OpenAI()
import httpx
client = OpenAI(
    base_url=base_url,
    timeout=httpx.Timeout(timeout, read=timeout / 2, write=timeout / 2, connect=10)
)

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', '-p', help='Prompt for the query')
parser.add_argument('--local', '-l', help='Use local data only')
parser.add_argument('--data_dir', default='data/', help='Directory for data (default: data/)')
parser.add_argument('--continue', '-c', action='store_true', dest='resume', help='Resume previous conversation')

# .chroma is not usable
# parser.add_argument('--persist_dir', default='.chroma', help='Directory for chroma dir (default: .chroma)')
parser.add_argument('--persist_dir', default='~/.chatgpt/chroma', help='Directory for chroma dir (default: ~/.chatgpt/chroma)')

args, unknown_args = parser.parse_known_args()
is_prompt = args.prompt if args.prompt else None
is_local = args.local if args.local else None
data_dir = args.data_dir
if not os.path.isabs(data_dir) and not os.path.isdir(data_dir):
    # Fall back to script-local data/ only if CWD data_dir is missing.
    script_root = Path(__file__).resolve().parent
    candidate = script_root / data_dir
    if candidate.is_dir():
        data_dir = str(candidate)
persist_dir = args.persist_dir
persist_dir = os.path.expanduser(persist_dir)
# Resolve persist_dir relative to the script directory and ensure it exists.
if not os.path.isabs(persist_dir):
    script_root = Path(__file__).resolve().parent
    persist_dir = str(script_root / persist_dir)
os.makedirs(persist_dir, exist_ok=True)
resume_chat = bool(args.resume)

config_dir = os.path.expanduser("~/.chatgpt")
os.makedirs(config_dir, exist_ok=True)
history_path = os.path.join(config_dir, "chat_history.json")
def load_chat_history():
    if not resume_chat or not os.path.isfile(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        # Convert to list of tuples expected by langchain history.
        return [(item.get("question", ""), item.get("answer", "")) for item in items if item]
    except Exception as exc:
        logger.warning(f"Failed to load chat history: {exc}")
        return []

def save_chat_history(history):
    try:
        items = [{"question": q, "answer": a} for q, a in history]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to save chat history: {exc}")

query = ' '.join(unknown_args)
# Support piped or multi-line stdin when no positional query is provided.
if not query and not sys.stdin.isatty():
    stdin_payload = sys.stdin.read()
    if stdin_payload:
        query = stdin_payload.strip()

chain = False
if os.listdir(data_dir):
  if PERSIST and os.path.exists("persist"):
    logger.debug("Reusing index...\n")
    vectorstore = Chroma(
      persist_directory=persist_dir,
      embedding_function=OpenAIEmbeddings(base_url=base_url)
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
  else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader(data_dir)
    if PERSIST:
      index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(base_url=base_url),
        vectorstore_kwargs={"persist_directory":persist_dir}
      ).from_loaders([loader])
    else:
      index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings(base_url=base_url)
      ).from_loaders([loader])

  # Debug: show where embeddings client will point
  try:
    dbg_client = OpenAIEmbeddings(base_url=base_url).client
    logger.debug(f"Embeddings client base_url: {dbg_client._client.base_url}")
  except Exception as e:
    logger.debug(f"Embeddings client base_url check failed: {e}")

  chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model=model, base_url=base_url),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
  )

# Modify this part to include a direct call to the GPT model
def is_uncertain(answer):
    # Define phrases that indicate uncertainty
    uncertain_phrases = ["I'm sorry,", "i don't know", "not sure", "unsure", "maybe", "I don't have", "don't have that information", "don't have enough"]
    # Check if the answer contains any of the uncertain phrases
    return any(phrase in answer.lower() for phrase in uncertain_phrases)

def get_openai_response(prompt, model=model):  # Adjust model as needed
    # Check if prompt is a string, if not, use the input object
    if not isinstance(prompt, str):
        messages = prompt["messages"]
        model = prompt.get("model", model)
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    # Assuming the response format aligns with the chat model's response structure
    return response.choices[0].message.content.strip()

def query_both_sources(query, chat_history):
    # Query the custom data via the retriever first to check for an uncertain response
    # logger.debug("Going to custom data response")
    custom_data_response = False
    query_chat_history = []

    try:
        # Try to parse query as JSON
        query_json = json.loads(query)

        # If successful, assume query is a JSON string representing the chained prompt
        query_chat_history = [(turn["question"], turn["answer"]) for turn in query_json["chat_history"]]
        query_json["chat_history"] = query_chat_history
    except json.JSONDecodeError:
        # If parsing as JSON fails, assume query is a normal question string
        query_json = {"question": query, "chat_history": chat_history, "timeout": 10}

    # Retrieve chain
    if chain: 
        logger.debug("Ask based on local data first")
        logger.debug(f"Query: {query_json}")
        custom_data_response = chain(query_json)

    # If the custom data's answer is uncertain, then query the GPT model
    if not is_local and not custom_data_response or ('answer' in custom_data_response and is_uncertain(custom_data_response['answer'])):
        logger.debug("Custom data response is uncertain, going to OpenAI response")
        messages = []

        # @TODO: Init role system
        # query_chat_hisory is array of tuples
        for item in query_chat_history:
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]}) 

        # Convert query_json to the format expected by get_openai_response
        messages.append({"role": "user", "content": query_json["question"]})

        gpt_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        logger.debug("Ask based on remote data")
        logger.debug(f"Args: {gpt_request}")
        direct_response = get_openai_response(gpt_request)
        return direct_response
    else:
        # If the custom data's answer is confident enough, use it directly
        logger.debug("Using custom data response.")
        return custom_data_response.get('answer', 'No answer found.')

chat_history = load_chat_history()
while True:
    if not query:
        if not is_prompt:
            sys.exit()
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    # Update to use the new function that queries both sources
    result_answer = query_both_sources(query, chat_history)
    print(result_answer)

    chat_history.append((query, result_answer))
    save_chat_history(chat_history)
    query = None

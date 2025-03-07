import os
import sys
import json
import argparse
import openai
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

# Check if constants.py exists in the current working directory
if not os.path.exists("constants.py"):
    raise FileNotFoundError("constants.py not found in the current directory")

import importlib.util
# Load constants.py from the current working directory
spec = importlib.util.spec_from_file_location("constants", "constants.py")
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

os.environ["OPENAI_API_KEY"] = constants.APIKEY

from oliver_framework.utils.logging import getlogger
logger=getlogger("chatgpt")

# Variables
model="gpt-3.5-turbo"
temperature=0.2
timeout=30

from openai import OpenAI
# client = OpenAI()
import httpx
client = OpenAI(timeout=httpx.Timeout(timeout, read=timeout / 2, write=timeout / 2, connect=10))

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--prompt', '-p', help='Prompt for the query')
parser.add_argument('--local', '-l', help='Use local data only')
parser.add_argument('--data_dir', default='data/', help='Directory for data (default: data/)')

# .chroma is not usable
# parser.add_argument('--persist_dir', default='.chroma', help='Directory for chroma dir (default: .chroma)')
parser.add_argument('--persist_dir', default='tmp/x', help='Directory for chroma dir (default: .chroma)')

args, unknown_args = parser.parse_known_args()
is_prompt = args.prompt if args.prompt else None
is_local = args.local if args.local else None
data_dir = args.data_dir
persist_dir = args.persist_dir

query = ' '.join(unknown_args)

chain = False
if os.listdir(data_dir):
  if PERSIST and os.path.exists("persist"):
    logger.debug("Reusing index...\n")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
  else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    loader = DirectoryLoader(data_dir)
    if PERSIST:
      index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":persist_dir}).from_loaders([loader])
    else:
      index = VectorstoreIndexCreator().from_loaders([loader])

  chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
  )

# Modify this part to include a direct call to the GPT model
def is_uncertain(answer):
    # Define phrases that indicate uncertainty
    uncertain_phrases = ["I'm sorry,", "i don't know", "not sure", "unsure", "maybe", "I don't have", "don't have that information", "don't have enough"]
    # Check if the answer contains any of the uncertain phrases
    return any(phrase in answer.lower() for phrase in uncertain_phrases)

def get_openai_response(prompt, model="gpt-3.5-turbo"):  # Adjust model as needed
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

chat_history = []
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
    query = None

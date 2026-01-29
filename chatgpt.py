import os
import sys
import json
import argparse
import base64
import mimetypes
import time
import logging
from pathlib import Path
from typing import Optional

# Disable telemetry early to avoid noisy runtime errors in containerized runs.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")

# Hard-disable PostHog telemetry to avoid runtime errors from mismatched capture signatures.
try:
    import posthog  # type: ignore

    def _noop(*_args, **_kwargs):
        return None

    posthog.capture = _noop
    posthog.flush = _noop
    if hasattr(posthog, "disable"):
        posthog.disable()
except Exception:
    pass

import openai
from dotenv import load_dotenv

openai.proxy = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
try:
    from langchain_chroma import Chroma
except Exception:
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
model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

from oliver_framework.utils.logging import getlogger
logger = getlogger("chatgpt")
env_log_level = os.environ.get("LOGGINGLEVEL", "").upper()
resolved_level = getattr(logging, env_log_level, None)
if not isinstance(resolved_level, int):
    resolved_level = logging.WARNING
logging.getLogger().setLevel(resolved_level)
noisy_level = logging.DEBUG if resolved_level <= logging.DEBUG else logging.ERROR
for noisy_logger in ("chromadb", "langchain", "langchain_core", "langchain_community", "openai", "httpx", "numexpr"):
    logging.getLogger(noisy_logger).setLevel(noisy_level)
if env_loaded:
    logger.debug(f"Loaded .env from: {env_loaded}")
logger.debug(f"Using OpenAI base_url: {base_url}")
logger.debug(f"Using OpenAI key: {masked_key}")
logger.debug(f"Using OpenAI model: {model}")

# Variables
temperature = 0.2
timeout = 30

from openai import OpenAI
import httpx
client = OpenAI(
    base_url=base_url,
    timeout=httpx.Timeout(timeout, read=timeout / 2, write=timeout / 2, connect=10),
)

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True


def is_uncertain(answer: str) -> bool:
    uncertain_phrases = [
        "i'm sorry,",
        "i don't know",
        "not sure",
        "unsure",
        "maybe",
        "i don't have",
        "don't have that information",
        "don't have enough",
        "don't have information",
        "no information",
    ]
    return any(phrase in answer.lower() for phrase in uncertain_phrases)


def get_openai_response(prompt, model=model):
    if not isinstance(prompt, str):
        messages = prompt["messages"]
        model = prompt.get("model", model)
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def _prepare_attachments(file_paths: list[str]):
    if not file_paths:
        return []
    items = []
    use_imgbb = bool(os.environ.get("IMGBB_API_KEY")) and openai_target.rstrip("/") != "https://api.openai.com"
    imgbb_expiration = 300
    for raw_path in file_paths:
        path = os.path.expanduser(raw_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Attachment not found: {raw_path}")
        mime, _ = mimetypes.guess_type(path)
        is_image = bool(mime and mime.startswith("image/"))
        if is_image and use_imgbb:
            with open(path, "rb") as f:
                payload = base64.b64encode(f.read()).decode("ascii")
            url = _upload_image_imgbb(payload, imgbb_expiration)
            logger.debug(f"ImgBB upload URL: {url}")
            items.append({"type": "input_image", "image_url": url, "detail": "auto"})
        elif is_image:
            with open(path, "rb") as f:
                upload = client.files.create(file=f, purpose="vision")
            items.append({"type": "input_image", "file_id": upload.id, "detail": "auto"})
        else:
            with open(path, "rb") as f:
                payload = base64.b64encode(f.read()).decode("ascii")
            items.append(
                {
                    "type": "input_file",
                    "file_data": payload,
                    "filename": os.path.basename(path),
                }
            )
    return items


def _extract_response_text(response):
    if hasattr(response, "output_text"):
        return response.output_text
    output = None
    if hasattr(response, "output"):
        output = response.output
    elif isinstance(response, dict):
        output = response.get("output")
        if response.get("output_text"):
            return response["output_text"]
    if not output:
        return str(response)
    texts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, dict):
                    text = text.get("value") or text.get("text")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip() or str(response)


def _upload_image_imgbb(image_base64: str, expiration_seconds: int) -> str:
    api_key = os.environ.get("IMGBB_API_KEY")
    if not api_key:
        raise RuntimeError("IMGBB_API_KEY is required for ImgBB upload.")
    last_exc = None
    for attempt in range(3):
        try:
            resp = httpx.post(
                "https://api.imgbb.com/1/upload",
                params={"key": api_key, "expiration": expiration_seconds},
                data={"image": image_base64},
                timeout=httpx.Timeout(timeout, read=timeout, connect=10),
            )
            resp.raise_for_status()
            payload = resp.json()
            url = payload.get("data", {}).get("url")
            if not url:
                raise RuntimeError("ImgBB response missing image URL.")
            return url
        except httpx.ReadTimeout as exc:
            last_exc = exc
            logger.warning(f"ImgBB upload timed out (attempt {attempt + 1}/3).")
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            logger.warning(f"ImgBB upload failed (attempt {attempt + 1}/3): {exc}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ImgBB upload failed after retries: {last_exc}")


def get_openai_response_with_files(messages, attachments, model=model, temperature=temperature):
    content_items = [{"type": "input_text", "text": messages[-1]["content"]}]
    content_items.extend(attachments)

    prepared_messages = []
    for msg in messages[:-1]:
        prepared_messages.append({"role": msg["role"], "content": msg["content"]})
    prepared_messages.append({"role": "user", "content": content_items})

    payload = {"model": model, "input": prepared_messages, "temperature": temperature}
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        payload["include"] = ["message.input_image.image_url"]
        logger.debug(f"Responses payload: {json.dumps(payload, ensure_ascii=False)}")
        summary = []
        for item in attachments:
            entry = {"type": item.get("type")}
            if item.get("type") == "input_image":
                entry["image_url"] = item.get("image_url")
                entry["file_id"] = item.get("file_id")
                entry["detail"] = item.get("detail")
            elif item.get("type") == "input_file":
                entry["filename"] = item.get("filename")
                entry["file_id"] = item.get("file_id")
            summary.append(entry)
        logger.debug(f"Attachment summary: {summary}")
    try:
        response = client.responses.create(**payload)
        return _extract_response_text(response).strip()
    except AttributeError:
        pass
    try:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for responses API.")
        resp = httpx.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return _extract_response_text(resp.json()).strip()
    except Exception as exc:
        raise RuntimeError(f"Responses API call failed: {exc}") from exc


def _resolve_data_dir(data_dir: str) -> str:
    if not os.path.isabs(data_dir) and not os.path.isdir(data_dir):
        script_root = Path(__file__).resolve().parent
        candidate = script_root / data_dir
        if candidate.is_dir():
            data_dir = str(candidate)
    return data_dir


def _resolve_persist_dir(persist_dir: str) -> str:
    persist_dir = os.path.expanduser(persist_dir)
    if not os.path.isabs(persist_dir):
        script_root = Path(__file__).resolve().parent
        persist_dir = str(script_root / persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def _build_chain(data_dir: str, persist_dir: str, system_prompt: Optional[str] = None):
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        return False
    if PERSIST and os.path.exists(persist_dir):
        logger.debug("Reusing index...\n")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(base_url=base_url),
        )
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader(data_dir)
        if PERSIST:
            index = VectorstoreIndexCreator(
                embedding=OpenAIEmbeddings(base_url=base_url),
                vectorstore_cls=Chroma,
                vectorstore_kwargs={"persist_directory": persist_dir},
            ).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator(
                embedding=OpenAIEmbeddings(base_url=base_url),
                vectorstore_cls=Chroma,
            ).from_loaders([loader])

    try:
        dbg_client = OpenAIEmbeddings(base_url=base_url).client
        logger.debug(f"Embeddings client base_url: {dbg_client._client.base_url}")
    except Exception as exc:
        logger.debug(f"Embeddings client base_url check failed: {exc}")

    combine_docs_chain_kwargs = {}
    if system_prompt:
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                f"{system_prompt}\n\n"
                "Use the following context to answer the user's question. "
                "If the context is empty or irrelevant, still respond according to your role.\n\n"
                "Context:\n{context}\n\n"
                "User: {question}\n"
                "Assistant:"
            ),
        )
        combine_docs_chain_kwargs["prompt"] = qa_prompt

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=model, base_url=base_url),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
    )


def query_both_sources(
    query,
    chat_history,
    chain,
    is_local=False,
    system_prompt: Optional[str] = None,
    attachments: Optional[list[str]] = None,
):
    custom_data_response = False
    query_chat_history = []

    try:
        query_json = json.loads(query)
        query_chat_history = [(turn["question"], turn["answer"]) for turn in query_json["chat_history"]]
        query_json["chat_history"] = query_chat_history
    except json.JSONDecodeError:
        query_json = {"question": query, "chat_history": chat_history, "timeout": 10}

    if chain and not attachments:
        logger.debug("Ask based on local data first")
        logger.debug(f"Query: {query_json}")
        custom_data_response = chain.invoke(query_json)

    if attachments:
        logger.debug("Attachments provided; skipping local retrieval.")
    if attachments or (not is_local and not custom_data_response) or (
        "answer" in custom_data_response and is_uncertain(custom_data_response["answer"])
    ):
        logger.debug("Custom data response is uncertain, going to OpenAI response")
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in query_chat_history:
            messages.append({"role": "user", "content": item[0]})
            messages.append({"role": "assistant", "content": item[1]})

        messages.append({"role": "user", "content": query_json["question"]})

        gpt_request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        logger.debug("Ask based on remote data")
        logger.debug(f"Args: {gpt_request}")
        if attachments:
            file_items = _prepare_attachments(attachments)
            direct_response = get_openai_response_with_files(
                messages,
                file_items,
                model=model,
                temperature=temperature,
            )
        else:
            direct_response = get_openai_response(gpt_request)
        return direct_response
    else:
        logger.debug("Using custom data response.")
        return custom_data_response.get("answer", "No answer found.")


def ask_question(
    question: str,
    data_dir: str = "data/",
    is_local: bool = False,
    persist_dir: str = "~/.chatgpt/chroma",
    chat_history: Optional[list[tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    attachments: Optional[list[str]] = None,
) -> str:
    data_dir = _resolve_data_dir(data_dir)
    persist_dir = _resolve_persist_dir(persist_dir)
    chain = _build_chain(data_dir, persist_dir, system_prompt=system_prompt)
    history = chat_history or []
    return query_both_sources(
        question,
        history,
        chain,
        is_local=is_local,
        system_prompt=system_prompt,
        attachments=attachments,
    )


def _load_chat_history(resume_chat: bool, history_path: str):
    if not resume_chat or not os.path.isfile(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [(item.get("question", ""), item.get("answer", "")) for item in items if item]
    except Exception as exc:
        logger.warning(f"Failed to load chat history: {exc}")
        return []


def _save_chat_history(history, history_path: str):
    try:
        items = [{"question": q, "answer": a} for q, a in history]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to save chat history: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", help="Prompt for the query")
    parser.add_argument("--local", "-l", help="Use local data only")
    parser.add_argument("--data_dir", default="data/", help="Directory for data (default: data/)")
    parser.add_argument("--continue", "-c", action="store_true", dest="resume", help="Resume previous conversation")
    parser.add_argument("--persist_dir", default="~/.chatgpt/chroma", help="Directory for chroma dir")
    parser.add_argument("--file", "-f", action="append", default=[], help="Attach file (repeatable)")

    args, unknown_args = parser.parse_known_args()
    is_prompt = args.prompt if args.prompt else None
    is_local = bool(args.local) if args.local is not None else False
    data_dir = args.data_dir
    persist_dir = args.persist_dir
    resume_chat = bool(args.resume)
    attachments = args.file or []

    query = " ".join(unknown_args)
    if not query and not sys.stdin.isatty():
        stdin_payload = sys.stdin.read()
        if stdin_payload:
            query = stdin_payload.strip()

    config_dir = os.path.expanduser("~/.chatgpt")
    os.makedirs(config_dir, exist_ok=True)
    history_path = os.path.join(config_dir, "chat_history.json")
    chat_history = _load_chat_history(resume_chat, history_path)

    while True:
        if not query:
            if not is_prompt:
                sys.exit()
            query = input("Prompt: ")
        if query in ["quit", "q", "exit"]:
            sys.exit()

        result_answer = ask_question(
            query,
            data_dir=data_dir,
            is_local=is_local,
            persist_dir=persist_dir,
            chat_history=chat_history if resume_chat else None,
            attachments=attachments,
        )
        print(result_answer)

        chat_history.append((query, result_answer))
        _save_chat_history(chat_history, history_path)
        query = None


if __name__ == "__main__":
    main()

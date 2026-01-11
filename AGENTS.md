# Repository Guidelines

## Project Structure & Module Organization
- `chatgpt.py` is the main Python entry point for retrieval + OpenAI calls.
- `chatgpt` is a wrapper shell script that runs the project-local Python.
- `data/` holds local documents used for retrieval (example: `data/data.txt`, `data/cat.pdf`).
- `tests/` contains a curl-based smoke test (`tests/test_curl_openai.sh`).
- `tmp/` stores vectorstore persistence (`tmp/x` by default).
- `constants.py.default` is a template for local configuration.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies.
- `python chatgpt.py "your question"` runs the retriever/LLM flow against `data/`.
- `python chatgpt.py --data_dir data/ "question"` overrides the data directory.
- `bash tests/test_curl_openai.sh` smoke-tests the configured OpenAI endpoint.
- `bash setup.sh` links `chatgpt` into `~/.local/bin/` for CLI use.

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 style, keep imports grouped (stdlib, third-party, local).
- Bash: keep `set -euo pipefail` for scripts; use uppercase for environment vars.
- Names: files and directories use lowercase with underscores if needed (e.g., `data_dir`).

## Testing Guidelines
- There is no unit test suite yet; the primary check is the curl smoke test in `tests/`.
- Run `bash tests/test_curl_openai.sh` to verify auth and endpoint wiring.
- If you add tests, keep scripts in `tests/` and document them here.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative summaries (e.g., "Update httpx version...").
- Keep commits focused; prefer one logical change per commit.
- PRs should include: what changed, how to test, and any env vars required.

## Security & Configuration Tips
- Set `OPENAI_KEY` (or `OPENAI_API_KEY`) via `.env` or environment variables.
- Optional: `OPENAI_TARGET` to route to a proxy; `OPENAI_MODEL` to override model.
- Do not commit secrets; keep `.env` local.

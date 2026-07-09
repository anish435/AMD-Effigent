# amd-hackathon-agent Technical Report

Generated from a file-by-file static audit of the cloned repository at `C:\Users\anish\Downloads\amd-hackathon-agent-main\amd-hackathon-agent-main`.

## 1. Architecture Overview

### Directory and Module Map

| Path | Role |
|---|---|
| `.dockerignore` | Excludes `.env`, bytecode, local input/output, and `.venv` from Docker context (`.dockerignore:1-6`). |
| `.env.example` | Documents Fireworks, local model, router, cache, compression, and log env vars (`.env.example:5-32`). |
| `.github/workflows/docker-build.yml` | Builds and pushes image to GHCR on `main` pushes and manual dispatch (`docker-build.yml:3-45`). |
| `.gitignore` | Ignores secrets, bytecode, build artifacts, local I/O, `.venv`, and `.agents` (`.gitignore:1-11`). |
| `assets/banner.png` | README banner image; binary PNG, 735,049 bytes. |
| `Dockerfile` | Multi-stage Python 3.11 image; installs requirements, copies app, creates `/input` and `/output`, runs `python run.py` (`Dockerfile:27-81`). |
| `docker-compose.yml` | Local compose service, but its command is mismatched with `run.py`; see gaps (`docker-compose.yml:9-23`). |
| `main.py` | Local CLI entrypoint: reads `--tasks` or stdin, emits detailed `AgentResponse` JSON and tracker report (`main.py:65-87`, `main.py:138-201`). |
| `run.py` | Harness/container entrypoint: reads `/input/tasks.json`, writes `/output/results.json`, returns harness-shaped output (`run.py:32-35`, `run.py:48-169`). |
| `requirements.txt` | Runtime deps: `openai`, `transformers`, `torch`, `pydantic`, `python-dotenv`, `huggingface-hub`, `accelerate` (`requirements.txt:1-7`). |
| `README.md` | Project overview and claimed optimizations; contains one inaccurate "semantic caching" claim (`README.md:43-45`, `README.md:123-124`). |
| `agent/__init__.py` | Package marker only (`agent/__init__.py:1`). |
| `agent/budget.py` | Category detector and dynamic `max_tokens` budget calculator (`agent/budget.py:23-75`, `agent/budget.py:82-121`). |
| `agent/cache.py` | In-memory normalized exact-match response cache (`agent/cache.py:48-125`). |
| `agent/compressor.py` | Regex prompt compressor for filler phrases and whitespace (`agent/compressor.py:20-100`). |
| `agent/config.py` | Env config dataclasses, validation, and `ALLOWED_MODELS` selection (`agent/config.py:23-174`). |
| `agent/executor.py` | Rule-based arithmetic, local HF model executor, remote Fireworks executor, output verifier, and hybrid orchestration (`agent/executor.py:47-618`). |
| `agent/models.py` | Pydantic schemas/enums for tasks, routing, token usage, execution result, and response (`agent/models.py:21-130`). |
| `agent/router.py` | Zero-cost heuristic router using length, keywords, structure, and sentence signals (`agent/router.py:38-267`). |
| `agent/tracker.py` | Local CLI/eval usage tracker and report printer (`agent/tracker.py:50-250`). |
| `agent/__pycache__/*.pyc` | Generated Python 3.13 bytecode for several modules; not source, ignored by Docker/git patterns. |
| `eval/__init__.py` | Eval package marker only (`eval/__init__.py:1`). |
| `eval/evaluate.py` | Local evaluation harness with simple expected-answer matching (`eval/evaluate.py:41-188`). |
| `eval/test_tasks.json` | 20 sample tasks spanning factual, math, sentiment, summarization, NER, debug, logic, code (`eval/test_tasks.json:1-82`). |

### Data Flow: `/input/tasks.json` to `/output/results.json`

Harness mode is implemented in `run.py`, not `main.py`.

1. Resolve paths and runtime guard:

```python
INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
MAX_RUNTIME_SECONDS = 570
```

Evidence: `run.py:32-35`.

2. Load config and resolve remote model:

```python
from agent.config import load_config, get_resolved_model
config = load_config()
resolved_model = get_resolved_model(config)
```

Evidence: `run.py:56-65`.

3. Read JSON tasks and normalize `task_id`/`id` into `Task(id=..., prompt=...)`:

```python
raw = json.loads(input_path.read_text(encoding="utf-8"))
for item in raw:
    task_id = item.get("task_id") or item.get("id", "")
    prompt = item.get("prompt", "")
    if task_id and prompt:
        tasks.append(Task(id=str(task_id), prompt=prompt))
```

Evidence: `run.py:83-92`.

4. For each task, route and execute:

```python
decision = router.route(task)
result = executor.execute(task, decision)
answer = result.output
```

Evidence: `run.py:123-126`.

5. Write harness-shaped output:

```python
results.append({
    "task_id": task.id,
    "answer": answer,
})
output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
```

Evidence: `run.py:140-152`.

### Harness Contract and Env Vars

The harness contract is explicitly documented:

```python
# Input:  /input/tasks.json  -> [{"task_id": "t1", "prompt": "..."}, ...]
# Output: /output/results.json -> [{"task_id": "t1", "answer": "..."}, ...]
# Env vars FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
```

Evidence: `run.py:10-16`.

Env var usage:

| Env var | Code use |
|---|---|
| `INPUT_PATH` | Overrides `/input/tasks.json` (`run.py:32`). |
| `OUTPUT_PATH` | Overrides `/output/results.json` (`run.py:33`). |
| `LOG_LEVEL` | Logging level in `run.py` and `AppConfig` (`run.py:39-45`, `agent/config.py:112-114`). |
| `FIREWORKS_API_KEY` | OpenAI-compatible Fireworks key (`agent/config.py:26-28`, `agent/executor.py:118-121`). |
| `FIREWORKS_BASE_URL` | OpenAI-compatible base URL (`agent/config.py:29-33`, `agent/executor.py:118-121`). |
| `FIREWORKS_MODEL` | Fallback remote model when no `ALLOWED_MODELS` (`agent/config.py:34-39`, `agent/config.py:143-145`). |
| `ALLOWED_MODELS` | Comma-split allowed model IDs (`agent/config.py:40-46`). |
| `FIREWORKS_TEMPERATURE` | Remote temperature (`agent/config.py:47-49`, `agent/executor.py:155`). |
| `FIREWORKS_MAX_TOKENS` | Default remote max tokens if no override (`agent/config.py:50-52`, `agent/executor.py:136`). |
| `LOCAL_MODEL_NAME`, `LOCAL_MODEL_DEVICE`, `LOCAL_MODEL_MAX_NEW_TOKENS`, `LOCAL_MODEL_TEMPERATURE`, `LOCAL_MODEL_DTYPE` | Local HF model settings (`agent/config.py:55-78`). |
| `ROUTER_COMPLEXITY_THRESHOLD` | Score cutoff for remote vs local (`agent/config.py:81-95`, `agent/router.py:131-135`). |
| `ROUTER_CONFIDENCE_FALLBACK_THRESHOLD` | Local confidence fallback threshold (`agent/config.py:90-95`, `agent/executor.py:588-600`). |
| `CACHE_ENABLED` | Enables in-memory cache (`agent/config.py:104-107`, `agent/executor.py:527`). |
| `COMPRESSION_ENABLED` | Enables remote prompt compression (`agent/config.py:108-111`, `agent/executor.py:530`, `agent/executor.py:560-564`). |

### Docker Setup

Docker build:

```dockerfile
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
FROM python:3.11-slim AS runtime
COPY --from=builder /install /usr/local
COPY agent/ ./agent/
COPY eval/ ./eval/
COPY run.py .
COPY main.py .
```

Evidence: `Dockerfile:27-54`.

Runtime defaults and entrypoint:

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOCAL_MODEL_NAME=google/gemma-2-2b-it \
    ROUTER_COMPLEXITY_THRESHOLD=0.6 \
    CACHE_ENABLED=true \
    COMPRESSION_ENABLED=true
ENTRYPOINT ["python", "run.py"]
```

Evidence: `Dockerfile:67-81`.

Invocation expected by comments:

```dockerfile
docker run -v ./input:/input -v ./output:/output \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  amd-routing-agent
```

Evidence: `Dockerfile:12-17`, `README.md:98-105`.

Compose issue: compose mounts eval data to `/app/eval` and output to `/app/output`, then passes CLI args:

```yaml
volumes:
  - ./eval:/app/eval:ro
  - ./output:/app/output
command: ["--tasks", "/app/eval/test_tasks.json", "--output", "/app/output/results.json"]
```

Evidence: `docker-compose.yml:18-23`. Because the image entrypoint is `python run.py` (`Dockerfile:81`) and `run.py` does not parse `--tasks`/`--output`, this compose command will not affect `INPUT_PATH`/`OUTPUT_PATH`; `run.py` still looks for `/input/tasks.json` (`run.py:32-33`, `run.py:75-83`).

## 2. Routing Logic

### Complexity Determination

Routing is pure rule-based heuristic scoring; no LLM/classifier is used. The module says:

```python
Uses a zero-cost heuristic classifier ... No LLM inference is used for routing itself.
```

Evidence: `agent/router.py:4-6`.

Signals:

```python
_WEIGHTS = {
    "length": 0.10,
    "complex_keywords": 0.30,
    "simple_keywords": 0.15,
    "structured_output": 0.10,
    "multi_part": 0.10,
    "question_depth": 0.05,
    "sentence_count": 0.20,
}
```

Evidence: `agent/router.py:112-121`.

The signal calculator returns length, keyword, output-structure, multi-part, question-depth, sentence-count, and informational word count:

```python
return {
    "length": self._length_signal(word_count),
    "complex_keywords": self._keyword_signal(prompt_lower, _COMPLEX_KEYWORDS),
    "simple_keywords": self._keyword_signal(prompt_lower, _SIMPLE_KEYWORDS),
    "structured_output": self._structured_output_signal(prompt),
    "multi_part": self._multi_part_signal(prompt),
    "question_depth": self._question_depth_signal(prompt),
    "sentence_count": self._sentence_count_signal(prompt),
    "word_count": word_count,
}
```

Evidence: `agent/router.py:155-169`.

### Exact Decision Tree

Route selection:

```python
score = self._aggregate(signals)
difficulty = self._score_to_difficulty(score)
chosen_route = (
    Route.REMOTE
    if score >= self._config.complexity_threshold
    else Route.LOCAL
)
```

Evidence: `agent/router.py:128-135`.

Difficulty bins:

```python
if score < 0.35:
    return TaskDifficulty.SIMPLE
if score < 0.65:
    return TaskDifficulty.MODERATE
return TaskDifficulty.COMPLEX
```

Evidence: `agent/router.py:243-249`.

Aggregation:

```python
if name == "simple_keywords":
    score += weight * (1.0 - value)
else:
    score += weight * value
return max(0.0, min(1.0, score))
```

Evidence: `agent/router.py:231-241`.

Important consequence: even a prompt with no "simple" keywords receives the full positive `0.15` contribution from `simple_keywords` because it adds `0.15 * (1.0 - 0.0)`.

Length branch:

```python
if word_count <= 10: return 0.1
if word_count <= 30: return 0.3
if word_count <= 80: return 0.5
if word_count <= 150: return 0.7
return 0.9
```

Evidence: `agent/router.py:171-182`.

Keyword branch:

```python
hits = sum(1 for kw in keywords if kw in prompt_lower)
return min(hits / 2.0, 1.0)
```

Evidence: `agent/router.py:184-189`. This is substring matching, not token-boundary matching, so words like `class` can match inside unrelated text.

Structured output branch:

```python
hits = sum(1 for pat in _STRUCTURED_OUTPUT_PATTERNS if pat.search(prompt))
return min(hits / 2.0, 1.0)
```

Evidence: `agent/router.py:191-195`; patterns are JSON/table/CSV/Markdown/YAML/XML (`agent/router.py:81-89`).

Multi-part branch:

```python
_MULTI_PART_PATTERN = re.compile(
    r"(\d+[\.\)]\s)|(\b(and also|additionally|furthermore|moreover)\b)",
    re.IGNORECASE,
)
matches = _MULTI_PART_PATTERN.findall(prompt)
return min(len(matches) / 3.0, 1.0)
```

Evidence: `agent/router.py:91-95`, `agent/router.py:197-201`.

Question-depth branch:

```python
q_count = prompt.count("?")
if q_count <= 1: return 0.1
if q_count <= 3: return 0.5
return 0.9
```

Evidence: `agent/router.py:203-211`.

Sentence-count branch:

```python
sentences = re.split(r'[.!?]+', prompt)
count = len([s for s in sentences if s.strip()])
if count <= 1: return 0.05
if count <= 2: return 0.2
if count <= 4: return 0.5
if count <= 6: return 0.75
return 1.0
```

Evidence: `agent/router.py:213-227`.

### Model Selection and `ALLOWED_MODELS`

`ALLOWED_MODELS` is parsed into a list:

```python
allowed_models: list[str] = field(
    default_factory=lambda: [
        m.strip()
        for m in os.getenv("ALLOWED_MODELS", "").split(",")
        if m.strip()
    ]
)
```

Evidence: `agent/config.py:40-46`.

Selection uses a fixed keyword priority and picks the sorted first model:

```python
size_priority = ["8b", "3b", "1b", "scout", "small", "mini", "instant",
                 "70b", "maverick", "large", "405b"]
selected = sorted(config.allowed_models, key=model_priority)[0]
```

Evidence: `agent/config.py:147-160`.

Flaw: the stated comment says "smaller models first," but the priority places `"8b"` before `"3b"` and `"1b"` (`agent/config.py:147-150`), so it may choose a larger model over smaller allowed models.

### Local Model or Classifier?

There is a local model executor, not a local router classifier:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
self._tokenizer = AutoTokenizer.from_pretrained(...)
self._model = AutoModelForCausalLM.from_pretrained(...)
```

Evidence: `agent/executor.py:291-312`.

Routing remains pure heuristics (`agent/router.py:4-6`, `agent/router.py:102-109`). There is no learned classifier, embedding model, or semantic router.

## 3. Token Efficiency Mechanisms

| Technique | Implementation | Estimated token savings |
|---|---|---|
| Heuristic local/remote routing | `score >= threshold ? remote : local` (`agent/router.py:131-135`). | Potentially high if local model works; on sample tasks default routing is 20/20 local, so remote scored-token savings could be near 100%, but accuracy/runtime depend on local HF model. |
| Rule-based arithmetic fast path | Regex for one binary arithmetic expression on prompts with <=20 words (`agent/executor.py:202-225`, `agent/executor.py:238-260`). | Low overall; sample set has 1/20 = 5% zero-token rule hits. For arithmetic-only workloads it can save 100% on matching tasks. |
| In-memory normalized exact-match cache | Lowercase/strip/collapse whitespace/trim punctuation then hash (`agent/cache.py:60-73`). | Depends on duplicate prompts in one process. With no duplicates, 0%. With exact normalized duplicates, 100% for repeat calls. |
| Prompt compression | Regex removes filler phrases and collapses whitespace (`agent/compressor.py:20-100`). | Usually small, likely 0-10% prompt-token savings; larger only for polite/verbose prompts. It does not summarize content. |
| Dynamic max_tokens | Category budget table plus 30% complexity bump and long-prompt floor (`agent/budget.py:82-121`). | Completion-token cap savings versus `FIREWORKS_MAX_TOKENS=512`; e.g. sentiment 80 saves up to 84%, simple_math 30 saves up to 94%, code 400 saves up to 22%, assuming model would otherwise use the cap. |
| Category-specific system prompts | Category detector chooses concise system prompt (`agent/executor.py:47-56`, `agent/executor.py:59-109`, `agent/executor.py:138-140`). | Indirect; may reduce rambly completions, but adds system prompt tokens to every remote call. |
| Cascading fallback | Local result verified by confidence/quality, else remote (`agent/executor.py:568-618`). | Saves remote tokens only when local generation succeeds and verifier accepts. If local model load fails, all local-routed non-rule tasks fallback remote. |
| Smallest allowed model selection | Keyword sort of `ALLOWED_MODELS` (`agent/config.py:147-160`). | Model-cost efficiency, not token-count efficiency. Implementation priority is questionable (`8b` before `3b`/`1b`). |

### Cache Type

The cache is exact normalized matching, not semantic/embedding-based:

```python
text = prompt.lower().strip()
text = re.sub(r"\s+", " ", text)
text = text.rstrip("?.! ")
key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
entry = self._cache.get(key)
```

Evidence: `agent/cache.py:60-85`. There are no embeddings, vector indexes, approximate nearest-neighbor searches, or semantic similarity thresholds.

Bug/semantic mismatch: cached remote results are returned as `Route.LOCAL`:

```python
route_used=Route.LOCAL,  # Cached = free = local
```

Evidence: `agent/cache.py:88-99`. This makes scored tokens zero, but loses provenance of whether the original answer came from remote; `CacheEntry.route_used` is stored (`agent/cache.py:116-120`) but not used on retrieval.

### Dynamic Max Tokens: Real Numbers

Base budgets:

```python
_CATEGORY_BUDGETS = {
    "code": 400,
    "sentiment": 80,
    "ner": 150,
    "summarization": 100,
    "math": 200,
    "simple_math": 30,
    "logic": 200,
    "explanation": 200,
    "general": 150,
}
```

Evidence: `agent/budget.py:82-94`.

Adjustments:

```python
if decision.complexity_score >= 0.6:
    base_budget = int(base_budget * 1.3)
if word_count > 50 and category not in ("simple_math", "sentiment"):
    base_budget = max(base_budget, 250)
return min(base_budget, 600)
```

Evidence: `agent/budget.py:112-121`.

Effective maxima:

| Category | Base | If complexity >=0.6 | Long-prompt floor | Final cap |
|---|---:|---:|---:|---:|
| code | 400 | 520 | max(520, 250) | <=600 |
| sentiment | 80 | 104 | no long floor | <=600 |
| ner | 150 | 195 | max(195, 250)=250 if >50 words | <=600 |
| summarization | 100 | 130 | 250 if >50 words | <=600 |
| math | 200 | 260 | 260 if >50 words | <=600 |
| simple_math | 30 | 39 | no long floor | <=600 |
| logic | 200 | 260 | 260 if >50 words | <=600 |
| explanation | 200 | 260 | 260 if >50 words | <=600 |
| general | 150 | 195 | 250 if >50 words | <=600 |

## 4. Prompt Engineering

All remote system prompts:

```python
_CATEGORY_PROMPTS = {
    "sentiment": "Classify the sentiment (positive/negative/neutral). State the label first, then briefly justify.",
    "ner": "Extract all named entities. For each, state the text and its type (Person, Organization, Location, Date, etc.).",
    "summarization": "Summarize the text concisely. Follow any length or format constraints given.",
    "code_debug": "Identify the bug in the code. Show the corrected implementation.",
    "code_gen": "Write correct, well-structured code that satisfies the spec. Include only the code.",
    "math": "Solve step-by-step. State the final numerical answer clearly.",
    "logic": "Reason through the constraints step-by-step. State the conclusion clearly.",
    "factual": "Answer accurately and concisely.",
}
```

Evidence: `agent/executor.py:47-56`.

The remote call uses exactly one system message and one user message:

```python
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]
```

Evidence: `agent/executor.py:149-157`.

Assessment: prompts are minimal and mostly efficient. The code/debug/code-gen prompts may be too terse for robust judge compliance. Math/logic explicitly require step-by-step reasoning, which improves accuracy but increases completion tokens.

## 5. Code Quality Audit

### Error Handling

Config loading in `run.py` catches all exceptions and exits non-zero:

```python
except Exception as exc:
    logger.error("Failed to load config: %s", exc)
    return 1
```

Evidence: `run.py:70-72`.

Malformed/missing input catches exceptions and exits non-zero:

```python
if not input_path.exists():
    logger.error("Input file not found: %s", input_path)
    return 1
...
except Exception as exc:
    logger.error("Failed to read input tasks: %s", exc)
    return 1
```

Evidence: `run.py:79-100`.

Per-task exceptions do not crash the whole run:

```python
except Exception as exc:
    logger.error("Task %s failed: %s", task.id, exc)
    answer = ""
```

Evidence: `run.py:136-139`.

Remote API failure returns an error string, not an exception:

```python
except Exception as exc:
    return ExecutionResult(
        output=f"[ERROR] Remote model call failed: {exc}",
        route_used=Route.REMOTE,
        token_usage=TokenUsage(),
        confidence=0.0,
    )
```

Evidence: `agent/executor.py:176-186`.

Local model failure is caught in two layers: `LocalExecutor.execute` returns an error result (`agent/executor.py:391-400`), and `HybridExecutor` also catches exceptions and calls remote (`agent/executor.py:568-585`). Because `LocalExecutor.execute` catches its own errors, many failures flow to verifier rejection rather than the outer exception branch.

Runtime timeout handling is coarse:

```python
if elapsed > MAX_RUNTIME_SECONDS:
    ... break
```

Evidence: `run.py:113-121`. It checks only between tasks; one long API/local call can still exceed the 10-minute limit.

### Edge Cases Handled

| Handled edge case | Evidence |
|---|---|
| Missing input file | `run.py:79-81`. |
| Empty/invalid tasks after normalization | `run.py:93-96`. |
| Missing task `id` in local CLI gets generated | `main.py:81-85`. |
| Single-task JSON object in local CLI | `main.py:77-80`. |
| Division by zero in rule math returns `inf` instead of crashing | `agent/executor.py:250-251`. |
| Error responses are not cached | `agent/cache.py:109-111`. |
| Low-confidence/bad local outputs fallback remote | `agent/executor.py:587-618`. |

### Edge Cases Missed or Weak

| Issue | Evidence and impact |
|---|---|
| `run.py` assumes input JSON is iterable list; if JSON is a dict, loop iterates keys and `item.get` crashes. | `run.py:83-92`; unlike `main.py`, no dict-to-list normalization. |
| Invalid task entries are silently skipped. | `run.py:87-92`; no per-item warning. |
| No API timeout/retry/backoff. | Remote call has no timeout or retry params (`agent/executor.py:148-157`). |
| Remote error string can be emitted as final answer with exit code 0. | `RemoteExecutor` returns `[ERROR]...` (`agent/executor.py:176-186`), then `run.py` stores `answer = result.output` (`run.py:123-143`). |
| Cache is process-local only; no persistence across runs. | Cache is plain dict initialized in memory (`agent/cache.py:55-58`). |
| README says semantic cache, but implementation is exact normalized hash. | README claim (`README.md:43-45`, `README.md:123-124`) vs code (`agent/cache.py:60-85`). |
| Compose command is ineffective with `run.py`. | `docker-compose.yml:23`, `Dockerfile:81`, `run.py:32-33`. |
| `ALLOWED_MODELS` priority prefers `8b` before `3b` and `1b`. | `agent/config.py:147-150`. |
| Category detectors are duplicated and can diverge. | Executor detector (`agent/executor.py:59-109`) and budget detector (`agent/budget.py:23-75`) are separate. |
| Router keyword matching is substring-based. | `kw in prompt_lower` (`agent/router.py:184-189`). |
| Local model dependencies make Docker image heavy. | `torch`, `transformers`, `accelerate`, `huggingface-hub` in `requirements.txt:2-7`. |

### Dependency Audit

| Dependency | Used? | Notes |
|---|---|---|
| `openai` | Yes | Remote Fireworks API client (`agent/executor.py:22`, `agent/executor.py:118-121`). |
| `transformers` | Yes, only if local path used | Lazy import in `LocalExecutor` (`agent/executor.py:291-312`). Heavy. |
| `torch` | Yes, only if local path used | Lazy import for local generation/confidence (`agent/executor.py:292`, `agent/executor.py:335`, `agent/executor.py:411`). Very heavy. |
| `pydantic` | Yes | Data schemas (`agent/models.py:14`, `agent/models.py:38-130`). |
| `python-dotenv` | Yes | Loads `.env` (`agent/config.py:16-20`). |
| `huggingface-hub` | Indirect | Useful for Transformers downloads, not directly imported. |
| `accelerate` | Indirect | Required by `device_map="auto"` in Transformers. |

## 6. Efficiency Estimate

### Static Routing on Included Sample Tasks

Using the repository defaults (`ROUTER_COMPLEXITY_THRESHOLD=0.6`), the 20 included `eval/test_tasks.json` tasks route:

| Task | Route | Score | Difficulty | Budget category | Budget |
|---|---|---:|---|---|---:|
| fact-1 | local | 0.250 | simple | explanation | 200 |
| fact-2 | local | 0.325 | simple | explanation | 200 |
| fact-3 | local | 0.175 | simple | explanation | 200 |
| math-1 | local | 0.225 | simple | math | 200 |
| math-2 | local | 0.360 | moderate | math | 200 |
| math-3 | local | 0.345 | simple | math | 200 |
| sent-1 | local | 0.470 | moderate | sentiment | 80 |
| sent-2 | local | 0.210 | simple | sentiment | 80 |
| summ-1 | local | 0.455 | moderate | summarization | 250 |
| summ-2 | local | 0.525 | moderate | summarization | 250 |
| ner-1 | local | 0.095 | simple | ner | 150 |
| ner-2 | local | 0.075 | simple | ner | 150 |
| debug-1 | local | 0.120 | simple | code | 400 |
| debug-2 | local | 0.095 | simple | code | 400 |
| logic-1 | local | 0.230 | simple | logic | 200 |
| logic-2 | local | 0.450 | moderate | logic | 200 |
| code-1 | local | 0.365 | moderate | code | 400 |
| code-2 | local | 0.395 | moderate | code | 400 |
| code-3 | local | 0.545 | moderate | code | 400 |
| math-4 | local | 0.100 | simple | simple_math | 30 |

Implications:

* 20/20 sample tasks initially route to local.
* 1/20 sample tasks are guaranteed 0 Fireworks tokens via rule arithmetic (`math-4`).
* The other 19/20 depend on local model availability and verifier acceptance; if local model loading fails, they fallback remote.

### Likely 0 Fireworks Token Rate

Code-guaranteed 0-token rate:

* Cache duplicates: unknown workload-dependent; 0% if all prompts unique.
* Rule arithmetic: likely small; 5% on included sample set.
* Local model accepted: high in theory if a local model is available and accurate, but not guaranteed in the container because model weights are not baked into Docker and may require network/download/GPU/CPU time.

Practical estimate:

* With no duplicate prompts and no preloaded local model: about 5% guaranteed 0 Fireworks tokens on sample-like tasks, because only simple regex arithmetic bypasses model calls.
* With local model successfully downloaded and accepted by verifier: up to 100% of the included sample initially tries local, but accuracy may suffer and local runtime may be large.

### Smallest vs Bigger Model

There is no per-task remote model escalation. All remote calls use one resolved model:

```python
self._resolved_model = get_resolved_model(config)
...
model=self._resolved_model
```

Evidence: `agent/executor.py:115-121`, `agent/executor.py:148-157`.

Therefore:

* `% smallest model`: 100% of remote calls use the single selected `ALLOWED_MODELS` winner.
* `% escalate to bigger model`: 0%; no code path changes model by task difficulty or confidence.

### Bottlenecks/Inefficiencies

* Local HF model lazy-load can dominate runtime; `AutoModelForCausalLM.from_pretrained` happens on first local task (`agent/executor.py:281-320`).
* No remote retries/timeouts (`agent/executor.py:148-157`).
* `main.py` computes `token_budget` but `HybridExecutor.execute` recomputes it (`main.py:112-116`, `agent/executor.py:548-550`).
* Category detection duplicated (`agent/executor.py:59-109`, `agent/budget.py:23-75`).
* Cache check happens before rule fast path (`agent/executor.py:542-556`), which is fine for repeats but increments cache miss on all first-time rule-solvable tasks.
* Local token accounting records local tokens (`agent/executor.py:381-385`) but scoring ignores them (`agent/models.py:125-130`); tracker reports "free local tokens" separately (`agent/tracker.py:161-163`).

## 7. Gaps / Weaknesses vs State-of-the-Art LLM Routing

| Missing/weak capability | Current implementation |
|---|---|
| Learned router/classifier | Pure heuristics (`agent/router.py:4-6`, `agent/router.py:126-151`). |
| Semantic routing | No embeddings or model confidence for routing; keyword and regex only (`agent/router.py:155-241`). |
| Semantic cache | Normalized hash only (`agent/cache.py:60-85`). |
| Confidence-based remote model escalation | Only local-to-remote fallback; no small-remote-to-large-remote escalation (`agent/executor.py:587-618`). |
| Multi-model policy per category | Single resolved model for all remote calls (`agent/executor.py:115-121`, `agent/executor.py:148-157`). |
| Cost/latency-aware routing | Router ignores model latency, historical accuracy, prompt token estimate, and API cost; uses prompt heuristics only. |
| Online feedback/calibration | No learning from eval outcomes; tracker only reports (`agent/tracker.py:99-250`). |
| Persistent cross-run cache | Cache is an in-memory dict (`agent/cache.py:55-58`). |
| Robust structured-output validation | Verifier checks generic length/repetition/gibberish, not JSON/schema/category correctness (`agent/executor.py:443-480`). |
| Retry/backoff/timeouts | Absent from remote execution (`agent/executor.py:148-157`). |

Notable bugs/logic flaws:

1. `docker-compose.yml` does not feed `run.py` because `run.py` ignores CLI args (`docker-compose.yml:23`, `Dockerfile:81`, `run.py:32-33`).
2. README claims semantic caching, but cache is exact normalized hash (`README.md:43-45`, `agent/cache.py:60-85`).
3. `ALLOWED_MODELS` priority chooses `8b` before `3b`/`1b` (`agent/config.py:147-150`).
4. Cached remote provenance is overwritten as `Route.LOCAL` (`agent/cache.py:88-99`).
5. Single-object JSON is supported by `main.py` but not `run.py` (`main.py:77-80`, `run.py:83-92`).
6. Router threshold default causes all included eval tasks to route local, including code/debug/math reasoning (`Dockerfile:71`, sample static route table above).
7. `total_tokens = 0` in `run.py` summary is assigned but never used (`run.py:159-163`).

## 8. Comparison Table

| Technique | Used? | Efficiency impact |
|---|---|---|
| Rule-based routing | Yes | Zero routing tokens; can avoid remote calls if local path succeeds. |
| Keyword/regex category detection | Yes | Cheap; brittle classification. |
| Learned router | No | Missing potential accuracy/cost calibration. |
| Semantic routing / embeddings | No | Misses paraphrase-aware routing. |
| Exact normalized cache | Yes | 100% savings for repeated normalized prompts in same process. |
| Semantic cache | No | README claim is inaccurate; misses paraphrase duplicate savings. |
| Persistent cache | No | No savings across container runs. |
| Prompt compression | Yes | Small prompt-token savings, mostly filler/whitespace. |
| Prompt summarization/compression by model | No | No deep compression for long inputs. |
| Category-specific system prompts | Yes | Likely accuracy/verbosity benefit; adds prompt tokens. |
| Dynamic `max_tokens` | Yes | Strong completion cap savings vs 512 default. |
| Rule-based arithmetic | Yes | Free for simple binary arithmetic; narrow regex coverage. |
| Local model execution | Yes | Zero scored tokens if model is available and accepted. |
| Confidence-based local fallback | Yes | Avoids bad local outputs but can increase remote usage. |
| Remote small-to-large escalation | No | No quality recovery using larger allowed model. |
| Per-category model choice | No | All remote tasks use one selected model. |
| API retries/backoff/timeouts | No | Reliability risk and possible wasted runtime. |
| Structured validators | No | Generic verifier only; no category-specific correctness checks. |

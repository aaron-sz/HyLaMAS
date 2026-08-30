"""Remote generation backends for the `baseline` and `text_mas` controls.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import debug


class RemoteBackendError(RuntimeError):
    pass


def _split_system(messages: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
    """Bedrock carries the system prompt outside the message list."""
    system, rest = None, []
    for m in messages:
        if m.get("role") == "system" and system is None:
            system = m.get("content", "")
        else:
            rest.append(m)
    return system, rest


class _Base:
    """Shared retry/concurrency scaffolding and the caller-facing contract."""

    kind = "base"

    def __init__(self, concurrency: int, retries: int) -> None:
        self.concurrency = max(1, int(concurrency))
        self.retries = max(1, int(retries))
        self._calls = 0
        self._prompt_tokens = 0

    def _one(self, prompt, messages, max_new_tokens, temperature, top_p):
        raise NotImplementedError

    def generate(self, prompts: List[str], messages: List[List[Dict]], *,
                 max_new_tokens: int, temperature: float, top_p: float
                 ) -> Tuple[List[str], List[int], List[Optional[bool]], Dict]:
        """One completion per item, issued concurrently, order preserved."""
        t0 = time.perf_counter()
        self._calls += len(prompts)
        items = list(zip(prompts, messages))
        with ThreadPoolExecutor(max_workers=min(self.concurrency, max(1, len(items)))) as pool:
            results = list(pool.map(
                lambda pm: self._one(pm[0], pm[1], max_new_tokens, temperature, top_p),
                items))
        stats = {
            "decode_sec": round(time.perf_counter() - t0, 4),
            "truncated": [r[2] for r in results],
            "max_new_tokens": max_new_tokens,
            "append_sec": 0.0, "append_forwards": 0, "append_positions": 0,
            "backend": self.kind,
        }
        return [r[0] for r in results], [r[1] for r in results], stats["truncated"], stats

    def probe(self) -> None:
        probe_msgs = [{"role": "user", "content": "hi"}]
        try:
            text, n, _ = self._one("hi", probe_msgs, 4, 0.0, 1.0)
        except Exception as exc:
            raise RemoteBackendError(
                f"{self.kind} backend probe failed: {exc}") from exc
        debug.dbg(f"backend: probe ok, {self.kind} returned {n} token(s)")

    def describe(self) -> Dict:
        return {"kind": self.kind, "concurrency": self.concurrency,
                "completions_requested": self._calls,
                "server_prompt_tokens": self._prompt_tokens}


class OpenAICompatBackend(_Base):
    """OpenAI-compatible /v1/completions. Stdlib only, no new dependency."""

    kind = "openai"

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None,
                 concurrency: int = 8, timeout: float = 600.0, retries: int = 4) -> None:
        super().__init__(concurrency, retries)
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: Dict) -> Dict:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(f"{self.base_url}{path}", data=body,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
                if exc.code not in (408, 429) and exc.code < 500:
                    break          # a 4xx will not fix itself
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self.retries:
                time.sleep(2 ** attempt)
        raise RemoteBackendError(f"{self.base_url}{path}: {last}")

    def _one(self, prompt, messages, max_new_tokens, temperature, top_p):
        payload = {"model": self.model, "prompt": prompt,
                   "max_tokens": int(max_new_tokens),
                   "temperature": float(temperature) if temperature and temperature > 0 else 0.0}
        if temperature and temperature > 0:
            payload["top_p"] = float(top_p)
        data = self._post("/completions", payload)
        try:
            choice = data["choices"][0]
        except Exception as exc:
            raise RemoteBackendError(f"unexpected response: {str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
        n = usage.get("completion_tokens")
        finish = choice.get("finish_reason")
        return ((choice.get("text") or "").strip(),
                int(n) if n is not None else -1,
                (finish == "length") if finish is not None else None)

    def describe(self) -> Dict:
        d = super().describe()
        d.update({"base_url": self.base_url, "model": self.model})
        return d


class BedrockBackend(_Base):
    """Amazon Bedrock `Converse`.
    """

    kind = "bedrock"

    def __init__(self, model_id: str, region: Optional[str] = None,
                 profile: Optional[str] = None, concurrency: int = 4,
                 timeout: float = 600.0, retries: int = 5) -> None:
        super().__init__(concurrency, retries)
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RemoteBackendError(
                "--backend bedrock needs boto3:  pip install boto3") from exc
        self.model_id = model_id
        self.region = region or os.environ.get("AWS_REGION") or "us-east-1"
        session = (boto3.Session(profile_name=profile, region_name=self.region)
                   if profile else boto3.Session(region_name=self.region))
        # Bedrock throttles per-account; let botocore back off before we do.
        cfg = Config(read_timeout=timeout, connect_timeout=30,
                     retries={"max_attempts": self.retries, "mode": "adaptive"})
        self.client = session.client("bedrock-runtime", config=cfg)

    def _one(self, prompt, messages, max_new_tokens, temperature, top_p):
        system, rest = _split_system(messages or [{"role": "user", "content": prompt}])
        payload = {
            "modelId": self.model_id,
            "messages": [{"role": m["role"],
                          "content": [{"text": m.get("content", "")}]} for m in rest],
            "inferenceConfig": {
                "maxTokens": int(max_new_tokens),
                "temperature": float(temperature) if temperature and temperature > 0 else 0.0,
            },
        }
        if temperature and temperature > 0:
            payload["inferenceConfig"]["topP"] = float(top_p)
        if system:
            payload["system"] = [{"text": system}]

        last = None
        for attempt in range(self.retries):
            try:
                resp = self.client.converse(**payload)
                blocks = (resp.get("output", {}).get("message", {}).get("content") or [])
                text = "".join(b.get("text", "") for b in blocks).strip()
                usage = resp.get("usage") or {}
                self._prompt_tokens += int(usage.get("inputTokens") or 0)
                n = usage.get("outputTokens")
                stop = resp.get("stopReason")
                return (text, int(n) if n is not None else -1,
                        (stop == "max_tokens") if stop is not None else None)
            except Exception as exc:  # ThrottlingException and friends
                last = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.retries:
                    time.sleep(min(30, 2 ** attempt))
        raise RemoteBackendError(f"bedrock converse({self.model_id}): {last}")

    def describe(self) -> Dict:
        d = super().describe()
        d.update({"model": self.model_id, "region": self.region,
                  "template_controlled_by": "aws (Converse applies the served "
                                            "model's own chat template)"})
        return d


def build_backend(args):
    """Construct the backend named by --backend, or None for the local path."""
    kind = getattr(args, "backend", "hf")
    if kind in (None, "hf"):
        return None

    if getattr(args, "method", None) == "latent_mas":
        raise ValueError(
            "--backend cannot run latent_mas. The method is the shared hybrid "
            "cache: agent n's Gated DeltaNet recurrent state has to reach agent "
            "n+1, and no serving API exposes it. Run latent_mas locally and use "
            "a remote backend only for --method baseline / text_mas.")

    if kind == "openai":
        base, model = getattr(args, "api_base", None), getattr(args, "api_model", None)
        if not base or not model:
            raise ValueError("--backend openai requires --api_base and --api_model")
        backend = OpenAICompatBackend(
            base_url=base, model=model,
            api_key=os.environ.get(getattr(args, "api_key_env", "OPENAI_API_KEY") or ""),
            concurrency=int(getattr(args, "api_concurrency", 8) or 8),
            timeout=float(getattr(args, "api_timeout", 600.0) or 600.0))
    elif kind == "bedrock":
        model = getattr(args, "api_model", None)
        if not model:
            raise ValueError(
                "--backend bedrock requires --api_model (a Bedrock model id or "
                "inference-profile arn, e.g. us.anthropic.claude-sonnet-4-20250514-v1:0)")
        backend = BedrockBackend(
            model_id=model, region=getattr(args, "aws_region", None),
            profile=getattr(args, "aws_profile", None),
            concurrency=int(getattr(args, "api_concurrency", 4) or 4),
            timeout=float(getattr(args, "api_timeout", 600.0) or 600.0))
    else:
        raise ValueError(f"unknown --backend {kind!r}")

    backend.probe()
    return backend

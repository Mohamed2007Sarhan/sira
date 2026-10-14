"""Configuration. Everything lives in one JSON-serialisable dataclass."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path


@dataclass
class Config:
    # ---- goal ---------------------------------------------------------------
    goal_text: str = "Become the strongest model possible on the SIRA Fresh Set"
    target_score: float = 0.90        # aggregate Fresh-Set pass rate that counts as "goal reached"
    confirm_rounds: int = 2           # must hold on N independent fresh sets in a row
    base_model: str = ""              # Hugging Face id (org/name) or local directory
    run_root: str = "runs"

    # ---- budget (the ONLY automatic stop besides goal/human) ---------------
    budget_usd: float = 50.0          # LLM API spend of the agents' "brain"
    budget_gpu_hours: float = 24.0    # wall time of training / generation processes
    budget_wall_hours: float = 72.0   # total wall-clock time of the run

    # ---- agents' brain (LLM) ------------------------------------------------
    llm_provider: str = "anthropic"   # anthropic | ollama | openai | openrouter | groq | together | mistral | deepseek | gemini | xai |
                                      # lmstudio | vllm | llamacpp | openai_compatible | mock
    llm_model: str = "claude-sonnet-5-5"
    llm_base_url: str = ""            # custom endpoint (Ollama: http://host:11434, OpenAI-compatible: https://host/v1)
    llm_num_ctx: int = 16384          # Ollama context window (its default is far too small for agents)
    llm_timeout_s: int = 300
    log_llm_calls: bool = True        # record every brain call (inputs, outputs, tokens, latency) in <run>/llm_calls.jsonl (local only)
    llm_json_mode: bool = False       # ask the server for JSON output (Ollama: format=json; OpenAI-style: response_format)
    llm_api_key_env: str = ""         # env var holding the key (defaults per provider)
    price_in_per_mtok: float = 3.0    # USD per 1M input tokens (adjust to your model)
    price_out_per_mtok: float = 15.0  # USD per 1M output tokens
    llm_max_tokens: int = 4096
    agent_max_steps: int = 14
    parallel_experiments: int = 3

    # ---- sandbox ------------------------------------------------------------
    sandbox_backend: str = "auto"     # auto | docker | bwrap | subprocess
    allow_weak_sandbox: bool = False  # subprocess backend does NOT isolate the host filesystem
    sandbox_network: str = "none"     # none | open  (experiment code; web tools are always open)
    docker_image: str = "sira-sandbox:latest"
    sandbox_timeout_s: int = 120
    sandbox_mem_mb: int = 2048
    sandbox_cpus: float = 1.0
    sandbox_pids: int = 128
    sandbox_max_output_bytes: int = 200_000
    sandbox_max_file_mb: int = 256
    sandbox_disk_mb: int = 512        # total size of files the experiment may leave in its workspace
    sandbox_tmp_mb: int = 256

    # ---- open internet (read side) -----------------------------------------
    search_provider: str = "duckduckgo"   # duckduckgo | searxng
    searxng_url: str = ""
    block_private_ips: bool = True    # protects YOUR LAN/metadata services only; the public internet is fully open
    max_fetch_bytes: int = 3_000_000
    max_download_mb: int = 500

    web_mode: str = "open"            # open | allowlist | blocklist  (editable live in the GUI)
    web_allow: list = field(default_factory=list)
    web_block: list = field(default_factory=list)

    # ---- self-extension / workstation / backups -----------------------------
    allow_agent_installs: bool = True     # agents may pip-install into their persistent workstation (needs network)
    approve_new_tools: bool = False       # human must approve tools/worlds the agents create
    station_mb: int = 1024                # quota of the agents' persistent computer
    backup_every_min: int = 30
    backup_keep: int = 6
    backup_weights: bool = True
    keep_generations: int = 3
    force_train_every: int = 2            # train at least every N cycles (always teaching itself)
    try_merge_candidates: bool = True     # also test weight-interpolation (model-soup) candidates
    report_every_cycles: int = 3          # frontier research report cadence
    train_layers: list = field(default_factory=list)      # [] = all layers; else only these block indices (LoRA/full)
    freeze_patterns: list = field(default_factory=list)   # full fine-tune: parameter-name substrings to freeze

    # ---- from-scratch model ---------------------------------------------------
    scratch_mode: bool = False            # create the model from scratch instead of taking one from Hugging Face
    scratch_arch: str = "gqa"             # llama | mha | gqa | mqa | cla2 | cla3 | custom   (architecture of the model born from scratch)
    arch_kv_heads: int = 0                # custom: number of KV heads (0 = size default)
    arch_cla_factor: int = 1              # custom: layers per shared K/V group (1 = no cross-layer sharing)
    arch_tolerance: float = 0.02          # a converted architecture is kept if its score >= parent - tolerance (it saves KV memory)
    scratch_language: str = "en"          # wikipedia language code (en, ar, ...)
    scratch_size: str = "auto"            # auto | tiny | small | base
    scratch_corpus_dir: str = ""          # optional folder with .txt files (used before downloading)
    scratch_wiki_mb: int = 40             # amount of text to fetch
    scratch_steps: int = 2000
    scratch_curriculum_steps: int = 600

    thinking: str = "medium"              # off | low | medium | high - deliberation step in every layer (see thinking.py)
    human_baseline: dict = field(default_factory=dict)   # optional {"math": 0.9, "code": 0.85, ...} human-expert scores for comparison

    # ---- quest mode: a big goal + a few headings; research -> learn -> exam -> autopsy -> retry; life in a virtual society ----------
    mode: str = "model"                   # model (improve a model) | quest (pursue quest_goal) | both | auto (everything, goal optional)
    quest_goal: str = ""                  # e.g. "reach artificial consciousness" or "reach AGI"
    quest_seeds: list = field(default_factory=list)       # a few small headings to start from
    quest_topics_per_round: int = 2
    quest_skill_pass: float = 0.6         # mean challenge skill needed (EMA of best scores on fresh self-made challenges)
    challenge_domains_per_cycle: int = 2
    challenges_per_domain: int = 2
    attempts_per_challenge: int = 4
    quest_questions: int = 12             # exam questions per round
    quest_pass_score: float = 0.8         # exam score needed (mechanically graded)
    quest_pass_rounds: int = 2            # consecutive rounds at/above the pass score
    quest_min_coverage: float = 0.5       # share of the goal's criteria answered with valid evidence
    examiner_provider: str = ""           # blank = same brain. The examiner may be a SMALLER model: it only phrases/relevance-checks,
    examiner_model: str = ""              # ground truth always comes from experiments, the database and the world engine
    examiner_base_url: str = ""
    examiner_api_key_env: str = ""
    examiner_weight: float = 0.25         # max share of a rubric score that may come from the examiner model (0 disables it)
    society_enabled: bool = True          # the agent lives in a virtual society of humans (deterministic world engine)
    society_episodes: int = 10            # episodes per society round
    society_pass: float = 0.6             # life score needed for the quest to count as passed (when society is enabled)
    society_llm_dialogue: bool = False    # let an LLM voice the personas (flavour only; scoring never depends on it)

    # ---- verification -------------------------------------------------------
    replication_runs: int = 2         # independent re-runs (different seeds) before a claim is "verified"

    # ---- training -----------------------------------------------------------
    train_problems: int = 160
    eval_problems: int = 80
    samples_per_problem: int = 4
    keep_per_problem: int = 2
    max_new_tokens: int = 512
    gen_temperature: float = 0.8
    max_seq_len: int = 1024
    epochs: float = 1.0
    learning_rate: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    micro_batch: int = 2
    grad_accum: int = 8
    research_sft_fraction: float = 0.10
    bootstrap_when_stuck: bool = True   # if the model solves (almost) nothing, add verified solver lessons (train problems only)
    algorithm: str = "auto"           # auto | rft | grpo | dpo      (how the model learns from its own verified samples)
    lora_variant: str = "lora"        # lora | rslora | dora
    lora_plus_ratio: float = 1.0      # >1 enables LoRA+ (lr of the B matrices = ratio * lr)
    optimizer: str = "adamw"          # adamw | adamw8bit (needs bitsandbytes + CUDA)
    accept_p: float = 0.2             # a child must also win a paired exact test (McNemar) at this p-value; 1.0 disables
    auto_install: bool = True         # install missing Python packages (training stack) at start-up
    auto_adapt: bool = True           # adapt batch sizes / parallelism / limits to this machine at start-up
    domains: list = field(default_factory=lambda: ["math", "logic", "physics", "chemistry", "code"])   # evaluation domains ("arith" = small-model ladder)
    training_method: str = "auto"     # auto | lora | qlora | full  (auto: LoRA/QLoRA for ready models, full for models born inside SIRA)
    worker_inprocess: bool = False    # run model workers inside this process (fast tests on tiny models; default is a separate process)
    retention_tol: float = 0.10       # a child is rejected if a capability the parent had (>= 50%) drops by more than this
    retention_min_n: int = 6          # minimum problems in a capability bucket for the guard to apply
    replay_fraction: float = 0.30     # share of replayed earlier verified examples mixed into every RFT round (prevents forgetting)
    nll_gain: float = 0.05            # progress lane: accept if exact match is not worse, retention is intact and held-out NLL improves by this (nats)
    val_fraction: float = 0.12        # share of RFT rows held out (never trained on) to measure that NLL
    min_gain: float = 0.01            # a child is accepted only if it beats its parent by this margin
    allow_cpu_training: bool = False
    require_approval: bool = False    # human approves each training run in the GUI

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in names})
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.sandbox_network not in ("none", "open"):
            raise ValueError("sandbox_network must be 'none' or 'open'")
        if self.sandbox_backend not in ("auto", "docker", "bwrap", "subprocess"):
            raise ValueError("bad sandbox_backend")
        if self.training_method not in ("auto", "lora", "qlora", "full"):
            raise ValueError("training_method must be auto|lora|qlora|full")
        from .evalcore import ALL_DOMAINS
        if not self.domains or any(d not in ALL_DOMAINS for d in self.domains):
            raise ValueError(f"domains must be a non-empty subset of {ALL_DOMAINS}")
        if self.mode not in ("model", "quest", "both", "auto"):
            raise ValueError("mode must be model|quest|both|auto")
        if self.thinking not in ("off", "low", "medium", "high"):
            raise ValueError("thinking must be off|low|medium|high")
        if self.web_mode not in ("open", "allowlist", "blocklist"):
            raise ValueError("web_mode must be open|allowlist|blocklist")
        if self.scratch_arch not in ("llama", "mha", "gqa", "mqa", "cla2", "cla3", "custom") or self.arch_cla_factor < 1 or self.arch_kv_heads < 0:
            raise ValueError("bad scratch_arch / arch_cla_factor / arch_kv_heads")
        if self.scratch_size not in ("auto", "tiny", "small", "base"):
            raise ValueError("bad scratch_size")
        from .llm import PROVIDERS
        if self.llm_provider not in PROVIDERS:
            raise ValueError(f"llm_provider must be one of {PROVIDERS}")
        if self.llm_provider == "openai_compatible" and not self.llm_base_url:
            raise ValueError("llm_provider 'openai_compatible' needs llm_base_url")
        if self.algorithm not in ("auto", "rft", "grpo", "dpo") or self.lora_variant not in ("lora", "rslora", "dora") \
                or self.optimizer not in ("adamw", "adamw8bit"):
            raise ValueError("bad algorithm / lora_variant / optimizer")
        if not 0 < self.target_score <= 1:
            raise ValueError("target_score must be in (0,1]")
        for k in ("budget_usd", "budget_gpu_hours", "budget_wall_hours"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be > 0")

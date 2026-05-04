from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    huggingface_token: str
    hf_llm_model: str
    hf_embed_model: str


def get_settings() -> Settings:
    load_dotenv()

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

    return Settings(
        huggingface_token=token,
        hf_llm_model=os.getenv(
            "HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"
        ).strip(),
        hf_embed_model=os.getenv(
            "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
    )


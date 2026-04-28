import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gorgon.models.backbone import load_backbone_4bit


def main() -> None:
    model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download the model.")
    model, tokenizer, heads = load_backbone_4bit(
        model_name=model_name,
        num_heads=4,
        token=hf_token,
    )
    print("Model loaded:", model_name)
    print("Hidden size:", model.config.hidden_size)
    print("Vocab size:", model.config.vocab_size)
    print("Heads:", len(heads))
    print("Tokenizer vocab size:", tokenizer.vocab_size)


if __name__ == "__main__":
    main()

from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    token=HF_TOKEN,
    local_dir="models/llama-3.1-8b"
)

print("Model downloaded successfully!")
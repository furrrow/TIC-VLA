from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="a8cheng/navila-llama3-8b-8f",
    filename="TIC-VLA-model.ckpt",
    local_dir="models/ticvla",
)
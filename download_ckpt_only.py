from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(
    repo_id="OpenGVLab/InternVL3-1B",
    repo_type="model",
    local_dir="models/InternVL3-1B",
    local_dir_use_symlinks=False,
)
# TIC-VLA pretrained checkpoint (~1.94 GB)
hf_hub_download(
    repo_id="handsomeYun/TIC-VLA",
    repo_type="dataset",
    filename="TIC-VLA-model.ckpt",
    local_dir="models/ticvla",
)
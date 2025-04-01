from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="out-wiki-zh-ft-overfitted/ckpt.pt",
    path_in_repo="out-wiki-zh-ft-overfitted/ckpt.pt",
    repo_id="larrythwu/wiki-zh-GPT",
    repo_type="model",
)
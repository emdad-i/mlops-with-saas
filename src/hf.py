from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="app/model.pkl",
    path_in_repo="model.pkl",
    repo_id="emdad-i/testing"
)
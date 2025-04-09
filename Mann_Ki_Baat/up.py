from huggingface_hub import Repository

repo = Repository(local_dir="/home/krish/content/Mann_Ki_Baat/checkpoint1.2B", clone_from="krishm_08/MKB_HI-MR")

from huggingface_hub import upload_folder

upload_folder(
    repo_id="krishm_08/MKB_HI-MR",
    folder_path="/home/krish/content/Mann_Ki_Baat/checkpoint1.2B",  # Directory containing your model files
    commit_message="Add model files"
)


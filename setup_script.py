from huggingface_hub import snapshot_download

with open("./token.txt", "r") as f:
    token = f.readline().strip()

print("Downloading the Vector Database from Hugging Face.")
database_path = snapshot_download(
    repo_id="URL FOR VECTOR DB",
    repo_type="dataset",
    cache_dir="./database",
    token=token,
)
print("Download Completed!")

with open("database_path.txt", "w") as f:
    f.write(database_path+"/vector_db")

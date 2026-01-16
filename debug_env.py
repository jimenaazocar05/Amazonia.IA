import os
from dotenv import load_dotenv

# Try default load
load_dotenv()
print(f"Default load - HF_TOKEN: {bool(os.getenv('HF_TOKEN'))}")
print(f"Default load - HF_REPO_ID: {bool(os.getenv('HF_REPO_ID'))}")

# Try explicit path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
print(f"Checking path: {env_path}")
print(f"Path exists: {os.path.exists(env_path)}")

load_dotenv(env_path, override=True)
print(f"Explicit load - HF_TOKEN: {bool(os.getenv('HF_TOKEN'))}")
print(f"Explicit load - HF_REPO_ID: {bool(os.getenv('HF_REPO_ID'))}")
print(f"HF_REPO_ID Value: {os.getenv('HF_REPO_ID')}")

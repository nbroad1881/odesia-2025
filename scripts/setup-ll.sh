
git config --global credential.helper store
git config --global user.email $GITHUB_EMAIL
git config --global user.name $GITHUB_NAME

# git clone https://github.com/nbroad1881/odesia-2025.git

# pip install "unsloth[cu124-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
pip install scipy
pip install liger-kernel

export HF_HUB_ENABLE_HF_TRANSFER=1

# huggingface-cli download unsloth/Qwen2.5-7B-bnb-4bit --cache-dir /workspace

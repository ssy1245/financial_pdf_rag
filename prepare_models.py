"""复制固定版本的模型到项目目录；默认只读现有缓存，--download 才允许下载。"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODELS = {
    "BAAI/bge-small-en-v1.5": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}


def prepare(allow_download=False):
    root = ROOT / "models"
    root.mkdir(exist_ok=True)
    for repo, revision in MODELS.items():
        target = root / repo.split("/")[-1]
        if target.exists():
            print(f"已存在，保留：{target}")
            continue
        source = snapshot_download(repo, revision=revision, local_files_only=not allow_download)
        with tempfile.TemporaryDirectory(dir=root, prefix=".prepare-") as temporary:
            staged = Path(temporary) / "model"
            # 解引用缓存符号链接，项目文件不依赖原来的全局缓存。
            shutil.copytree(source, staged, symlinks=False)
            if not (staged / "config.json").is_file() or not list(staged.glob("*.safetensors")):
                raise ValueError(f"模型缓存不完整：{repo}")
            (staged / "local-model.json").write_text(json.dumps(
                {"repo": repo, "revision": revision}, indent=2) + "\n")
            staged.rename(target)
        print(f"已准备：{target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="允许下载缺失的模型文件")
    prepare(parser.parse_args().download)

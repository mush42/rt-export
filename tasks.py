# coding: utf-8


import os
os.putenv("HF_ASSETS_CACHE", os.path.join(os.path.dirname(__file__), ".hf.cache"))

import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from huggingface_hub.constants import HF_ASSETS_CACHE
from huggingface_hub import hf_api
from huggingface_hub import hf_hub_download
from invoke import task


ASSETS_DIR = Path(HF_ASSETS_CACHE).expanduser().resolve()
PIPER_CKPT_REPO = "rhasspy/piper-checkpoints"
REPO_TYPE = "dataset"

HERE = Path(os.path.dirname(__file__))
CHECKPOINTS_FILE = HERE / "checkpoints.json"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_DIR = HERE / "packed"
ZIP_DIR.mkdir(parents=True, exist_ok=True)


@task
def write_checkpoint_info(c, refresh=False):
    if not refresh and os.path.isfile(CHECKPOINTS_FILE):
        print("Info already downloaded. Doing nothing")
        return
    all_files = {
        PurePosixPath(pth)
        for pth in hf_api.list_repo_files(PIPER_CKPT_REPO, repo_type=REPO_TYPE)
    }
    all_checkpoints = filter(lambda p: p.match("*.ckpt"), all_files)
    keyed_checkpoints = {}
    for pth in all_checkpoints:
        key = "-".join(str(pth.parent).split("/")[1:])
        keyed_checkpoints[key] = {
            "checkpoint": str(pth),
            "config": str(pth.with_name("config.json")),
        }
        if pth.with_name("MODEL_CARD") in all_files:
            keyed_checkpoints[key]["model_card"] = str(pth.with_name("MODEL_CARD"))
    with open(CHECKPOINTS_FILE, "w", encoding="utf-8") as file:
        json.dump(keyed_checkpoints, file, ensure_ascii=False, indent=2)
    print(f"Wrote checkpoint info to file: `{file.name}`")


@task
def clone_and_checkout(c, force=False):
    if os.path.isdir("piper"):
        if force:
            shutil.rmtree("piper")
        else:
            return
    c.run("git clone https://github.com/mush42/piper")
    with c.cd("piper"):
        c.run("git checkout streaming")
    with c.cd("piper/src/python"):
        c.run("source ./build_monotonic_align.sh")


def export_single_checkpoint(c, voice_key, info):
    output_path = OUTPUT_DIR / voice_key
    print("Converting model")
    checkpoint = hf_hub_download(
        PIPER_CKPT_REPO,
        repo_type=REPO_TYPE,
        filename=info["checkpoint"]
    )
    c.run(
        "python -m piper_train.export_onnx_streaming --debug "
        f"{checkpoint} {str(output_path)}"
    )
    config = hf_hub_download(
        PIPER_CKPT_REPO,
        repo_type=REPO_TYPE,
        filename=info["config"]
    )
    shutil.copy(config, output_path)
    if "model_card" in info:
        model_card = hf_hub_download(
            PIPER_CKPT_REPO,
            repo_type=REPO_TYPE,
            filename=info["model_card"]
        )
        shutil.copy(model_card, output_path)
    with ZipFile.open(os.path.join(ZIP_DIR, f"{voice_key}.zip")) as zfile:
        for pth in output_path.iterdir():
            zfile.write(os.fspath(pth), pth.name)
    shutil.rmtree(ASSETS_DIR)


@task(
    pre=[clone_and_checkout, write_checkpoint_info,]
)
def run(c):
    with open(CHECKPOINTS_FILE, "r", encoding="utf-8") as file:
        checkpoint_info = json.load(file)
    os.chdir("./piper/src/python")
    for voice_key, info in checkpoint_info.items():
        export_single_checkpoint(c, voice_key, info)
    os.chdir(os.fspath(HERE))

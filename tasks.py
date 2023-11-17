# coding: utf-8


import json
import os
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from huggingface_hub import hf_api
from huggingface_hub import hf_hub_download
from invoke import task


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
        print("Checkpoint Info already downloaded. Doing nothing")
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


def export_single_checkpoint(c, voice_key, info, tmp_dir):
    lang, name, quality = voice_key.split("-")
    streaming_key = "-".join([lang, f"{name}+RT", quality])
    voice_zip = ZIP_DIR / f"{streaming_key}.zip"
    if voice_zip.is_file():
        print(f"Voice {streaming_key} already converted")
        return
    print(f"Making voice: {streaming_key}")
    output_path = OUTPUT_DIR / voice_key
    checkpoint = hf_hub_download(
        PIPER_CKPT_REPO,
        repo_type=REPO_TYPE,
        filename=info["checkpoint"],
        cache_dir=tmp_dir
    )
    c.run(
        "python -m piper_train.export_onnx_streaming --debug "
        f"{checkpoint} {str(output_path)}",
        hide=True
    )
    config = hf_hub_download(
        PIPER_CKPT_REPO,
        repo_type=REPO_TYPE,
        filename=info["config"],
        cache_dir=tmp_dir
    )
    config_dict = json.loads(
        Path(config).read_text(encoding="utf-8")
    )
    config_dict["key"] = streaming_key
    config_dict["streaming"] = True
    with open(output_path / "config.json", "w", encoding="utf-8") as cfg_file:
        json.dump(config_dict, cfg_file, ensure_ascii=False, indent=2)
    if "model_card" in info:
        model_card = hf_hub_download(
            PIPER_CKPT_REPO,
            repo_type=REPO_TYPE,
            filename=info["model_card"],
            cache_dir=tmp_dir
        )
        shutil.copy(model_card, output_path)
    with ZipFile(os.fspath(voice_zip), "w") as zfile:
        for pth in output_path.iterdir():
            zfile.write(os.fspath(pth), pth.name)
    print(f"Exported  voice: {streaming_key}")


@task(
    pre=[clone_and_checkout, write_checkpoint_info,]
)
def run(c):
    with open(CHECKPOINTS_FILE, "r", encoding="utf-8") as file:
        checkpoint_info = json.load(file)
    os.chdir("./piper/src/python")
    for voice_key, info in checkpoint_info.items():
        with TemporaryDirectory() as tmp_dir:
            export_single_checkpoint(c, voice_key, info, tmp_dir)
    os.chdir(os.fspath(HERE))
    print("Done all")
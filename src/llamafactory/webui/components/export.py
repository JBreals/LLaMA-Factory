# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import tempfile
from collections.abc import Generator
from typing import TYPE_CHECKING, Union
from urllib.parse import urlparse

from ...extras.constants import PEFT_METHODS
from ...extras.misc import torch_gc
from ...extras.packages import is_gradio_available
from ...train.tuner import export_model
from ..common import DEFAULT_EXPORT_ROOT, get_save_dir, load_config, normalize_model_path
from ..locales import ALERTS


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


GPTQ_BITS = ["8", "4", "3", "2"]


def _resolve_s3_local_model(path: str, model_name: str) -> str:
    r"""Map an S3 model path to the local cache root (MODEL_DOWNLOAD_DIR)."""
    cache_root = os.getenv("MODEL_DOWNLOAD_DIR", "/app/storage/models")
    # Prefer the explicit model name, otherwise use the last segment of the path.
    leaf = model_name
    if not leaf:
        parsed = urlparse(path) if path.startswith("s3://") else None
        raw_path = parsed.path if parsed else path
        leaf = raw_path.rstrip("/").split("/")[-1]
    local_path = os.path.join(cache_root, leaf)
    if not os.path.isdir(local_path):
        raise ValueError(f"Local S3 model not found under {local_path}. Download it to MODEL_DOWNLOAD_DIR first.")
    return local_path


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    r"""Return (bucket, prefix) from an s3:// URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def _upload_directory_to_s3(local_dir: str, target_uri: str) -> None:
    r"""Upload files under local_dir to the given s3:// URI using boto3."""
    try:
        import boto3
        from boto3.s3.transfer import S3Transfer, TransferConfig
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise ModuleNotFoundError("boto3 is required for uploading models to S3.") from exc

    bucket, prefix = _parse_s3_uri(target_uri)
    threshold = int(os.getenv("S3_MULTIPART_THRESHOLD", 64 * 1024 * 1024))
    chunk_size = int(os.getenv("S3_MULTIPART_CHUNKSIZE", 64 * 1024 * 1024))
    concurrency = int(os.getenv("S3_MAX_CONCURRENCY", 16))
    config = TransferConfig(
        multipart_threshold=threshold,
        multipart_chunksize=chunk_size,
        max_concurrency=max(1, concurrency),
        use_threads=True,
    )
    transfer = S3Transfer(boto3.client("s3"), config)

    try:
        for root, _, files in os.walk(local_dir):
            for name in files:
                local_path = os.path.join(root, name)
                rel_key = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
                s3_key = f"{prefix.rstrip('/')}/{rel_key}" if prefix else rel_key
                transfer.upload_file(local_path, bucket, s3_key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(str(exc)) from exc


def can_quantize(checkpoint_path: Union[str, list[str]]) -> "gr.Dropdown":
    if isinstance(checkpoint_path, list) and len(checkpoint_path) != 0:
        return gr.Dropdown(value="none", interactive=False)
    else:
        return gr.Dropdown(interactive=True)


def save_model(
    lang: str,
    model_name: str,
    hub_name: str,
    model_path: str,
    finetuning_type: str,
    checkpoint_path: Union[str, list[str]],
    template: str,
    export_size: int,
    export_quantization_bit: str,
    export_quantization_dataset: str,
    export_device: str,
    export_legacy_format: bool,
    export_dir: str,
    export_hub_model_id: str,
    extra_args: str,
) -> Generator[str, None, None]:
    user_config = load_config()
    error = ""
    model_path = normalize_model_path(model_path, hub_name)
    if hub_name == "s3" and model_path:
        try:
            model_path = _resolve_s3_local_model(model_path, model_name)
        except Exception as exc:
            error = str(exc)

    user_export_dir = export_dir
    upload_target = None
    is_s3_export = False
    if isinstance(user_export_dir, str):
        if user_export_dir.startswith("s3://"):
            upload_target = user_export_dir
            is_s3_export = True
        elif hub_name == "s3":
            upload_target = "s3://" + user_export_dir.lstrip("/")
            is_s3_export = True

    if is_s3_export:
        export_tmp_root = os.getenv("EXPORT_TMP_ROOT", "/app/storage/exports")
        os.makedirs(export_tmp_root, exist_ok=True)
        export_dir = tempfile.mkdtemp(prefix="export_", dir=export_tmp_root)
    else:
        if user_export_dir and not os.path.isabs(user_export_dir) and DEFAULT_EXPORT_ROOT:
            export_dir = os.path.join(DEFAULT_EXPORT_ROOT, user_export_dir)
        else:
            export_dir = user_export_dir

    if not model_name:
        error = ALERTS["err_no_model"][lang]
    elif not model_path:
        error = ALERTS["err_no_path"][lang]
    elif not user_export_dir:
        error = ALERTS["err_no_export_dir"][lang]
    elif export_quantization_bit in GPTQ_BITS and not export_quantization_dataset:
        error = ALERTS["err_no_dataset"][lang]
    elif export_quantization_bit not in GPTQ_BITS and not checkpoint_path:
        error = ALERTS["err_no_adapter"][lang]
    elif export_quantization_bit in GPTQ_BITS and checkpoint_path and isinstance(checkpoint_path, list):
        error = ALERTS["err_gptq_lora"][lang]

    try:
        json.loads(extra_args)
    except json.JSONDecodeError:
        error = ALERTS["err_json_schema"][lang]

    if error:
        gr.Warning(error)
        yield error
        return

    args = dict(
        model_name_or_path=model_path,
        cache_dir=user_config.get("cache_dir", None),
        finetuning_type=finetuning_type,
        template=template,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id or None,
        export_size=export_size,
        export_quantization_bit=int(export_quantization_bit) if export_quantization_bit in GPTQ_BITS else None,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        trust_remote_code=True,
    )
    args.update(json.loads(extra_args))

    if checkpoint_path:
        if finetuning_type in PEFT_METHODS:  # list
            args["adapter_name_or_path"] = ",".join(
                [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
            )
        else:  # str
            args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

    yield ALERTS["info_export_local"][lang]
    try:
        export_model(args)
        torch_gc()
    except Exception as e:
        msg = ALERTS["err_export_local_failed"][lang] + f"\n\n```\n{e}\n```"
        gr.Warning(msg)
        if is_s3_export:
            try:
                shutil.rmtree(export_dir)
            except Exception:
                pass
        yield msg
        return
    if is_s3_export:
        try:
            yield ALERTS["info_export_uploading"][lang] + upload_target
            _upload_directory_to_s3(export_dir, upload_target)
            yield ALERTS["info_export_uploaded"][lang] + upload_target
        except ModuleNotFoundError:
            msg = ALERTS["err_export_upload_not_found"][lang]
            gr.Warning(msg)
            yield msg
            return
        except Exception as e:
            msg = ALERTS["err_export_upload_failed"][lang] + f"\n\n```\n{e}\n```"
            gr.Warning(msg)
            yield msg
            return
        finally:
            try:
                shutil.rmtree(export_dir)
            except Exception:
                pass

    yield ALERTS["info_exported"][lang]


def create_export_tab(engine: "Engine") -> dict[str, "Component"]:
    with gr.Row():
        export_size = gr.Slider(minimum=1, maximum=100, value=5, step=1)
        export_quantization_bit = gr.Dropdown(choices=["none"] + GPTQ_BITS, value="none")
        export_quantization_dataset = gr.Textbox(value="data/c4_demo.jsonl")
        export_device = gr.Radio(choices=["cpu", "auto"], value="cpu")
        export_legacy_format = gr.Checkbox()

    with gr.Row():
        export_dir = gr.Textbox()
        export_hub_model_id = gr.Textbox()
        extra_args = gr.Textbox(value="{}")

    checkpoint_path: gr.Dropdown = engine.manager.get_elem_by_id("top.checkpoint_path")
    checkpoint_path.change(can_quantize, [checkpoint_path], [export_quantization_bit], queue=False)

    export_btn = gr.Button()
    info_box = gr.Markdown(show_label=False, elem_classes=["export-status"])

    export_btn.click(
        save_model,
        [
            engine.manager.get_elem_by_id("top.lang"),
            engine.manager.get_elem_by_id("top.model_name"),
            engine.manager.get_elem_by_id("top.hub_name"),
            engine.manager.get_elem_by_id("top.model_path"),
            engine.manager.get_elem_by_id("top.finetuning_type"),
            engine.manager.get_elem_by_id("top.checkpoint_path"),
            engine.manager.get_elem_by_id("top.template"),
            export_size,
            export_quantization_bit,
            export_quantization_dataset,
            export_device,
            export_legacy_format,
            export_dir,
            export_hub_model_id,
            extra_args,
        ],
        [info_box],
    )

    return dict(
        export_size=export_size,
        export_quantization_bit=export_quantization_bit,
        export_quantization_dataset=export_quantization_dataset,
        export_device=export_device,
        export_legacy_format=export_legacy_format,
        export_dir=export_dir,
        export_hub_model_id=export_hub_model_id,
        extra_args=extra_args,
        export_btn=export_btn,
        info_box=info_box,
    )

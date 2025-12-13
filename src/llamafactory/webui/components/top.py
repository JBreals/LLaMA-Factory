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

from typing import TYPE_CHECKING

from ...data import TEMPLATES
from ...extras.constants import METHODS, SUPPORTED_MODELS
from ...extras.misc import use_modelscope, use_openmind
from ...extras.packages import is_gradio_available
from ..common import save_config
from ..control import (
    can_quantize,
    can_quantize_to,
    check_template,
    get_model_info,
    list_checkpoints,
    switch_hub,
    suggest_model_name,
)


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


def create_top() -> dict[str, "Component"]:
    with gr.Row():
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], value=None, scale=1)
        available_models = list(SUPPORTED_MODELS.keys()) + ["Custom"]
        model_name = gr.Dropdown(choices=available_models, value=None, scale=2, allow_custom_value=True)
        model_name_text = gr.Textbox(
            scale=2, placeholder="Model name (s3에서 비우면 경로 끝 이름으로 자동 설정)", visible=False
        )
        model_path = gr.Textbox(scale=2)
        default_hub = "modelscope" if use_modelscope() else "openmind" if use_openmind() else "huggingface"
        hub_name = gr.Dropdown(choices=["huggingface", "modelscope", "openmind", "s3"], value=default_hub, scale=2)
        data_source = gr.Dropdown(choices=["local", "huggingface", "s3"], value="local", scale=1)

    with gr.Row():
        finetuning_type = gr.Dropdown(choices=METHODS, value="lora", scale=1)
        checkpoint_path = gr.Dropdown(multiselect=True, allow_custom_value=True, scale=6)

    with gr.Row():
        quantization_bit = gr.Dropdown(choices=["none", "8", "4"], value="none", allow_custom_value=True)
        quantization_method = gr.Dropdown(choices=["bnb", "hqq", "eetq"], value="bnb")
        template = gr.Dropdown(choices=list(TEMPLATES.keys()), value="default")
        rope_scaling = gr.Dropdown(choices=["none", "linear", "dynamic", "yarn", "llama3"], value="none")
        booster = gr.Dropdown(choices=["auto", "flashattn2", "unsloth", "liger_kernel"], value="auto")

    def _fill_model_info(name: str | None, hub: str, cur_path: str | None, cur_tmpl: str | None):
        # On s3/custom, keep current path/template; otherwise fill from mapping
        if hub == "s3" or name not in available_models:
            return gr.Textbox(value=cur_path), gr.Dropdown(value=cur_tmpl)
        path, tmpl = get_model_info(name)
        return gr.Textbox(value=path), gr.Dropdown(value=tmpl)

    model_name.change(_fill_model_info, [model_name, hub_name, model_path, template], [model_path, template], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    ).then(check_template, [lang, template])
    model_name.input(save_config, inputs=[lang, hub_name, model_name, model_path, data_source], queue=False)
    model_name_text.input(save_config, inputs=[lang, hub_name, model_name_text, model_path, data_source], queue=False)
    model_path.input(save_config, inputs=[lang, hub_name, model_name, model_path, data_source], queue=False)
    model_path.change(
        suggest_model_name, [hub_name, model_path, model_name_text, model_name], [model_name_text], queue=False
    )
    finetuning_type.change(can_quantize, [finetuning_type], [quantization_bit], queue=False).then(
        list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False
    )
    checkpoint_path.focus(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False)
    quantization_method.change(can_quantize_to, [quantization_method], [quantization_bit], queue=False)
    hub_name.change(switch_hub, inputs=[hub_name], queue=False).then(
        get_model_info, [model_name], [model_path, template], queue=False
    ).then(list_checkpoints, [model_name, finetuning_type], [checkpoint_path], queue=False).then(
        check_template, [lang, template]
    ).then(
        suggest_model_name, [hub_name, model_path, model_name_text, model_name], [model_name_text], queue=False
    )
    hub_name.input(save_config, inputs=[lang, hub_name, model_name, model_path, data_source], queue=False)
    data_source.input(save_config, inputs=[lang, hub_name, model_name, model_path, data_source], queue=False)
    def _toggle_model_input(h: str, n_sel: str | None, n_txt: str | None, m_path: str | None):
        if h == "s3":
            leaf = ""
            if not n_txt and m_path:
                cleaned = m_path.rstrip("/")
                leaf = cleaned.split("/")[-1] if cleaned else ""
            txt_val = n_txt or leaf or (n_sel if n_sel not in available_models else "")
            return (gr.Dropdown(visible=False, value=None), gr.Textbox(visible=True, value=txt_val))
        safe_val = (
            n_sel if n_sel in available_models else ("Custom" if "Custom" in available_models else available_models[0])
        )
        return (gr.Dropdown(visible=True, value=safe_val), gr.Textbox(visible=False, value=n_txt))

    hub_name.change(
        _toggle_model_input, [hub_name, model_name, model_name_text, model_path], [model_name, model_name_text], queue=False
    )
    model_name_text.change(lambda x: x, [model_name_text], [model_name], queue=False)

    return dict(
        lang=lang,
        model_name=model_name,
        model_name_text=model_name_text,
        model_path=model_path,
        hub_name=hub_name,
        data_source=data_source,
        finetuning_type=finetuning_type,
        checkpoint_path=checkpoint_path,
        quantization_bit=quantization_bit,
        quantization_method=quantization_method,
        template=template,
        rope_scaling=rope_scaling,
        booster=booster,
    )

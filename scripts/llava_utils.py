import re

import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.eval.run_llava import load_images
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)


def get_prompt(
    query: str,
    model,
    model_name,
    system_prompt: str,
    explicit_conv_mode: str | None = None,
):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if explicit_conv_mode is not None and explicit_conv_mode != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, \
            while `--conv-mode` is {}, using {}".format(
                conv_mode, explicit_conv_mode, explicit_conv_mode
            )
        )
    else:
        explicit_conv_mode = conv_mode

    conv = conv_templates[explicit_conv_mode].copy()

    conv.system = system_prompt

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()


def process_image(image_file, image_processor, model):
    images = load_images([image_file])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    return (image_sizes, images_tensor)


def get_input_ids(prompt, tokenizer):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    return input_ids

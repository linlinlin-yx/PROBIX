import json
import os

import requests
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

_LOCAL_MISTRAL_MODEL = None
_LOCAL_MISTRAL_TOKENIZER = None
_LOCAL_MISTRAL_MODEL_PATH = None

DEFAULT_PREDICTION_TARGETS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {"backend": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct"},
    "mistralai/Mistral-7B-Instruct-v0.3": {"backend": "openrouter", "model": "mistralai/mistral-7b-instruct-v0.3"},
    "gpt4": {"backend": "openai", "model": "gpt-4"},
    "gpt-4": {"backend": "openai", "model": "gpt-4"},
}


def resolve_prediction_target(base_model, transfer_mode="off", target_model=None, hf_local_path=None):
    if transfer_mode not in {"on", "off"}:
        raise ValueError(f"Unsupported transfer_mode: {transfer_mode}")

    if transfer_mode == "on":
        if not target_model:
            raise ValueError("`--target_model` is required when `--transfer_mode on`.")
        requested_model = target_model
    else:
        requested_model = target_model or base_model
        if hf_local_path is not None:
            if requested_model != "mistralai/Mistral-7B-Instruct-v0.3":
                raise ValueError("`--hf_local_path` is only supported for `mistralai/Mistral-7B-Instruct-v0.3`.")
            return {
                "backend": "local_mistral",
                "model": hf_local_path,
                "requested_model": requested_model,
            }

    if requested_model in DEFAULT_PREDICTION_TARGETS:
        target = DEFAULT_PREDICTION_TARGETS[requested_model].copy()
        target["requested_model"] = requested_model
        return target

    if ":" in requested_model:
        backend, routed_model = requested_model.split(":", 1)
        if backend in {"openrouter", "openai"} and routed_model:
            return {
                "backend": backend,
                "model": routed_model,
                "requested_model": requested_model,
            }

    raise ValueError(
        f"Unsupported target model `{requested_model}`. Use a known model name or an explicit `openrouter:<model_id>` / `openai:<model_id>` route."
    )


def ensure_prediction_target_available(base_model, transfer_mode="off", target_model=None, hf_local_path=None):
    target = resolve_prediction_target(
        base_model,
        transfer_mode=transfer_mode,
        target_model=target_model,
        hf_local_path=hf_local_path,
    )

    if target["backend"] == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for the selected OpenRouter target route.")
    if target["backend"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the selected OpenAI target route.")
    if target["backend"] == "local_mistral":
        if not os.path.exists(target["model"]):
            raise RuntimeError(f"Local Mistral path `{target['model']}` does not exist.")
        tokenizer_file = os.path.join(target["model"], "tokenizer.model.v3")
        if not os.path.exists(tokenizer_file):
            raise RuntimeError(
                f"Could not find `{tokenizer_file}`. The local Mistral directory must include `tokenizer.model.v3`."
            )
        try:
            load_local_mistral_model(target["model"])
        except Exception as error:
            raise RuntimeError(
                f"Failed to load the local Mistral model from `{target['model']}`: {error}"
            ) from error

    return target



def get_prediction_call_model(base_model, text, dataset="AG-News", transfer_mode="off", target_model=None, hf_local_path=None):
    target = resolve_prediction_target(
        base_model,
        transfer_mode=transfer_mode,
        target_model=target_model,
        hf_local_path=hf_local_path,
    )
    print(
        f"prediction_target: requested={target['requested_model']} backend={target['backend']} model={target['model']} dataset={dataset}"
    )

    if dataset.lower() == "ag-news":
        prompt = f"Classify the following news text into one of these categories: [0: World, 1: Sports, 2: Business, 3: Sci/Tech]. Return only the number. Text: {text}"
    elif dataset.lower() == "sst2":
        prompt = f"Classify the sentiment of the movie review as positive (1) or negative (0). Return only a single digit (0 or 1), making a best-effort judgment based on available words if incomplete or ambiguous. Text: {text}"
    elif dataset.lower() == "strategyqa":
        prompt = f"Answer the following question with True or False based on reasoning. Return only True or False. Question: {text}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if target["backend"] == "openai":
        return get_prediction_call_online_model(prompt, dataset, model=target["model"])
    if target["backend"] == "openrouter":
        return get_prediction_openrouter(prompt, dataset, model=target["model"])
    if target["backend"] == "local_mistral":
        return get_prediction_call_local_mistral_model(prompt, dataset, model_path=target["model"])

    raise ValueError(f"Unsupported backend: {target['backend']}")


def get_prediction_call_online_model(prompt, dataset="AG-News", model="gpt-4"):
    max_retries = 3
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.3,
            top_p=0.5,
        )

        decoded_text = response.choices[0].message.content.strip()
        print(f"Raw output (attempt {attempt + 1}): {decoded_text}")

        label = parse_label_from_response(decoded_text, dataset)
        if label is not None:
            return label

    print(f"Skipping sample after {max_retries} failed attempts.")
    return None


def get_prediction_openrouter(prompt, dataset, model):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            print(f"Raw output (attempt {attempt + 1}): {result}")
            label = parse_label_from_response(result, dataset)
            if label is not None:
                return label
        except requests.RequestException as error:
            print(f"API call failed: {error}")
        except ValueError as error:
            print(f"Failed to parse prediction: {error}")

    return None


def load_local_mistral_model(model_path):
    global _LOCAL_MISTRAL_MODEL, _LOCAL_MISTRAL_TOKENIZER, _LOCAL_MISTRAL_MODEL_PATH
    if _LOCAL_MISTRAL_MODEL is not None and _LOCAL_MISTRAL_MODEL_PATH == model_path:
        return _LOCAL_MISTRAL_MODEL, _LOCAL_MISTRAL_TOKENIZER

    from mistral_inference.transformer import Transformer
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_file(os.path.join(model_path, "tokenizer.model.v3"))
    model = Transformer.from_folder(model_path)

    _LOCAL_MISTRAL_MODEL = model
    _LOCAL_MISTRAL_TOKENIZER = tokenizer
    _LOCAL_MISTRAL_MODEL_PATH = model_path
    return _LOCAL_MISTRAL_MODEL, _LOCAL_MISTRAL_TOKENIZER


def get_prediction_call_local_mistral_model(prompt, dataset, model_path):
    from mistral_inference.generate import generate
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    model, tokenizer = load_local_mistral_model(model_path)
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate(
        [tokens],
        model,
        max_tokens=16,
        temperature=0.0,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0]).strip()
    print(f"Local Mistral raw Parsed Result: {result}")
    return parse_label_from_response(result, dataset)


def parse_label_from_response(result, dataset):
    if not result:
        raise ValueError("No valid response from API")
    digits = "".join(filter(str.isdigit, result))
    if dataset.lower() in ["ag-news", "sst2"]:
        if digits:
            return int(digits[0])
        raise ValueError(f"No valid digit found in response: {result}")
    if dataset.lower() == "strategyqa":
        return 1 if "true" in result.lower() else 0
    raise ValueError(f"No valid digit found in response: {result}")

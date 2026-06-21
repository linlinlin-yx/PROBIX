import json
import os

import requests
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# In white-box mode, the target model defaults to the base model.
DEFAULT_PREDICTION_TARGETS = {
    "meta-llama/Meta-Llama-3-8B-Instruct": {"backend": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct"},
    "mistralai/Mistral-7B-Instruct-v0.3": {"backend": "ollama", "model": "mistral"},
    "gpt4": {"backend": "openai", "model": "gpt-4"},
    "gpt-4": {"backend": "openai", "model": "gpt-4"},
}


def resolve_prediction_target(base_model, transfer_mode="off", target_model=None):
    if transfer_mode not in {"on", "off"}:
        raise ValueError(f"Unsupported transfer_mode: {transfer_mode}")

    if transfer_mode == "on":
        if not target_model:
            raise ValueError("`--target_model` is required when `--transfer_mode on`.")
        requested_model = target_model
    else:
        requested_model = target_model or base_model

    if requested_model in DEFAULT_PREDICTION_TARGETS:
        target = DEFAULT_PREDICTION_TARGETS[requested_model].copy()
        target["requested_model"] = requested_model
        return target

    if ":" in requested_model:
        backend, routed_model = requested_model.split(":", 1)
        if backend in {"openrouter", "openai", "ollama"} and routed_model:
            return {
                "backend": backend,
                "model": routed_model,
                "requested_model": requested_model,
            }

    return {
        "backend": "ollama",
        "model": requested_model,
        "requested_model": requested_model,
    }


def ensure_prediction_target_available(base_model, transfer_mode="off", target_model=None):
    target = resolve_prediction_target(base_model, transfer_mode=transfer_mode, target_model=target_model)

    if target["backend"] == "ollama":
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                "Ollama is required for this target route, but http://localhost:11434 is not reachable. "
                f"Please start Ollama and make sure the `{target['model']}` model is available. Original error: {error}"
            ) from error

        data = response.json()
        models = data.get("models", [])
        available_names = {item.get("name", "") for item in models}
        available_models = {item.get("model", "") for item in models}
        acceptable_names = {
            target["model"],
            f"{target['model']}:latest",
        }
        if not ((available_names & acceptable_names) or (available_models & acceptable_names)):
            pretty_available = ", ".join(sorted(name for name in available_names if name)) or "none"
            raise RuntimeError(
                f"Prediction target `{target['model']}` was not found in local Ollama. "
                f"Available tags: {pretty_available}. Please run `ollama pull {target['model']}` or choose another `--target_model`."
            )

    elif target["backend"] == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for the selected OpenRouter target route.")
    elif target["backend"] == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the selected OpenAI target route.")

    return target



def get_prediction_call_model(base_model, text, dataset="AG-News", transfer_mode="off", target_model=None):
    target = resolve_prediction_target(base_model, transfer_mode=transfer_mode, target_model=target_model)
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
    if target["backend"] == "ollama":
        return get_prediction_call_ollama_model(prompt, dataset, model=target["model"])

    raise ValueError(f"Unsupported backend: {target['backend']}")


def get_prediction_call_ollama_model(prompt, dataset, model, api_base="http://localhost:11434"):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "max_new_tokens": 5,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{api_base}/api/generate", json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        result = data.get("response", data.get("text", ""))
        print(f"API raw Parsed Result: {result}")
        return parse_label_from_response(result, dataset)
    except requests.RequestException as error:
        print(f"API call failed: {error}")
        raise
    except ValueError as error:
        print(f"Failed to parse prediction: {error}")
        raise


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

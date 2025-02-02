import os
from pathlib import Path
from datetime import datetime
import json
import asyncio
from tqdm.asyncio import tqdm
import hydra
from dotenv import load_dotenv
import pandas as pd
from huggingface_hub import AsyncInferenceClient
import random
from itertools import chain
from omegaconf import OmegaConf
from openai import AsyncOpenAI

def response_to_json(result):

    if isinstance(result, str):
        return {"result": result}

    r = result.__dict__
    u = result.usage.__dict__
    c = result.choices[0].__dict__
    m = result.choices[0].message.__dict__

    r["usage"] = u
    c["message"] = m
    r["choices"] = [c]

    return r

async def call_api(client, text, model_name, parameters=None):
    if parameters is None:
        parameters = {}

    messages = [{"role": "user", "content": text}]

    # if any(["jamba" in model_name.lower()]):
    #     from ai21.models.chat import UserMessage
    #     messages = [
    #         UserMessage(content=text)
    #     ]
    try:
        result = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            **parameters,
        )
    except Exception as e:
        print(model_name)
        print(e)
        print(text)
        result = "Request failed: " + str(e)

    return response_to_json(result)

def decide_client(model_name):
    if "mini" in model_name.lower():
        return "hf"
    if "llama" in model_name.lower():
        return "together"
    elif "deepseek" in model_name.lower():
        return "deepseek"
    elif "minimax" in model_name.lower():
        return "minimax"
    elif "jamba" in model_name.lower():
        return "ai21"
    else:
        raise ValueError(f"Unknown model: {model_name}")


async def make_parallel_requests(
    clients, texts, model_names, ids, output_dir, max_concurrent=10, parameters=None, alternate_mapping=None
):

    semaphore = asyncio.Semaphore(max_concurrent)


    async def single_request(text, model_name, id_):
        original_model_name = model_name


        client = clients[decide_client(model_name)]

        if alternate_mapping is not None and model_name in alternate_mapping:
            model_name = alternate_mapping[model_name]
        try:
            async with semaphore:
                try:
                    result = await call_api(
                        client,
                        text,
                        model_name,
                        parameters,
                    )
                except Exception as e:
                    result = f"Request failed: {e}"
                    print(f"({model_name}) Request failed: {e}")

                with open(
                    Path(output_dir)
                    / f"{datetime.now().isoformat()}_{id_}_{original_model_name.replace('/', '_')}.json",
                    "w",
                ) as f:
                    json.dump(
                        {
                            "text": text,
                            "response": result,
                            "id": id_,
                            "model": original_model_name,
                        },
                        f,
                    )

                return result
        except Exception as e:
            print(e)
            print(model_name)
            return "Request failed: " + str(e)

    tasks = [single_request(t, m, i) for t, m, i in zip(texts, model_names, ids)]
    results = await tqdm.gather(*tasks, desc="Processing requests")

    return results


def find_previous_samples(j_dir):

    j_files = list(Path(j_dir).glob("*.json"))

    successful = set()
    unsuccessful = set()
    for j_file in j_files:

        date_str, id_, *model_name = j_file.name.split("_")
        model_name = "/".join(model_name).rsplit(".", maxsplit=1)[0]

        id_ = id_.strip()

        data = json.load(open(j_file))
        if isinstance(data["response"], dict):
            successful.add((id_, model_name))
            continue
        if data["response"].startswith("Request failed:"):
            unsuccessful.add((id_, model_name))
        else:
            successful.add((id_, model_name))

    print(f"Num successful: {len(successful)}")
    print(f"Num unsuccessful: {len(unsuccessful)}")

    return {
        "successful": successful,
        "unsuccessful": unsuccessful,
    }


def find_length_limited_samples(j_dir):

    j_files = list(Path(j_dir).glob("*.json"))

    successful = set()
    unsuccessful = set()
    for j_file in j_files:

        date_str, id_, *model_name = j_file.name.split("_")
        model_name = "/".join(model_name).rsplit(".", maxsplit=1)[0]

        id_ = id_.strip()

        data = json.load(open(j_file))
        if isinstance(data["response"], dict):
            if data["response"]["choices"][0]["finish_reason"] == "length":
                unsuccessful.add((id_, model_name))
            else:
                successful.add((id_, model_name))

    to_process = list(unsuccessful - successful)
    print("Num length limited samples:", len(to_process))

    return to_process


def load_json_files(j_files):
    texts, model_names, ids = [], [], []
    for j_file in j_files:
        data = json.load(open(j_file))
        texts.append(data["text"])
        model_names.append(data["model"])
        ids.append(data["id"])

    return texts, model_names, ids


@hydra.main(config_path="conf", version_base=None)
def main(cfg):

    loaded = load_dotenv("../.env", override=True)
    if not loaded:
        raise ValueError("Failed to load .env file")
    
    # set seed for reproducibility
    random.seed(cfg.seed)

    json_output_dir = Path(os.environ["PROJECT_DIR"]) / cfg.json_output_dir
    json_output_dir.mkdir(parents=True, exist_ok=True)


    clients = {
        "hf": AsyncInferenceClient(token=os.environ[cfg.hf_inference_api_key_env_var]),
        "together": AsyncOpenAI(api_key=os.environ[cfg.together_api_key_env_var], base_url="https://api.together.xyz/v1"),
        "deepseek": AsyncOpenAI(api_key=os.environ[cfg.deepseek_api_key_env_var], base_url="https://api.deepseek.com"),
        "minimax": AsyncOpenAI(api_key=os.environ[cfg.minimax_api_key_env_var], base_url="https://api.minimaxi.chat/v1"),
        "ai21": AsyncOpenAI(api_key=os.environ[cfg.ai21_api_key_env_var], base_url="https://api.ai21.com/studio/v1"),
    }

    df = pd.read_parquet(Path(os.environ["PROJECT_DIR"]) / cfg.data_file)

    temp_df = pd.DataFrame({"text": df.text, "model": model_names, "id": range(len(df)), "response": df.response})

    if not (json_output_dir / "data.parquet").exists():
        temp_df.to_parquet(json_output_dir / "data.parquet")
    else:
        temp_df = pd.read_parquet(json_output_dir / "data.parquet")

    previous_samples = find_previous_samples(json_output_dir)

    if len(previous_samples["successful"]) > 0:
        successful_ids, successful_models = zip(*previous_samples["successful"])
        successful_df = pd.DataFrame({"model": successful_models, "id": successful_ids})

        successful_df["model-id"] = successful_df["model"] + "-" + successful_df["id"]
        temp_df["model-id"] = temp_df["model"] + "-" + temp_df["id"]

        # ignore succesful runs
        temp_df = temp_df[~(temp_df["model-id"].isin(successful_df["model-id"]))]


    if cfg.retry_failed:
        
        failed_ids, failed_models = zip(*previous_samples["unsuccessful"])
        prev_failed_df = pd.DataFrame({"model": failed_models, "id": failed_ids})
        prev_failed_df["model-id"] = prev_failed_df["model"] + "-" + prev_failed_df["id"]

        temp_df = temp_df[temp_df["model-id"].isin(prev_failed_df["model-id"])]

    elif cfg.retry_length_limited:
        length_limited_samples = find_length_limited_samples(json_output_dir)
        length_limited_ids, length_limited_models = zip(*length_limited_samples)
        length_limited_df = pd.DataFrame({"model": length_limited_models, "id": length_limited_ids})
        length_limited_df["model-id"] = length_limited_df["model"] + "-" + length_limited_df["id"]

        temp_df = temp_df[temp_df["model-id"].isin(length_limited_df["model-id"])]

        cfg.parameters.max_tokens = 4000

    
    print(f"Num samples to process: {len(temp_df)}")

    texts = temp_df.text.tolist()
    ids = temp_df.id.tolist()
    model_names = temp_df.model.tolist()

    if cfg.debug:
        texts = texts[:10]
        ids = ids[:10]
        model_names = model_names[:10]

    _ = asyncio.run(
            make_parallel_requests(
                clients=clients,
                texts=texts,
                model_names=model_names,
                ids=ids,
                output_dir=json_output_dir,
                max_concurrent=cfg.max_concurrent,
                parameters=OmegaConf.to_container(cfg.parameters),
                alternate_mapping=cfg.alternate_mapping,
            )
        )


if __name__ == "__main__":
    main()
from argparse import ArgumentParser
import json
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
import logging

import pandas as pd

import jinja2 as j2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TranslatedJSON(BaseModel):
    orig_instruction:str
    orig_criteria:str
    orig_score1_description:str
    orig_score2_description:str
    orig_score3_description:str
    orig_score4_description:str
    orig_score5_description:str
    orig_response:str
    orig_reference_answer:str
    orig_feedback:str
    orig_score:int


def load_template(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        return j2.Template(f.read().strip())


def load_dataset_slice(dataset_path, slice=slice(0, None, 1)):
    df= pd.read_json(dataset_path)
    df=df[df.columns[2:-1]]

    aux=[]
    for index, row in df[slice].iterrows():
        aux.append(json.dumps(row.to_dict()))

    df={"instance":aux}
    return df


def batch_generator(dataset, batch_size=1):
    keys = list(dataset.keys())
    values = list(zip(*dataset.values()))
    for i in range(0, len(values), batch_size):
        yield [
            {key: value for key, value in zip(keys, _values)}
            for _values in values[i : i + batch_size]
        ]

def postprocess_output(output):
    try:
        conversation = json.loads(output)
        return conversation
    except json.JSONDecodeError:
        ...

    print("Not loading json",output)
    # Try to fix the output by finding the last bracket and removing everything after it
    last_bracket = output.rfind("}")

    try:
        conversation = json.loads(output[: last_bracket + 1])
        return conversation
    except json.JSONDecodeError as e:
        logger.error("Failed to fix output")
        logger.error(e)
        logger.error(output)

        return []


def main(args):
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop="\n```",
        guided_decoding=GuidedDecodingParams(json=TranslatedJSON.model_json_schema()),
        frequency_penalty=args.frequency_penalty,
    )

    # Prompt
    if  args.target_lang not in ["en","eu","es"]:
        print("LANG NOT FOUND. DEFAULT TO BASQUE")
        args.target_lang ="eu"
    
    # Load few-shot examples
    with open(f"./translate/translation_prompts/few_shot_{args.few_shot_folder}/{args.target_lang}_examples.txt", "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    ex1_target_lang, ex2_target_lang = lines

    with open(f"./translate/translation_prompts/few_shot_{args.few_shot_folder}/{args.source_lang}_examples.txt", "r", encoding="utf-8") as f:
        source_lines = [line.rstrip("\n") for line in f]
    ex1_source_lang, ex2_source_lang = source_lines

    #system_prompt
    lang_dict = {"en": "English", "eu": "Basque", "es": "Spanish"}
    target_lang = lang_dict[args.target_lang]
    source_lang = lang_dict[args.source_lang]
    system_prompt = load_template(args.prompt_path).render(source_lang=source_lang, target_lang=target_lang)

    output_file_path = os.path.join(
        args.output_path,
        os.path.basename(args.dataset_path).split('.')[0]
        + f"_{args.target_lang}_{args.dataset_start}_{args.dataset_end}.jsonl",
    )

    if os.path.exists(output_file_path):
        with open(output_file_path, "rt") as f:
            progress = sum(1 for _ in f)
    else:
        progress = 0

    dataset = load_dataset_slice(
        args.dataset_path, slice(args.dataset_start + progress, args.dataset_end, 1)
    )

    os.makedirs(args.output_path, exist_ok=True)
    
    with open(output_file_path, "a") as f:
        for i, batch in enumerate(batch_generator(dataset, batch_size=args.batch_size)):
            
            prompts_fs = [[
                {"role": "system", "content": system_prompt},

                {"role": "user","content": ex1_source_lang},
                {"role": "assistant", "content": ex1_target_lang},

                {"role": "user", "content": ex2_source_lang},
                {"role": "assistant","content": ex2_target_lang},

                {"role": "user","content": example["instance"]},
                ] for example in batch]
            
            outputs = [
                output.outputs[0].text.strip()
                for output in llm.chat(
                    prompts_fs,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )
            ]

            for original, output in zip(batch, outputs):
                output_dict = postprocess_output(output)
                try:
                    print(json.dumps(output_dict, ensure_ascii=False), file=f)
                except UnicodeEncodeError: # Emojis raise this error
                    logger.error("Failed to encode output")
                    logger.error(output)
            
            logger.warning(f"Processed {progress + i * args.batch_size} examples")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="HiTZ/Latxa-Llama-3.1-70B-Instruct",
        help="The HF model id.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="./translate/translation_prompts/system_prompt.j2",
        help="The prompt.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The dataset path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/",
        help="The output path.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="eu",
        help="Target language.",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        default="en",
        help="Source language.",
    )
    parser.add_argument(
        "--few_shot_folder",
        type=str,
        default="feedback_collection",
        help="Few shot instance source. Options are feedback_collection, recon and flask.",
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()

    main(args)

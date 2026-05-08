import os
import re
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def check_output(output,reg_pattern):
    """Check if the output matches the expected pattern."""

    pattern = re.compile(reg_pattern,re.DOTALL)
    invalid_indices =[]
    for i, resp in enumerate(output):
        if not pattern.search(resp):
            invalid_indices.append(i)
    return invalid_indices

def main(args):
    
    # Scoring pattern
    if args.benchmark in ["recon","flask"]:
        reg_pattern = r".{90,}\[(RESULT|EMAITZA|RESULTADO)\]\s*[1-5]"
    else:
        print("Benchmark not among the options")
    
    # Initialize LLM 
    llm = LLM(
        model=args.model_path,
        dtype=args.dtype,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        stop=None,
        guided_decoding=GuidedDecodingParams(regex=reg_pattern),
        frequency_penalty=args.frequency_penalty,
    )

    # Per language loop
    for lang in args.lang_list.split(","):
        print("Evaluating model ", args.model_path, "in ", lang)
    
        # Read benchmark file and prepare inputs
        bench_file=f"{args.dataset_path}{args.benchmark}/{args.benchmark}_{lang}.jsonl"
        print("File: ",bench_file)
        dataset=pd.read_json(bench_file,lines=True)
        input=dataset["inputs"].tolist()
        d={"inputs":input}

        # Create output file and folder
        os.makedirs(args.output_path+args.benchmark, exist_ok=True)
        output_file_path = args.output_path+args.benchmark+"/"+os.path.basename(args.model_path)+"_"+args.benchmark+ f"_{lang}.jsonl"
        
        print("Generating outputs...")

        # Per repetition
        for i in range(args.repetitions):
            print(f"Iteration {i}")
            out = []
            
            # Regenerate if the pattern does not match the regex
            regen_limit = args.regen_limit

            final_outputs = [None] * len(input)  # Placeholder for full output
            pending_batch = input.copy()
            pending_indices = list(range(len(input)))

            prev = len(pending_indices)  # Previous number of invalid indices


            while pending_indices:

                # Generate outputs
                outputs = llm.generate(pending_batch, sampling_params, use_tqdm=True)
                texts = [output.outputs[0].text.strip() for output in outputs]

                # Check if valid
                invalid_indices = check_output(texts,reg_pattern)

                # Insert valid outputs into final_outputs
                for local_idx, text in enumerate(texts):
                    if local_idx not in invalid_indices:
                        original_idx = pending_indices[local_idx]
                        final_outputs[original_idx] = text

                if invalid_indices:
                    print(f"{len(invalid_indices)} invalid outputs found.")
                    print("Sample invalid output: ", texts[invalid_indices[0]])
                    print("Regenerating...")

                    if len(invalid_indices) == prev:
                        regen_limit-=1
                    else:
                        regen_limit = args.regen_limit
                    
                    if regen_limit <= 0:  # Mark invalid indices with [FORMAT ERROR] at the beginning
                        print("Regeneration limit reached. Marking invalid outputs with [FORMAT ERROR]...")
                        for local_idx, text in enumerate(texts):
                            if local_idx in invalid_indices:
                                original_idx = pending_indices[local_idx]
                                final_outputs[original_idx] = "[FORMAT ERROR] "+text
                        break

                    prev=len(invalid_indices)
                    
                    # Prepare next round with invalid instances
                    pending_batch = [pending_batch[i] for i in invalid_indices]
                    pending_indices = [pending_indices[i] for i in invalid_indices]
                else:
                    print("Generation ended")
                    break
            
            # Save "i"th outputs
            out+=final_outputs
            d[f"outputs_{i}"]=out
            pd.DataFrame(d).to_json(output_file_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="HiTZ/Latxa-Llama-3.1-8B-Instruct",
        help="The HF model id.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./evaluation/benchmarks/",
        help="The dataset path.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="recon",
        help="Benchmark name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./evaluation/evaluation_results/",
        help="The output path.",
    )
    parser.add_argument(
        "--lang_list",
        type=str,
        default="en",
        help="Target language.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--regen_limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=8100)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)

    args = parser.parse_args()

    main(args)
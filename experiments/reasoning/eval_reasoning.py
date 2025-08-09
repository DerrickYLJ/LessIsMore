import argparse
import json
import os
import logging
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import load as load_sparse_attention_model_internal
import extraction.sg_lang_utils.answer_extraction as answer_extraction
from extraction.utils.grader import check_is_correct

from experiments.reasoning.utils import load_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task_max_problems = {
    "aime": 30,
    "aime2025": 30,
    "math": 500,
    "gpqa": 198,
    "livecodebench": 100,
}

task_dataset = {
    "aime": "aime",
    "aime2025": "aime2025",
    "math": "math500",
    "gpqa": "gpqa",
    "livecodebench": "livecodebench",
}


def load_model_and_tokenizer(args, is_sparse_attention: bool):
    if is_sparse_attention:
        logger.info(
            f"Loading sparse attention model: {args.model_name} with attn_type: {args.attn_type}"
        )
        # Assuming load_sparse_attention_model_internal handles its own offline/cache logic or doesn't need internet
        return load_sparse_attention_model_internal(
            args.model_name,
            attn_type=args.attn_type,
            top_k=args.top_k,
            sparse_layer_start=args.sparse_layer_start,
            correction_layer=args.correction_layer,
            attention_sink=args.attention_sink,
            lim_ratio=args.lim_ratio,
        )


def load_2024_dataset(args, local_files_only: bool = False) -> List[dict]:
    logger.info(
        f"Loading {args.task} dataset {task_dataset[args.task]} (local_files_only={local_files_only})"
    )

    data_file = f"experiments/reasoning/data/{task_dataset[args.task]}/test.jsonl"

    examples = list(load_jsonl(data_file))

    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def inference_single(
    task,
    model,
    tokenizer,
    problem_prompt: str,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> str:

    system_prompt = (
        "Please reason step by step, and put your final answer within \boxed{}."
        if task in ["aime", "aime2025", "math"]
        else ""
    )

    messages = [
        {
            "role": "user",
            "content": f"{problem_prompt}\n" + system_prompt,
        }
    ]

    # Use the tokenizer's chat template
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_tensor = tokenizer(
        full_prompt, return_tensors="pt", return_attention_mask=False
    ).to(model.device)

    outputs = model.generate(
        **input_tensor,
        max_new_tokens=max_tokens,
        top_p=0.95,
        temperature=temperature,
    )
    new_tokens = outputs[0, input_tensor["input_ids"].shape[-1] :]

    print(f"=========Generated {len(new_tokens)} tokens.=========")

    out_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    logger.debug(f"Generated {len(new_tokens)} tokens. (temp={temperature}")

    return out_text.strip()


def load_existing_results_new_format(
    filename: str, n_attempts: int
) -> Dict[str, Dict[str, Dict]]:
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            for i in range(n_attempts):
                attempt_key = f"attempt_{i+1}"
                if attempt_key not in data:
                    data[attempt_key] = {}
            return data
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {filename}. Starting fresh.")
    return {f"attempt_{i+1}": {} for i in range(n_attempts)}


def save_overall_results(filename: str, data: Dict):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def main(args):
    os.makedirs("results", exist_ok=True)
    model_name_slug = args.model_name.replace("/", "_")
    is_sparse_attention_attn = (args.attn_type == "tidal" or args.attn_type == "lim")

    problem_range_slug = "_range-all"

    if is_sparse_attention_attn:
        top_k_str = f"_topk{args.top_k}" if args.top_k else ""
        results_file = f"results/{args.task}_{args.attn_type}_{top_k_str}_{model_name_slug}_temp{args.temperature}_sink{args.attention_sink}_sf{args.lim_ratio}_n{args.n}{problem_range_slug}.json"
    else:
        attn_str = args.attn_type if args.attn_type else "standard"
        results_file = f"results/{args.task}_{attn_str}_{model_name_slug}_temp{args.temperature}_n{args.n}{problem_range_slug}.json"

    logger.info(f"Results will be saved to: {results_file}")

    is_offline_datasets = (
        os.getenv("HF_DATASETS_OFFLINE", "0") == "1"
        or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    )

    dataset_loaded = load_2024_dataset(args, local_files_only=is_offline_datasets)
    if not dataset_loaded:
        logger.error("Failed to load dataset. Exiting.")
        return

    dataset_as_list_full = list(dataset_loaded)

    dataset_to_process_this_run = dataset_as_list_full[:]
    original_dataset_size = len(dataset_as_list_full)
    print("Original dataset size:", original_dataset_size)

    logger.info(f"Final results will be saved to: {results_file}")

    if not dataset_to_process_this_run:
        logger.info(
            "No problems selected for processing in this run. Exiting evaluation part."
        )
        return

    model, tokenizer = load_model_and_tokenizer(args, is_sparse_attention=is_sparse_attention_attn)

    overall_results_data = load_existing_results_new_format(results_file, args.n)

    logger.info(
        f"Loaded {len(dataset_to_process_this_run)} problems for this run. Will process up to {args.n} attempts for each, skipping already completed (problem, attempt) pairs from file '{results_file}'."
    )

    for attempt_num_one_based in range(1, args.n + 1):
        for item_data in tqdm(
            dataset_to_process_this_run, desc="Evaluating problems"
        ):
            problem_id_str = str(int(item_data["idx"]))
            problem_text = (
                item_data["problem"] if args.task != "gpqa" else item_data["question"]
            )
            parsed_correct_answer = (
                int(item_data["answer"])
                if args.task in {"aime", "aime2025"}
                else item_data["answer"]
            )

            attempt_key = f"attempt_{attempt_num_one_based}"
            if attempt_key not in overall_results_data:
                overall_results_data[attempt_key] = {}
            if problem_id_str in overall_results_data[attempt_key]:
                logger.debug(
                    f"Skipping Problem {problem_id_str} Attempt {attempt_num_one_based} as it exists in {results_file}."
                )
                continue
            response_text = inference_single(
                args.task,
                model,
                tokenizer,
                problem_text,
                args.max_gen_len,
                args.temperature,
            )
            predicted_answer = answer_extraction.extract_math_answer(
                        problem_text, response_text, "limo"
                    )
            if isinstance(predicted_answer, list) and len(predicted_answer) > 0:
                predicted_answer = predicted_answer[-1]
            elif isinstance(predicted_answer, list) and len(predicted_answer) == 0:
                predicted_answer = ""
            gt_answer = str(parsed_correct_answer)
            is_correct_this_attempt = check_is_correct(predicted_answer, gt_answer)
            overall_results_data[attempt_key][problem_id_str] = {
                "problem": problem_text,
                "response": response_text,
                "predicted_answer": predicted_answer,
                "correct_answer": parsed_correct_answer,
                "is_correct_this_attempt": is_correct_this_attempt,
            }

            save_overall_results(results_file, overall_results_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLMs on AIME 2024 problems")

    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default=None,
        help="Path to the locally saved dataset directory (e.g., output of dataset.save_to_disk).",
    )
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Path to the locally saved model/tokenizer directory (output of model.save_pretrained).",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or local path (if pre-downloaded and using offline mode).",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of attempts per problem (pass@n)."
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        default=None,
        help="Attention type ('lim', 'tidal', or None/other for standard).",
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=4096, help="Maximum new tokens per attempt."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy).",
    )
    parser.add_argument(
        "--top_k", type=int, default=128, help="Top-k for sparse attention decode."
    )
    parser.add_argument(
        "--sparse_layer_start",
        type=int,
        default=2,
        help="Layer for sparse attention.",
    )
    parser.add_argument(
        "--correction_layer", type=int, default=13, help="Correction layer."
    )
    parser.add_argument(
        "--attention_sink",
        type=int,
        default=0,
        help="Attention sinks.",
    )
    parser.add_argument(
        "--lim_ratio",
        type=float,
        default=1,
        help="Scale factor for token budget.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="aime",
        choices=["aime", "aime2025", "math", "gpqa", "livecodebench"],
        help="Task name for evaluation (default: 'aime2024').",
    )

    args = parser.parse_args()

    if args.attn_type and args.attn_type.lower() == "none":
        args.attn_type = None
    if args.attn_type and (args.attn_type.lower() != "tidal" or args.attn_type.lower() != "lim"):
        logger.info(
            f"attn_type '{args.attn_type}' is not 'lim' or 'tidal'. Using standard HuggingFace model evaluation logic."
        )

    main(args)

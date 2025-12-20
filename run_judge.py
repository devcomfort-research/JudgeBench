from typing import List, Dict, Any
import argparse
import asyncio
import json
import os
import random

import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.asyncio import tqdm_asyncio

import utils.file_operations as file_operations
import utils.judges as judges
import utils.metrics as metrics

console = Console()


async def judge_pairs(
    pairs: List[Dict[str, Any]],
    judge_name: str,
    judge_model: str,
    concurrency_limit: int = 1,
    reverse_order: bool = False,
    output_file: str = None,
    progress: Progress = None,
    global_semaphore: asyncio.Semaphore = None,
) -> List[Dict[str, Any]]:
    """Judge a list of answer pairs asynchronously using a specified judge.

    Parameters
    ----------
    pairs : List[Dict[str, Any]]
        A list of dictionaries, each representing a pair of responses to be judged.
    judge_name : str
        The name/type of the judge to use (e.g., 'arena_hard').
    judge_model : str
        The specific model identifier to be used by the judge.
    concurrency_limit : int, optional
        Maximum number of concurrent judgment tasks (default is 1).
    reverse_order : bool, optional
        Whether to judge each pair in both (A, B) and (B, A) orders (default is False).
    output_file : str, optional
        Path to a JSONL file where each completed judgment will be appended.
    progress : Progress, optional
        Rich Progress object to show progress.
    global_semaphore : asyncio.Semaphore, optional
        A shared semaphore to limit concurrency across all models.

    Returns
    -------
    List[Dict[str, Any]]
        The list of pairs updated with judgment results.
    """
    # Use global_semaphore if provided, otherwise create a local one
    semaphore = (
        global_semaphore if global_semaphore else asyncio.Semaphore(concurrency_limit)
    )
    judge = judges.get_judge_from_judge_name_and_model(judge_name, judge_model)
    file_lock = asyncio.Lock()

    async def judge_pair(pair: Dict[str, Any]):
        question = pair["question"]
        response_A = pair["response_A"]
        response_B = pair["response_B"]

        async def get_single_judgment(q, r1, r2, is_reversed=False):
            async with semaphore:
                try:
                    return await judge.get_judgment(q, r1, r2)
                except Exception as e:
                    order_str = " (reversed)" if is_reversed else ""
                    console.log(
                        f"[red]Failed to judge pair {pair['pair_id']}{order_str} for {judge_model}: {e}[/red]"
                    )
                    return None

        if reverse_order:
            # Run both original and reversed order in parallel
            judgments = await asyncio.gather(
                get_single_judgment(question, response_A, response_B),
                get_single_judgment(question, response_B, response_A, is_reversed=True),
            )
        else:
            judgment = await get_single_judgment(question, response_A, response_B)
            judgments = [judgment]

        pair["judge_name"] = judge_name
        pair["judgments"] = judgments
        return pair

    tasks = [asyncio.create_task(judge_pair(pair)) for pair in pairs]

    if progress:
        pair_task = progress.add_task(f"[cyan]Model: {judge_model}", total=len(pairs))
        for future in asyncio.as_completed(tasks):
            pair = await future
            if output_file is not None:
                async with file_lock:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            progress.advance(pair_task)
    else:
        for future in tqdm_asyncio.as_completed(tasks):
            pair = await future
            if output_file is not None:
                async with file_lock:
                    with open(output_file, "a") as f:
                        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return pairs


async def main(args: argparse.Namespace) -> None:
    """Main execution function for the judging script.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing judge settings and input path.
    """
    load_dotenv()
    random.seed(args.seed)

    # 1. Split models and validate
    models_list = [m.strip() for m in args.judge_model.split(",")]
    invalid_models = []
    for model in models_list:
        try:
            # Check if model is valid via litellm. validate_environment returns a dict.
            # It might return missing keys if some env vars are not set.
            validation = litellm.validate_environment(model)
            if validation.get("missing_keys"):
                invalid_models.append(
                    f"{model} (Missing environment variables: {', '.join(validation['missing_keys'])})"
                )
        except Exception as e:
            invalid_models.append(f"{model} (Validation error: {str(e)})")

    if invalid_models:
        error_msg = "\n".join(invalid_models)
        raise ValueError(
            f"The following models are unavailable or missing required configuration:\n{error_msg}"
        )

    # 2. Read pairs once
    original_pairs = file_operations.read_jsonl(args.pairs)
    dataset_name = os.path.basename(args.pairs).replace(".jsonl", "")

    # 3. Process all models concurrently with rich progress
    global_semaphore = asyncio.Semaphore(args.concurrency_limit)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[bold green]Overall Model Progress", total=len(models_list)
        )

        async def run_single_model(model: str):
            # Re-read or copy pairs for each model
            pairs_to_judge = [p.copy() for p in original_pairs]

            file_name = f"{dataset_name},judge_name={args.judge_name},judge_model={model.replace('/', '_')}.jsonl"
            os.makedirs("./outputs", exist_ok=True)
            file_path = os.path.join("./outputs", file_name)

            if os.path.exists(file_path):
                console.log(
                    f"[yellow]File {file_path} already exists. Skipping finished pairs for {model}.[/yellow]"
                )
                existing_pairs = file_operations.read_jsonl(file_path)
                existing_pair_ids = {pair["pair_id"] for pair in existing_pairs}
                pairs_to_judge = [
                    pair
                    for pair in pairs_to_judge
                    if pair["pair_id"] not in existing_pair_ids
                ]
                console.log(
                    f"[blue]{model}: Skipped {len(original_pairs) - len(pairs_to_judge)} already judged pairs.[/blue]"
                )

            if pairs_to_judge:
                await judge_pairs(
                    pairs_to_judge,
                    args.judge_name,
                    model,
                    reverse_order=not args.single_game,
                    concurrency_limit=args.concurrency_limit,
                    output_file=file_path,
                    progress=progress,
                    global_semaphore=global_semaphore,
                )

            # Compute final metrics for this model
            all_history_pairs = file_operations.read_jsonl(file_path)
            console.log(f"[bold green]Results for {model}:[/bold green]")
            for source in [
                "mmlu-pro",
                "livebench-reasoning",
                "livebench-math",
                "livecodebench",
                "",
            ]:
                score = metrics.compute_final_metrics(
                    all_history_pairs,
                    not args.single_game,
                    include_fn=lambda x: x["source"].startswith(source),
                )
                label = source if source else "Overall"
                console.log(
                    f"  {model} - {label}: [bold green]{score:.2f}%[/bold green]"
                )

            progress.advance(overall_task)

        # Create tasks for all models and run concurrently
        model_tasks = [run_single_model(model) for model in models_list]
        await asyncio.gather(*model_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge_name", type=str, required=True
    )  # name of judge, should correspond to an entry in utils/judges/get_judge_from_judge_name_and_model.
    parser.add_argument(
        "--judge_model", type=str, required=True
    )  # model to be used by judge.
    parser.add_argument(
        "--single_game", action="store_true"
    )  # by default, we run each pair through twice (A,B) and (B,A). This flag will only run the original ordering, and should be used if a judge is order-independent.
    parser.add_argument("--seed", type=int, default=42)  # seed to use.
    parser.add_argument(
        "--concurrency_limit", type=int, default=1
    )  # We use asyncio to speed things up, 10 is usally a good value here.
    parser.add_argument(
        "--pairs", type=str, required=True
    )  # path to jsonl containing pairs for judging
    args = parser.parse_args()
    asyncio.run(main(args))

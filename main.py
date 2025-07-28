#!/usr/bin/env python3

import hashlib
import json
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rich.progress import Progress
from sklearn.metrics import accuracy_score

from ontology_learning.tasks import TASKS, Task
from ontology_learning.prediction.llms4ol import predict as predict_llms4ol
from ontology_learning.prediction.neon_gpt import predict as predict_neongpt
from ontology_learning.prediction.neon_cot import predict as predict_neoncot
from ontology_learning.prediction.zero_shot import predict as predict_zero_shot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--task", default="t1")
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--dataset", default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--fix",
        action="store_true",
    )
    parser.add_argument(
        "--strip",
        action="store_true",
    )
    parser.add_argument("--save")
    parser.add_argument("--top", type=int)
    parser.add_argument("--sample", type=int)

    args = parser.parse_args()
    assert args.top is None or args.sample is None, (
        "'top' and 'sample' are mutually exclusive"
    )
    return args


def load_data(task: Task, dataset: str):
    return pd.read_csv(
        task.path / f"{dataset}.tsv",
        sep="\t",
        index_col="index",
    )


def predict(mode: str):
    match mode:
        case "0-shot":
            return predict_zero_shot
        case "llms4ol":
            return predict_llms4ol
        case "neongpt":
            return predict_neongpt
        case "neoncot":
            return predict_neoncot
        case _:
            raise ValueError(f"mode not supported: '{mode}'")


@dataclass(frozen=True, kw_only=True)
class Result:
    task_name: str
    mode: str
    model: str
    predictions: list[str | None]
    true_vals: list[str]


def accuracy_summary(res: Result, baseline: list[str | None] | None) -> str:
    assert len(res.true_vals) == len(res.predictions), (
        "predictions and true values must be equal in length"
    )
    if baseline is not None:
        assert len(res.predictions) == len(baseline), (
            "predictions and baseline predictions must be equal in length"
        )

    true_vals, preds, baseline_preds = [], [], []
    for ix, pred in enumerate(res.predictions):
        if pred is None:
            continue
        if baseline is not None and baseline[ix] is None:
            continue
        preds.append(pred)
        true_vals.append(res.true_vals[ix])
        if baseline:
            baseline_preds.append(baseline[ix])

    baseline_correct = [bp == tv for bp, tv in zip(baseline_preds, true_vals)]
    baseline_success_indices = [
        i for i, correct in enumerate(baseline_correct) if correct
    ]
    baseline_failure_indices = [
        i for i, correct in enumerate(baseline_correct) if not correct
    ]

    success_predictions = [preds[i] for i in baseline_success_indices]
    success_true_vals = [true_vals[i] for i in baseline_success_indices]
    success_accuracy = accuracy_score(success_true_vals, success_predictions)

    failure_predictions = [preds[i] for i in baseline_failure_indices]
    failure_true_vals = [true_vals[i] for i in baseline_failure_indices]
    failure_accuracy = accuracy_score(failure_true_vals, failure_predictions)

    accuracy = accuracy_score(true_vals, preds)
    summary = ""
    summary += f"{'=' * 80}\n"
    summary += f"ACCURACY SUMMARY :: {res.task_name} :: {res.mode} :: {res.model}\n"
    summary += f"{'=' * 80}\n"
    summary += f"Overall Accuracy: {accuracy * 100:.2f}%\n"
    if baseline is not None:
        summary += f"Accuracy on baseline successes: {success_accuracy * 100:.2f}%\n"
        summary += f"Accuracy on baseline failures: {failure_accuracy * 100:.2f}%\n"
    summary += "-" * 80

    return summary


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:6]


def run_experiment(
    progress,
    dataset,
    task_name,
    mode,
    model,
    debug,
    fix,
    strip,
    top,
    sample,
    save_path,
) -> Result:
    if not (task := TASKS.get(task_name)):
        raise Exception(f"err: unknown task '{task_name}'")

    data = load_data(task, dataset)
    if top is not None:
        data = data[:top]

    if sample is not None:
        data = data.sample(n=sample, random_state=42)

    curr_task = progress.add_task(f"[cyan]{task_name} / {mode}", total=len(data))

    predictions, true_vals = [], []
    for _, row in data.iterrows():
        task_dir = None
        if save_path:
            task_dir = save_path / task.name / short_hash(row["text"])
            task_dir.mkdir(parents=True, exist_ok=True)
            with open(task_dir / "INPUT", "w") as f:
                f.write(row["text"])

        progress.update(curr_task, advance=1)
        res = predict(mode)(
            model=model,
            task=task,
            input=row["text"],
            debug=debug,
            fix=fix,
            strip=strip,
            task_dir=task_dir,
        )

        predictions.append(res)
        true_vals.append(row["answer"])

        if task_dir:
            if mode == "0-shot":
                with open(
                    task_dir / ("success" if res == row["answer"] else "failure"), "w"
                ) as f:
                    f.write(f"{res}")
            else:
                targetf = (
                    task_dir / f"{mode}.success"
                    if res == row["answer"]
                    else task_dir / f"{mode}.failure"
                )
                shutil.move(task_dir / mode, targetf)

        if debug:
            if res == row["answer"]:
                print("Success!")
            else:
                print(f"Failure: '{res}' vs '{row['answer']}'")

    progress.remove_task(curr_task)

    return Result(
        predictions=predictions,
        true_vals=true_vals,
        task_name=task.name,
        mode=mode,
        model=model,
    )


def iter_modes(args):
    if args.mode == "all":
        yield from ["0-shot", "llms4ol", "neongpt", "neoncot"]
    else:
        yield args.mode


def iter_tasks(args):
    if args.task == "all":
        yield from (f"t{t}" for t in range(1, len(TASKS) + 1))
    else:
        yield args.task


def run_tasks(args):
    per_mode, per_task = {}, {}
    tasks = [
        (task_name, mode) for task_name in iter_tasks(args) for mode in iter_modes(args)
    ]
    save_path = Path(args.save) if args.save else None

    with Progress() as progress:
        label = f"Tasks / {args.model}"
        if args.top is not None:
            label += f" / top-{args.top}"
        if args.sample is not None:
            label += f" / sample-{args.sample}"
        main_task = progress.add_task(f"[green]{label}", total=len(tasks))

        for task_name, mode in tasks:
            progress.update(main_task, advance=1)
            res = run_experiment(
                progress=progress,
                dataset=args.dataset,
                task_name=task_name,
                mode=mode,
                model=args.model,
                debug=args.debug,
                fix=args.fix,
                strip=args.strip,
                top=args.top,
                sample=args.sample,
                save_path=save_path,
            )
            if mode not in per_mode:
                per_mode[mode] = {"predictions": [], "true_vals": []}
            if task_name not in per_task:
                per_task[task_name] = {}
            per_mode[mode]["predictions"].extend(res.predictions)
            per_mode[mode]["true_vals"].extend(res.true_vals)
            per_task[task_name][mode] = {
                "predictions": res.predictions,
                "true_vals": res.true_vals,
            }

    per_task_loss = {}

    for task_name, modes in per_task.items():
        for m, res in modes.items():
            if task_name not in per_task_loss:
                per_task_loss[task_name] = {}

            tvals = res["true_vals"]

            if m == "0-shot":
                per_task_loss[task_name]["baseline"] = round(
                    accuracy_score(tvals, [p or "Z" for p in res["predictions"]]), 3
                )
                continue

            pacc = accuracy_score(tvals, [p or "Z" for p in res["predictions"]])
            bacc = accuracy_score(
                tvals, [p or "Z" for p in modes["0-shot"]["predictions"]]
            )
            loss = bacc - pacc
            per_task_loss[task_name][m] = round(loss, 3) if loss > 0 else 0.0

    print("Per task loss:")
    print(json.dumps(per_task_loss))

    for mode, res in per_mode.items():
        summary = accuracy_summary(
            res=Result(
                predictions=res["predictions"],
                true_vals=res["true_vals"],
                task_name="all",
                model=args.model,
                mode=mode,
            ),
            baseline=(
                per_mode["0-shot"]["predictions"]
                if "0-shot" in per_mode and mode != "0-shot"
                else None
            ),
        )
        print(summary)


def main():
    load_dotenv()
    args = parse_args()
    return run_tasks(args)


if __name__ == "__main__":
    main()

# Based on https://github.com/zacharyhorvitz/Fk-Diffusion-Steering/tree/main/discrete_diffusion/evaluation
# Based on https://github.com/xhan77/ssd-lm
'''

Example usage:

python evaluation/evaluate.py \
--generations_file '[path_to_gen].jsonl' \
--metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext,unique_edit \
--output_file '[path_to_gen]_eval.txt'

# unique_edit: normalized Levenshtein (max_len) <= 0.05 clustering.


'''
import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

_LM_ROOT = Path(__file__).resolve().parents[1]
if str(_LM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LM_ROOT))
from text_samples_edit_distance import unique_count_for_texts
from model_revisions import pretrained_kwargs
from evaluation.group_metric_stats import (
    group_distinctness,
    group_means,
    group_perplexities,
    group_unique_ratio_sweep,
    group_unique_ratios,
    particle_group_slices,
    replace_metric_std_section,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)


def conditional_perplexity(
    generations_df, model, tokenizer, device='cuda', write_file=None
):
    perplexities = []
    all_perplexities = []
    generation_nlls = []
    generation_token_counts = []
    goodperplexities = []
    total_nll = 0
    total_tokens = 0
    g = 0
    ct = 0
    if write_file is not None:
        fout = open(write_file, "w")

    # for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc='Evaluating PPL',
    ):
        # prompt_input_ids = torch.LongTensor([row.prompt['tokens']]).to(device)
        prompt = row.context_string
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        if not (
            prompt_input_ids.shape[1] == 1
            and prompt_input_ids[0].tolist()[0] == tokenizer.bos_token_id
        ):  # this means unconditional, prompt is BOS token (verify)
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0] * (
                prompt_input_ids.shape[1] - 1
            )
            # print("in")
        else:
            prompt_loss = 0
            # print("out")
        # for every generation conditioned on the prompt
        generations = row.string
        for gen in generations:
            full_input_ids = tokenizer.encode(f'{prompt}{gen}', return_tensors='pt').to(
                device
            )
            full_loss = model(full_input_ids, labels=full_input_ids)[0] * (
                full_input_ids.shape[1] - 1
            )
            loss = (full_loss - prompt_loss) / (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )

            ppl = np.exp(loss.item())
            generation_nll = (full_loss - prompt_loss).item()
            generation_token_count = (
                full_input_ids.shape[1] - prompt_input_ids.shape[1]
            )
            all_perplexities.append(ppl)
            generation_nlls.append(generation_nll)
            generation_token_counts.append(generation_token_count)
            if ppl < 100:  # for sanity
                goodperplexities.append(ppl)
                # perplexities.append(ppl)
                g += 1

            if ppl < 1e4:
                perplexities.append(ppl)
            else:
                print("ppl values are weirldly large. Check for errors")
                print(f"\n########\n{gen}\n########\n")

            total_nll += generation_nll
            total_tokens += generation_token_count
            # print(full_input_ids[0], prompt_input_ids[0])
            # print(full_loss, prompt_loss)
            # input()
            if write_file is not None:
                fout.write(
                    f"{ppl}, {generation_nll}, {generation_token_count}\n"
                )

    if write_file is not None:
        fout.close()
    return (
        np.nanmean(perplexities),
        np.exp(total_nll / total_tokens),
        all_perplexities,
        generation_nlls,
        generation_token_counts,
    )


def predict_binary_labels(
    texts, model_name, positive_label_idx=1, batch_size=32, desc=None
):
    revision_kwargs = pretrained_kwargs(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **revision_kwargs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    labels = []
    iterator = range(0, len(texts), batch_size)
    for start in tqdm(iterator, desc=desc):
        encoded_input = tokenizer(
            texts[start : start + batch_size],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            predicted = model(**encoded_input).logits.argmax(dim=-1).tolist()
        labels.extend(int(label == positive_label_idx) for label in predicted)

    del model
    torch.cuda.empty_cache()
    return labels


def fluency_classify(generations_df):
    labels = predict_binary_labels(
        flatten_prompt_outputs(generations_df),
        'textattack/roberta-base-CoLA',
        desc='cola',
    )
    return sum(labels) / len(labels), labels


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(
        generations_df.iterrows(),
        total=len(generations_df.index),
        desc='Evaluating dist-n',
    ):
        generations = row['string']
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i + 1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)

    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)


def flatten_prompt_outputs(generations_df):
    """Same order as toxicity eval: one string per generation (prompt + output)."""
    machine_text = []
    for i, row in generations_df.iterrows():
        prompt = row['context_string']
        for output in row['string']:
            machine_text.append(f'{prompt}{output}')
    return machine_text


def predict_toxic_labels(generations_df, ctr_label_idx):
    return predict_binary_labels(
        flatten_prompt_outputs(generations_df),
        'SkolkovoInstitute/roberta_toxicity_classifier',
        positive_label_idx=ctr_label_idx,
        desc='toxic (roberta)',
    )


def compute_toxic_from_labels(acc_list):
    return sum(acc_list) / len(acc_list)


def predict_toxic_ext_labels(generations_df, ctr_label_idx):
    return predict_binary_labels(
        flatten_prompt_outputs(generations_df),
        'textdetox/xlmr-large-toxicity-classifier',
        positive_label_idx=ctr_label_idx,
        desc='toxic_ext (xlmr)',
    )


def compute_toxic_ext_from_labels(acc_list):
    return sum(acc_list) / len(acc_list)


def mean_unique_ratio_per_particle_groups(
    texts: list[str],
    threshold: float,
    num_particles: int,
    filter_labels: list[int] | None = None,
) -> tuple[float, int]:
    """
    Split `texts` into consecutive groups of length `num_particles` (last group may be shorter).

    In each group: optionally keep only entries where `filter_labels` == 1 (aligned slice).
    Let U = unique cluster count (normalized Levenshtein <= threshold) on that sublist.
    Record ratio U / num_particles for the group.

    Returns (mean of ratios over groups, number of groups).
    """
    n = len(texts)
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    if filter_labels is not None and len(filter_labels) != n:
        raise ValueError("filter_labels length must match texts")

    ratios: list[float] = []
    for start in range(0, n, num_particles):
        end = min(start + num_particles, n)
        chunk_t = texts[start:end]
        if filter_labels is None:
            sub = chunk_t
        else:
            chunk_l = filter_labels[start:end]
            sub = [t for t, lab in zip(chunk_t, chunk_l) if lab == 1]
        u = unique_count_for_texts(sub, threshold, 'max_len')
        ratios.append(u / float(num_particles))

    if not ratios:
        return 0.0, 0
    return sum(ratios) / len(ratios), len(ratios)


def run_unique_edit_metrics(
    generations_df,
    threshold: float,
    output_dir,
    output_file,
    num_particles: int | None,
    toxic_labels=None,
    toxic_ext_labels=None,
    particle_slices=None,
    precomputed_group_ratios=None,
):
    """
    Normalized Levenshtein (max_len) clustering at <= threshold.

    If `num_particles` is set: consecutive groups of that size (indices 0..P-1, P..2P-1, ...).
    Per group, unique cluster count U; score U/num_particles; report mean over groups.
    Same for toxic / toxic_ext: within each group, only texts with label 1 are clustered.

    If `num_particles` is None: fall back to global clustering (single pool).
    Pass precomputed label lists when available to avoid reloading classifiers.
    """
    all_texts = flatten_prompt_outputs(generations_df)
    n_all = len(all_texts)
    print(
        f'computing unique_edit (norm Levenshtein <= {threshold}, max_len), n={n_all}, '
        f'num_particles={num_particles}'
    )

    if toxic_labels is None:
        toxic_labels = predict_toxic_labels(generations_df, 1)
    if toxic_ext_labels is None:
        toxic_ext_labels = predict_toxic_ext_labels(generations_df, 1)

    n_tox = sum(toxic_labels)
    n_toxe = sum(toxic_ext_labels)

    if num_particles is not None and num_particles > 0:
        if particle_slices is None:
            particle_slices = particle_group_slices(
                [len(all_texts)], num_particles
            )
        if precomputed_group_ratios is None:
            precomputed_group_ratios = {
                'all': group_unique_ratios(
                    all_texts, particle_slices, threshold, num_particles
                ),
                'toxic': group_unique_ratios(
                    all_texts,
                    particle_slices,
                    threshold,
                    num_particles,
                    toxic_labels,
                ),
                'toxic_ext': group_unique_ratios(
                    all_texts,
                    particle_slices,
                    threshold,
                    num_particles,
                    toxic_ext_labels,
                ),
            }
        ratios_all = precomputed_group_ratios['all']
        ratios_tox = precomputed_group_ratios['toxic']
        ratios_toxe = precomputed_group_ratios['toxic_ext']
        counts_all = [ratio * num_particles for ratio in ratios_all]
        counts_tox = [ratio * num_particles for ratio in ratios_tox]
        counts_toxe = [ratio * num_particles for ratio in ratios_toxe]
        mean_all = sum(ratios_all) / len(ratios_all) if ratios_all else 0.0
        mean_tox = sum(ratios_tox) / len(ratios_tox) if ratios_tox else 0.0
        mean_toxe = sum(ratios_toxe) / len(ratios_toxe) if ratios_toxe else 0.0
        mean_count_all = sum(counts_all) / len(counts_all) if counts_all else 0.0
        mean_count_tox = sum(counts_tox) / len(counts_tox) if counts_tox else 0.0
        mean_count_toxe = (
            sum(counts_toxe) / len(counts_toxe) if counts_toxe else 0.0
        )
        g_all, g_tox, g_toxe = (
            len(ratios_all),
            len(ratios_tox),
            len(ratios_toxe),
        )
        lines = [
            f'unique_edit (all generations, mean unique/num_particles over {g_all} groups of {num_particles}, '
            f'norm Levenshtein <= {threshold}) = {mean_all:.6f}  (total lines {n_all})\n',
            f'unique_edit (toxic predicted toxic, per-group mean unique/num_particles) = {mean_tox:.6f}  '
            f'({g_tox} groups, {n_tox} toxic lines)\n',
            f'unique_edit (toxic_ext predicted toxic, per-group mean unique/num_particles) = {mean_toxe:.6f}  '
            f'({g_toxe} groups, {n_toxe} toxic_ext lines)\n',
            f'unique_edit_count (all generations, per-group mean unique count) = {mean_count_all:.6f}\n',
            f'unique_edit_count (toxic predicted toxic, per-group mean unique count) = {mean_count_tox:.6f}\n',
            f'unique_edit_count (toxic_ext predicted toxic, per-group mean unique count) = {mean_count_toxe:.6f}\n',
        ]
    else:
        u_all = unique_count_for_texts(all_texts, threshold, 'max_len')
        tox_texts = [t for t, lab in zip(all_texts, toxic_labels) if lab == 1]
        u_tox = unique_count_for_texts(tox_texts, threshold, 'max_len')
        toxe_texts = [t for t, lab in zip(all_texts, toxic_ext_labels) if lab == 1]
        u_toxe = unique_count_for_texts(toxe_texts, threshold, 'max_len')
        lines = [
            f'unique_edit (all generations, global, norm Levenshtein <= {threshold}) = {u_all} / {n_all}\n',
            f'unique_edit (toxic predicted toxic, global subset) = {u_tox} / {n_tox}\n',
            f'unique_edit (toxic_ext predicted toxic, global subset) = {u_toxe} / {n_toxe}\n',
        ]

    with open(output_dir / output_file, 'a') as fo:
        for line in lines:
            fo.write(line)
            print(line.strip())

    if num_particles is None or num_particles <= 0:
        return {}
    return {
        f'unique_edit#{threshold:g} all': ratios_all,
        f'unique_edit#{threshold:g} toxic': ratios_tox,
        f'unique_edit#{threshold:g} toxic_ext': ratios_toxe,
        f'unique_edit_count#{threshold:g} all': counts_all,
        f'unique_edit_count#{threshold:g} toxic': counts_tox,
        f'unique_edit_count#{threshold:g} toxic_ext': counts_toxe,
    }


@click.command()
@click.option(
    '--generations_file',
    required=True,
    type=str,
    help='a jsonl file with generations and attribute scores',
)
@click.option(
    '--output_file', required=True, type=str, help='filename to write the results to'
)
@click.option(
    '--metrics',
    required=True,
    type=str,
    help='which metrics to compute, write comma separeted, ppl-mid,ppl-big,cola,self-bleu,zipf,repetition,dist-n',
)
@click.option('--extra', required=False, type=str, help='extra params')
@click.option(
    '--num-particles',
    type=int,
    default=None,
    metavar='P',
    help='For unique_edit: group flat generations into consecutive chunks of size P '
    '(mean of unique/P per chunk). Typical: smc.num_particles.',
)
def main(generations_file, output_file, metrics, extra, num_particles):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    generations_df = pd.read_json(generations_file, lines=True)

    metricset = set(metrics.strip().split(","))  # cannot use lower here
    toxic_labels_cache = None
    toxic_ext_labels_cache = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row_lengths = [len(strings) for strings in generations_df['string']]
    particle_slices = (
        particle_group_slices(row_lengths, num_particles)
        if num_particles is not None and num_particles > 0
        else None
    )
    metric_groups = {}
    ### calculate quality metrics

    # Fluency
    fo = open(output_dir / output_file, 'w')  # creating the file
    fo.close()

    # print(metrics)
    if "ppl" in metrics:
        for metric in metricset:
            if "ppl" in metric:
                eval_modelname = metric.split("#")[1]
                print(f'computing {eval_modelname} ppl')
                if 'llama3' in eval_modelname:
                    LLAMA_TOKEN = os.environ['LLAMA_TOKEN']
                    model_name = "meta-llama/Meta-Llama-3-8B"
                    print(f"Loading {model_name}")
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        model_name, use_auth_token=LLAMA_TOKEN
                    ).to(device)
                    eval_tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_auth_token=LLAMA_TOKEN
                    )
                else:
                    revision_kwargs = pretrained_kwargs(eval_modelname)
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        eval_modelname, **revision_kwargs
                    ).to(device)
                    eval_tokenizer = AutoTokenizer.from_pretrained(
                        eval_modelname, **revision_kwargs
                    )
                torch.cuda.empty_cache()
                with torch.no_grad():
                    (
                        ppl,
                        total_ppl,
                        generation_ppls,
                        generation_nlls,
                        generation_token_counts,
                    ) = conditional_perplexity(
                        generations_df,
                        eval_model,
                        eval_tokenizer,
                        device=device,
                        write_file=output_dir
                        / (output_file + ".ppl-" + eval_modelname.replace("/", "-")),
                    )
                if particle_slices is not None:
                    group_ppls, group_total_ppls = group_perplexities(
                        generation_ppls,
                        generation_nlls,
                        generation_token_counts,
                        particle_slices,
                    )
                    metric_groups[f'{eval_modelname} perplexity'] = group_ppls
                    metric_groups[
                        f'{eval_modelname} total perplexity'
                    ] = group_total_ppls

                # write output results
                with open(output_dir / output_file, 'a') as fo:
                    fo.write(
                        f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n'
                    )
                    print(
                        f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n'
                    )

                del eval_model

    # cola
    if "cola" in metricset:
        print("computing fluency (cola)")
        # cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        cola_accuracy, cola_labels = fluency_classify(generations_df)
        if particle_slices is not None:
            metric_groups['cola acceptability accuracy'] = group_means(
                cola_labels, particle_slices
            )

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)
        if particle_slices is not None:
            flat_outputs = [
                output
                for strings in generations_df['string']
                for output in strings
            ]
            group_dist1, group_dist2, group_dist3 = group_distinctness(
                flat_outputs, particle_slices
            )
            metric_groups['dist-1'] = group_dist1
            metric_groups['dist-2'] = group_dist2
            metric_groups['dist-3'] = group_dist3

        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    if "toxic" in metricset:
        toxic_labels_cache = predict_toxic_labels(generations_df, 1)
        acc = compute_toxic_from_labels(toxic_labels_cache)
        if particle_slices is not None:
            metric_groups['toxic acc'] = group_means(
                toxic_labels_cache, particle_slices
            )
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'toxic acc = {acc}\n')
            print(f'toxic acc = {acc}')

    if "toxic_ext" in metricset:
        toxic_ext_labels_cache = predict_toxic_ext_labels(generations_df, 1)
        acc = compute_toxic_ext_from_labels(toxic_ext_labels_cache)
        if particle_slices is not None:
            metric_groups['toxic_ext acc'] = group_means(
                toxic_ext_labels_cache, particle_slices
            )
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'toxic_ext acc = {acc}\n')
            print(f'toxic_ext acc = {acc}')

    unique_tokens = [
        m for m in metrics.strip().split(",")
        if m == "unique_edit" or m.startswith("unique_edit#")
    ]
    unique_thresholds = []
    seen_unique_thresholds = set()
    for token in unique_tokens:
        thr = 0.05 if token == "unique_edit" else float(token.split("#", 1)[1])
        if thr in seen_unique_thresholds:
            continue
        seen_unique_thresholds.add(thr)
        unique_thresholds.append(thr)
    unique_sweeps = None
    if unique_thresholds:
        if toxic_labels_cache is None:
            toxic_labels_cache = predict_toxic_labels(generations_df, 1)
        if toxic_ext_labels_cache is None:
            toxic_ext_labels_cache = predict_toxic_ext_labels(
                generations_df, 1
            )
    if unique_thresholds and particle_slices is not None:
        unique_texts = flatten_prompt_outputs(generations_df)
        unique_sweeps = {
            'all': group_unique_ratio_sweep(
                unique_texts,
                particle_slices,
                unique_thresholds,
                num_particles,
            ),
            'toxic': group_unique_ratio_sweep(
                unique_texts,
                particle_slices,
                unique_thresholds,
                num_particles,
                toxic_labels_cache,
            ),
            'toxic_ext': group_unique_ratio_sweep(
                unique_texts,
                particle_slices,
                unique_thresholds,
                num_particles,
                toxic_ext_labels_cache,
            ),
        }
    if len(unique_thresholds) > 1:
        with open(output_dir / output_file, 'a') as fo:
            fo.write('\n--- unique_edit threshold sweep ---\n')
    for thr in unique_thresholds:
        unique_group_metrics = run_unique_edit_metrics(
            generations_df,
            thr,
            output_dir,
            output_file,
            num_particles,
            toxic_labels=toxic_labels_cache,
            toxic_ext_labels=toxic_ext_labels_cache,
            particle_slices=particle_slices,
            precomputed_group_ratios=(
                {
                    subset: sweep[thr]
                    for subset, sweep in unique_sweeps.items()
                }
                if unique_sweeps is not None
                else None
            ),
        )
        metric_groups.update(unique_group_metrics)

    if particle_slices is not None:
        replace_metric_std_section(
            output_dir / output_file, metric_groups, num_particles
        )


if __name__ == '__main__':
    main()

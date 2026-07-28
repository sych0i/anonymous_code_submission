"""

Convert the output of the MDLM model to the format expected by the SSD-LM evaluation script.

"""

# Current:
# {"text": "<|endoftext|>\n\The last time ..."}

# Goal (for SSD-LM eval script):
# {
#   "context_len": 5,
#   "context": [],
#   "context_string": "",
#   "len": 0,
#   "tokens": [],
#   "string": [],
#   "gold_tokens": [],
#   "gold_string": ""
# }

import os
import glob
import json

import click

from transformers import AutoTokenizer
from model_revisions import pretrained_kwargs


def get_possible_prompts(prompt_path):
    with open(prompt_path) as f:
        return [json.loads(line)["context_string"] for line in f]


def file_to_exp_info(file):
    parent_dir = os.path.dirname(file)
    info_path = os.path.join(parent_dir, 'info.json')

    with open(info_path) as f:
        relevant_config = json.load(f)['fk_steering']

    relevant_keys = [
        'potential_type',
        'k_particles',
        'lmbda',
        'reward_fn',
        'reward_label',
        'num_x0_samples',
    ]
    relevant_config = '_'.join([str(relevant_config[key]) for key in relevant_keys])

    return relevant_config


def _strip_leading_endoftext(text: str) -> str:
    prefix = "<|endoftext|>"
    if text.startswith(prefix):
        text = text[len(prefix) :].lstrip()
    return text


def load_records(file):
    """JSONL lines from eval.py: {prompt, text, ...} or legacy {text} only."""
    records = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_texts(file):
    return [r["text"] for r in load_records(file)]


def process_prompted_output(prompt_to_text, tokenizer, trim_len=50):
    prompt_to_data = {prompt: {} for prompt in prompt_to_text}

    for prompt, texts in prompt_to_text.items():
        cleaned_texts = []
        tokenized = []
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        prompt_to_data[prompt]["context_string"] = prompt
        prompt_to_data[prompt]["context_len"] = prompt_len
        prompt_to_data[prompt]["context"] = prompt_tokens

        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)[
                prompt_len : prompt_len + trim_len
            ]
            decoded_text = tokenizer.decode(tokenized_text)

            print('\t', decoded_text)

            cleaned_texts.append(decoded_text)
            tokenized.append(tokenized_text)

        prompt_to_data[prompt]["string"] = cleaned_texts
        prompt_to_data[prompt]["tokens"] = tokenized
        prompt_to_data[prompt]["len"] = len(tokenized[0])

    return prompt_to_data


def process_file(*, file, prompts, expected_per, tokenizer, max_len):
    # config_info = file_to_exp_info(file)
    config_info = "abc"  # not needed, I think
    records = load_records(file)

    # prompt -> list of full decoded strings (prefix \n\n + continuation), for matching / trim
    prompt_to_text: dict[str, list[str]] = {}

    # Prefer longest canonical prompt first when matching raw text (legacy lines without "prompt")
    sorted_prompts = sorted(prompts, key=len, reverse=True)

    for obj in records:
        raw = obj["text"]
        raw = _strip_leading_endoftext(raw)
        body = "\n\n" + raw.strip()

        declared = obj.get("prompt")
        if declared is not None:
            key = declared
        else:
            found = [p for p in sorted_prompts if body.startswith(p)]
            if len(found) != 1:
                raise ValueError(
                    f"Could not match a unique prompt for one line in {file} "
                    f"(found {len(found)} matches). Add a 'prompt' field in JSONL or fix text."
                )
            key = found[0]

        prompt_to_text.setdefault(key, [])
        prompt_to_text[key].append(body)

    # Drop prompts with no samples (e.g. full prompt list but partial eval)
    active = {p: xs for p, xs in prompt_to_text.items() if len(xs) > 0}
    if not active:
        raise ValueError(f"No samples found in {file}")

    counts = {p: len(xs) for p, xs in active.items()}
    if expected_per is None:
        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            raise ValueError(
                f"Samples per prompt differ {counts}; pass --expected_per or fix the file. "
                "Use the same number of lines per prompt when mixing multiple prompts."
            )
        n_expected = next(iter(unique_counts))
    else:
        n_expected = expected_per
        bad = {p: c for p, c in counts.items() if c != n_expected}
        if bad:
            raise ValueError(
                f"--expected_per={n_expected} but got counts {bad} in {file}"
            )

    prompt_to_data = process_prompted_output(active, tokenizer, max_len)
    return config_info, prompt_to_data


@click.command()
@click.option(
    '--glob_expression',
    default="../outputs/openwebtext-train/*/*/*/sample_evaluation/*/text_samples.jsonl",
    help='Glob pattern for input files.',
)
@click.option(
    '--prompt_path',
    default='pplm_discrim_prompts_orig.jsonl',
    help='Path to the prompt file.',
)
@click.option(
    '--max_len', default=50, type=int, help='Max length of generated text to consider.'
)
@click.option(
    '--expected_per',
    default=None,
    type=int,
    help='Samples per prompt; must match for every prompt present. '
    'Omit to infer automatically (all prompts must then have the same count).',
)
def main(glob_expression, prompt_path, max_len, expected_per):
    tokenizer_name = 'roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, **pretrained_kwargs(tokenizer_name)
    )

    prompts = get_possible_prompts(prompt_path)
    print(prompts)

    files = list(glob.glob(glob_expression))
    print(files)
    assert len(files) > 0

    for file in files:
        print(file)
        config_info, prompt_to_data = process_file(
            file=file,
            prompts=prompts,
            expected_per=expected_per,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        # get parent dir path
        s_path = os.path.join(os.path.dirname(file), config_info + '_ssdlm_gen.jsonl')

        with open(s_path, 'w') as f:
            for _, data in prompt_to_data.items():
                f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()

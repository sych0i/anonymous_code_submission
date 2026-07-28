"""Pinned Hugging Face revisions used by the released toxic-text experiment."""

MODEL_REVISIONS = {
    "kuleshov-group/mdlm-owt": "d0958fa851335ece6c15260ce0025f030673c0fb",
    "gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "gpt2-large": "32b71b12589c2f8d625668d2335a01cac3249519",
    "gpt2-xl": "15ea56dee5df4983c59b2538573817e1667135e2",
    "roberta-large": "722cf37b1afa9454edce342e7895e588b6ff1d59",
    "s-nlp/roberta_toxicity_classifier": (
        "048c25bb1e199b98802784f96325f4840f22145d"
    ),
    "SkolkovoInstitute/roberta_toxicity_classifier": (
        "048c25bb1e199b98802784f96325f4840f22145d"
    ),
    "textattack/roberta-base-CoLA": (
        "3ccf3a400f2fa75ff257eac171047603ffbe84f1"
    ),
    "textdetox/xlmr-large-toxicity-classifier": (
        "b9c7c563427c591fc318d91eb592381ae2fbde66"
    ),
}


def revision_for(model_name: str) -> str | None:
    return MODEL_REVISIONS.get(model_name)


def pretrained_kwargs(model_name: str) -> dict[str, str]:
    revision = revision_for(model_name)
    return {} if revision is None else {"revision": revision}

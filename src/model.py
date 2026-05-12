"""
BERT-based binary classifier for jailbreak prompt detection.
Wraps BertForSequenceClassification with a convenience build function.
"""

from transformers import BertForSequenceClassification, AutoConfig

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2  # 0=normal, 1=jailbreak


def build_model(
    model_name: str = MODEL_NAME,
    num_labels: int = NUM_LABELS,
) -> BertForSequenceClassification:
    """Load pretrained BERT and replace the classification head for num_labels classes."""
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model : {model_name}")
    print(f"  Total params     : {n_params:.1f}M")
    print(f"  Trainable params : {n_train:.1f}M")
    return model

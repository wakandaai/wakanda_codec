# codec/evaluation/evaluate.py

from codec.evaluation.espnet import compute_STOI, compute_PESQ, compute_MCD
from codec.evaluation.torchmetrics import compute_NISQA, compute_DNSMOS
from codec.evaluation.speaker_sim import compute_speaker_similarity
from codec.evaluation.utmos import compute_UTMOS
from codec.evaluation.wer import compute_wer, compute_cer
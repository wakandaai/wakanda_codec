# codec/evaluation/mcd.py
"""Compute Mel-Cepstral Distortion (MCD) between two audio signals."""

from pymcd.mcd import Calculate_MCD

def create_mcd_toolbox() -> Calculate_MCD:
    """Create and return an instance of the MCD calculation toolbox."""
    return Calculate_MCD(MCD_mode="plain")

def compute_mcd(mcd_toolbox: Calculate_MCD, reference_path: str, decoded_path: str) -> float:
    """
    Compute the Mel-Cepstral Distortion (MCD) between a reference and a decoded audio file.

    Args:
        mcd_toolbox: An instance of the MCD calculation toolbox.
        reference_path (str): Path to the reference audio file.
        decoded_path (str): Path to the decoded audio file.

    Returns:
        float: The computed MCD value.
    """
    mcd_value = mcd_toolbox.calculate_mcd(reference_path, decoded_path)
    return mcd_value
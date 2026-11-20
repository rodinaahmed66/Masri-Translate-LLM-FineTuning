import random

INSTR_EN2AR = "Translate the following English text into Egyptian Arabic"
INSTR_AR2EN = "Translate the following Egyptian Arabic text into English"


def to_alpaca(dataset, instruction):
    rows = []
    for ar, en in zip(dataset["Arabic"], dataset["English"]):
        rows.append(
            {
                "instruction": instruction,
                "input": en,
                "output": ar,
            }
        )
    return rows


def to_alpaca_bidirectional(
    dataset,
    instr_en2ar: str = INSTR_EN2AR,
    instr_ar2en: str = INSTR_AR2EN,
    seed: int = 42,
):
    """Build Alpaca rows for both translation directions, interleaved and shuffled."""
    rows = []
    for ar, en in zip(dataset["Arabic"], dataset["English"]):
        rows.append({"instruction": instr_en2ar, "input": en, "output": ar})
        rows.append({"instruction": instr_ar2en, "input": ar, "output": en})
    random.Random(seed).shuffle(rows)
    return rows

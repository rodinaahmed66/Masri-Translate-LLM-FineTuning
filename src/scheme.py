import random


def to_alpaca(dataset,INSTR_EN2AR,INSTR_AR2EN):
    rows = []
    for ar, en in zip(dataset["Arabic"], dataset["English"]):
        rows.append(
            {
                "instruction": INSTR_EN2AR,
                "input": en,
                "output": ar,
            }
        )
    
        rows.append(
            {
                "instruction": INSTR_AR2EN,
                "input": ar,
                "output": en,
            }
        )

    random.Random(42).shuffle(rows)
    return rows

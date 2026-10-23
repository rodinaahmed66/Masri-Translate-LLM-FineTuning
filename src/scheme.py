def to_alpaca(dataset,instruction):
    rows = []
    for ar, en in zip(dataset["Arabic"], dataset["English"]):
        rows.append({
            "instruction": instruction,
            "input": en,
            "output": ar,
        })
    return rows
    


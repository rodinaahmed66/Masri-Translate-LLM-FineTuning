def to_alpaca(dataset):
    rows = []
    for ar, en in zip(dataset["Arabic"], dataset["English"]):
        rows.append({
            "instruction": "Translate the following English text into Egyptian Arabic.",
            "input": en,
            "output": ar,
        })
    return rows
    


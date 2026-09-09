import pandas as pd

df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
df.to_csv("spam.csv", index=False)

ham = (df["label"] == "ham").sum()
spam = (df["label"] == "spam").sum()
print(f"Dataset ready: {len(df)} messages | Ham: {ham} | Spam: {spam}")

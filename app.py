import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

from scoring import escalation_score, misunderstanding_score, empathy_score

os.makedirs("outputs", exist_ok=True)

ds = load_dataset("go_emotions", "simplified")
df = pd.DataFrame(ds["train"])  # columns include text, labels

label_names = ds["train"].features["labels"].feature.names

def has_label(labels, name: str) -> int:
    target = label_names.index(name)
    return int(target in labels)

for lab in ["anger", "annoyance", "confusion", "gratitude", "approval", "neutral"]:
    df[lab] = df["labels"].apply(lambda x, lab=lab: has_label(x, lab))

# Add our scores
df["escalation_risk"] = df["text"].apply(escalation_score)
df["misunderstanding_risk"] = df["text"].apply(misunderstanding_score)
df["empathy_level"] = df["text"].apply(empathy_score)

# Save sample for citations / optional app examples
sample = df.sample(500, random_state=7).copy()
sample.to_csv("outputs/goemotions_sample_scored.csv", index=False)
print("Saved: outputs/goemotions_sample_scored.csv")

# Group comparisons
conflict_mask = (df["anger"] == 1) | (df["annoyance"] == 1)
calm_mask = (df["gratitude"] == 1) | (df["approval"] == 1)
confused_mask = (df["confusion"] == 1)

def group_mean(col, mask):
    return float(df.loc[mask, col].mean())

print("\n=== Mean scores by emotion group (GoEmotions train) ===")
print("Conflict-ish (anger OR annoyance):")
print("  escalation:", round(group_mean("escalation_risk", conflict_mask), 2))
print("  misunderstanding:", round(group_mean("misunderstanding_risk", conflict_mask), 2))
print("  empathy:", round(group_mean("empathy_level", conflict_mask), 2))
print("Calm-ish (gratitude OR approval):")
print("  escalation:", round(group_mean("escalation_risk", calm_mask), 2))
print("  misunderstanding:", round(group_mean("misunderstanding_risk", calm_mask), 2))
print("  empathy:", round(group_mean("empathy_level", calm_mask), 2))
print("Confusion label:")
print("  escalation:", round(group_mean("escalation_risk", confused_mask), 2))
print("  misunderstanding:", round(group_mean("misunderstanding_risk", confused_mask), 2))
print("  empathy:", round(group_mean("empathy_level", confused_mask), 2))

# One clean chart
plot_df = pd.DataFrame({
    "Escalation Risk": pd.concat(
        [df.loc[conflict_mask, "escalation_risk"], df.loc[calm_mask, "escalation_risk"]],
        ignore_index=True
    ),
    "Group": (["Conflict-ish"] * int(conflict_mask.sum())) + (["Calm-ish"] * int(calm_mask.sum()))
})

plt.figure()
plot_df.boxplot(column="Escalation Risk", by="Group")
plt.suptitle("")
plt.title("Escalation Risk by Emotion Group (GoEmotions)")
plt.ylabel("Score (0–100)")
plt.tight_layout()
plt.savefig("outputs/goemotions_escalation_boxplot.png", dpi=200)
print("Saved: outputs/goemotions_escalation_boxplot.png")
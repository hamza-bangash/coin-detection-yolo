import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# 🔧 CHANGE THIS
LABELS_DIR = "final_dataset/labels/train"

# Optional: your class names
CLASS_NAMES = [
    "I","A","L","C","R","S","T","V","O","M",
    "E","N","F","P","B","X","D","Q","G","H"
]

# Count classes
class_counts = Counter()

for file in Path(LABELS_DIR).rglob("*.txt"):
    with open(file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            class_counts[cls] += 1

# Sort by frequency (descending)
sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
classes = [i[0] for i in sorted_items]
counts = [i[1] for i in sorted_items]

labels = [CLASS_NAMES[c] for c in classes]

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts)

# Add values on top
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             str(count), ha='center', va='bottom', fontsize=9)

plt.title("Class Distribution (Sorted)")
plt.xlabel("Class")
plt.ylabel("Instances")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
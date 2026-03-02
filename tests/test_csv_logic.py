import argparse
import csv
from standalone_translator import translate

print("Testing direct CSV translation logic...")
with open("benchmark.csv", "r") as f:
    rows = list(csv.DictReader(f))

texts = [r["prompt"] for r in rows]
print(f"Loaded {len(texts)} prompts for translation: {texts}")

try:
    print("Sending batch to translate()...")
    res = translate(texts, "pt")
    print(f"Result: {res}")
except Exception as e:
    print(f"Error during translation: {e}")

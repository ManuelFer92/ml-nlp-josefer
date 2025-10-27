import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "emotions.csv"

def load_split(filename, split):
    df = pd.read_csv(DATA_DIR / filename, sep=';', header=None, names=['text', 'label'], dtype=str, encoding='utf-8')
    df['text'] = df['text'].str.strip()
    df['label'] = df['label'].str.strip()
    df['split'] = split
    return df

def create_emotions_csv():
    df = pd.concat([
        load_split("train.txt", "train"),
        load_split("val.txt", "val"),
        load_split("test.txt", "test"),
    ], ignore_index=True)

    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 0].drop_duplicates(subset=['text', 'label'])

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"✅ CSV generado en: {OUTPUT_CSV} | Total filas: {len(df)}")

if __name__ == "__main__":
    create_emotions_csv()

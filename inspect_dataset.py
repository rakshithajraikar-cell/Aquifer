
import pandas as pd
from pathlib import Path

file_path = Path("data/karnataka_groundwater.csv")

if not file_path.exists():
    print("Dataset not found!")
    print("Check that the CSV is inside the data folder.")
else:
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("\nRows and columns:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

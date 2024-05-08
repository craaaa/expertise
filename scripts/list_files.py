import argparse
from pathlib import Path

# Initialize parser
parser = argparse.ArgumentParser(
    prog="list_files.py",
    description="Lists files for object naming task in stimuli_to_test.txt",
)
parser.add_argument("dataset", help="Name of dataset to list: van_hoef|oxford_iit")
args = parser.parse_args()

filenames = []
dataset = args.dataset
for category in ["Tree", "Flower", "Dog", "Bird", "Cat"]:
    files_path = Path(__file__).parent.parent.joinpath(f"data/{dataset}/{category}")
    filenames += list(files_path.glob("*.png")) + list(
        files_path.glob("*.jpg")
    )  # jpg or png files

with open(Path(__file__).parent.resolve() / "stimuli_to_test.txt", "w") as f:
    for path in filenames:
        f.write(str(path) + "\n")

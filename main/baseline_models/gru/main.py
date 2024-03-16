import subprocess

from variables import time_window

print(f"Running GRU for {time_window}h window.")

scripts = [
    "1_data_processing.py",
    "2_train.py",
    "3_eval.py",
    "4_confidence_interval.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished running GRU.")

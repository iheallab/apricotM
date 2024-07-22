import subprocess

from variables import time_window

print(f"Preparing prospective {time_window}h window data.")

scripts = [
    "1_extract_prosp.py",
    "2_build_dataset.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished prospective data processing.")

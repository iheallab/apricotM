import subprocess

from variables import time_window

print(f"Running episode prediction for {time_window}h window data.")

scripts = [
    "1_recalibration.py",
    "2_episode_prediction.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished episode prediction.")

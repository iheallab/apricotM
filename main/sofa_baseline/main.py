import subprocess

from variables import time_window

print(f"Calculating SOFA for {time_window}h window data.")

scripts = [
    "1_calculate_sofa.py",
    "2_sofa_probs.py",
    "3_sofa_acuity_criteria.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished SOFA calculation.")

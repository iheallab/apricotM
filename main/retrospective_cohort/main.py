import subprocess

from variables import time_window

print(f"Preparing {time_window}h window data.")

# scripts = [
#     "1_filter_split_cohort.py",
#     "2_build_static.py",
#     "3_build_seq.py",
#     "4_outcomes.py",
#     "5_build_hdf5.py",
# ]

scripts = [
    "4_outcomes.py",
    "5_build_hdf5.py",
]

for script in scripts:

    print("-" * 40)
    print(f"Running {script}...")

    subprocess.run(["python", script])

    print(f"Finished running {script}")

print("-" * 40)
print("Finished data processing.")

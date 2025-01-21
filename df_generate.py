import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of records to generate
num_records = 1000

# Generate Machine_IDs
machine_ids = [f"M_{i:03d}" for i in np.random.randint(1, 100, num_records)]

# Generate random temperatures between 50°C and 120°C
temperatures = np.random.uniform(50, 120, num_records)

# Generate random run time between 0 to 5000 hours
run_times = np.random.uniform(0, 5000, num_records)

# Generate downtime flag (1 if temperature > 100 and run time > 3000, else 0)
downtime_flags = np.where((temperatures > 100) & (run_times > 3000), 1, 0)

# Create DataFrame
df = pd.DataFrame({
    "Machine_ID": machine_ids,
    "Temperature": temperatures,
    "Run_Time": run_times,
    "Downtime_Flag": downtime_flags
})

# Display first few rows
print(df.head())

# Save to CSV
df.to_csv("synthetic_machine_data.csv", index=False)

#!.venv/bin/python3

import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Number of rows for the test data
num_rows = 100

# Generate test data
data = {
    'id': range(1, num_rows + 1),
    'name': [fake.name() for _ in range(num_rows)],
    'address': [fake.address().replace("\n", ", ") for _ in range(num_rows)],
    'email': [fake.email() for _ in range(num_rows)],
    'age': [random.randint(18, 80) for _ in range(num_rows)],
    'salary': [round(random.uniform(30000, 120000), 2) for _ in range(num_rows)],
    'joined_date': [fake.date_this_decade() for _ in range(num_rows)],
    'category': [random.choice(['A', 'B', 'C', 'D']) for _ in range(num_rows)],
    'boolean_flag': [random.choice([True, False]) for _ in range(num_rows)],
    'description': [fake.text(max_nb_chars=100) for _ in range(num_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('test_data.csv', index=False)

print("Test data CSV generated successfully!")

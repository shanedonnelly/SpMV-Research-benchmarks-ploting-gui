import streamlit as st # type: ignore
import pandas as pd
from filter_component import filter_dataframe
import datetime
import numpy as np

# Create a synthetic dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles', 'Chicago', 'New York', 'Los Angeles', 'Chicago', 'New York'],
    'Salary': [2000, 2500, 1000, 3000, 1500, 2200, 2800, 1200, 1800, 2400],
    'Birthdate': [datetime.datetime(1990, 1, 1, 12, 0, 0), datetime.datetime(1990, 2, 1, 12, 0, 0),
              datetime.datetime(1990, 3, 1, 12, 0, 0), datetime.datetime(1990, 4, 1, 12, 0, 0),
              datetime.datetime(1990, 5, 1, 12, 0, 0), datetime.datetime(1990, 6, 1, 12, 0, 0),
              datetime.datetime(1990, 7, 1, 12, 0, 0), datetime.datetime(1990, 8, 1, 12, 0, 0),
              datetime.datetime(1990, 9, 1, 12, 0, 0), datetime.datetime(1990, 10, 1, 12, 0, 0)]
}
df = pd.DataFrame(data)

# Ensure the date columns are of datetime type
df['Birthdate'] = pd.to_datetime(df['Birthdate'])


# Streamlit app
#st.title("Title Before")

# Use the filter component
filtered_df = filter_dataframe(df, max_unique=5)

#st.title("Title After")

# Display the number of rows in the filtered DataFrame
#st.write(f"Number of rows in the filtered DataFrame: {len(filtered_df)}")
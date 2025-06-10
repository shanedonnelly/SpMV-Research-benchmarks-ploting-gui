import configparser
import os
import pandas as pd
import glob

def create_config():
    """Create configuration file with default settings"""
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {
        'max_unique': 20,
        'show_df_by_default': 'False'
    }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    
    print("Configuration file created successfully.")

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['csv', 'pickle', 'subset_pickle']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")

def convert_csv_to_pickle():
    """Convert all CSV files in csv/ folder to Pickle format"""
    # Check if csv directory exists
    if not os.path.exists('csv'):
        print("CSV directory not found. Creating empty directory.")
        os.makedirs('csv')
        return
    
    # Check if pickle directory exists
    if not os.path.exists('pickle'):
        os.makedirs('pickle')
    
    # Get all CSV files in the csv directory
    csv_files = glob.glob('csv/*.csv')
    
    if not csv_files:
        print("No CSV files found in the csv/ directory.")
        return
    
    # Convert each CSV file to Pickle
    for csv_file in csv_files:
        # Get the base filename without extension
        base_name = os.path.basename(csv_file)[:-4]
        pickle_file = f"pickle/{base_name}.pkl"
        
        # Read CSV and write to Pickle
        try:
            df = pd.read_csv(csv_file)
            df.to_pickle(pickle_file)
            print(f"Converted {csv_file} to {pickle_file}")
        except Exception as e:
            print(f"Error converting {csv_file}: {str(e)}")

if __name__ == "__main__":
    print("Starting setup...")
    create_directories()
    create_config()
    convert_csv_to_pickle()
    print("Setup complete!")
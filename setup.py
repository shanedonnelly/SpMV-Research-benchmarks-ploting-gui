import configparser
import os
import pandas as pd  # type: ignore


def create_config():
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {
        'max_unique': 20,
        'show_df_by_default': True
    }

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def convert_csv_to_pickle():
    csv_folder = "csv/"
    pickle_folder = "pickle/"

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
        print(f"Created folder: {csv_folder}. Please add your CSV files there.")
        return

    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_folder, filename)
            pickle_filename = filename.replace(".csv", ".pkl")
            pickle_path = os.path.join(pickle_folder, pickle_filename)

            try:
                df = pd.read_csv(csv_path)
                df.to_pickle(pickle_path)
                print(f"Converted {csv_path} to {pickle_path}")
            except Exception as e:
                print(f"Error converting {csv_path}: {e}")


if __name__ == "__main__":
    create_config()
    convert_csv_to_pickle()
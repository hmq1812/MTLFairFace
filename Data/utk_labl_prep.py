import os
import csv
from fairface_labl_prep import process_csv

# Define a mapping from the UTKFace dataset race to the FairFace race labels
race_mapping = {
    '0': 'White',
    '1': 'Black',
    '2': 'Asian',
    '3': 'Indian',
    '4': 'Other'
}

# Define a function to map age to FairFace age groups
def map_age_to_group(age):
    if age < 3:
        return '0-2'
    elif age < 10:
        return '3-9'
    elif age < 20:
        return '10-19'
    elif age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return 'more than 70'

# Define a function to map UTKFace gender to FairFace gender labels
def map_gender_to_label(gender):
    return 'Male' if gender == '0' else 'Female'
    
# Process files and write labels to a CSV
def create_label_file(input_dir, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['file', 'age', 'gender', 'race'])
        
        # Walk through the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg'):
                try:
                    # Parse the filename
                    parts = filename.split('_')
                    if len(parts) < 4:
                        print(f"Skipping {filename}: unexpected filename format.")
                        continue
                    
                    age, gender, race, _ = parts[0:4]
                    age_group = map_age_to_group(int(age))
                    gender_label = map_gender_to_label(gender)
                    race_label = race_mapping.get(race, 'Unknown')

                    # Write to CSV
                    writer.writerow([filename, age_group, gender_label, race_label])
                except ValueError as e:
                    print(f"Error processing file {filename}: {e}")
                    continue



if __name__ == "__main__":
    create_label_file('Data/UTKface_Aligned_cropped/UTKFace', 'utk_label_train.csv')
    process_csv('utk_label_train.csv', 'utk_label_train_encoded.csv')

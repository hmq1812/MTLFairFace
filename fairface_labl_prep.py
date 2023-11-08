import csv
import config

# Define the classes
age_classes = config.AGE_CLASSES
gender_classes = config.GENDER_CLASSES
race_classes = config.RACE_CLASSES

# Define a function to map the labels to numbers or -1 if not in classes
def map_labels(row):
    row['age'] = age_classes.index(row['age']) if row['age'] in age_classes else -1
    row['gender'] = gender_classes.index(row['gender']) if row['gender'] in gender_classes else -1
    row['race'] = race_classes.index(row['race']) if row['race'] in race_classes else -1
    return row

# Process the CSV file and write to a new CSV file
def process_csv(input_file_path, output_file_path):
    with open(input_file_path, mode='r') as infile, open(output_file_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # Original field names
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            try:
                mapped_row = map_labels(row)  # Map the labels for the current row
                writer.writerow(mapped_row)  # Write the processed row to the new CSV file
            except ValueError:
                # This except block catches the ValueError raised when index is not found
                # and continues to the next row without writing the current row
                continue

if __name__ == "__main__":
    process_csv('Data/FairFaceData/fairface_label_train.csv', 'Data/FairFaceData/fairface_label_train_encoded.csv')
    process_csv('Data/FairFaceData/fairface_label_val.csv', 'Data/FairFaceData/fairface_label_val_encoded.csv')

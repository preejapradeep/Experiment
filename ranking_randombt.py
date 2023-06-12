directory = ''
# Change the working directory to the location of the file
os.chdir(os.path.dirname(directory))

csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Create an empty DataFrame to store the combined results
combined_results = pd.DataFrame()

for file_name in csv_files:
    file_path = os.path.join(directory, file_name)
    random_bt_results = pd.read_csv(file_path)

    # Filter the rows where 'Random BT' equals 'Case'
    filtered_results = random_bt_results.loc[random_bt_results['Random BT'] == random_bt_results['Case']]

    # Select the desired columns
    filtered_results = filtered_results[['Random BT', 'Rank', 'Structural Similarity', 'Edits', 'Metrics']]
    
    # Append the filtered results to the combined_results DataFrame
    combined_results = pd.concat([combined_results, filtered_results], ignore_index=True)

# Store the combined results into a single CSV file
combined_results.to_csv(directory + f'random_bt_ranking.csv', index=False)

# Save the combined results to an Excel file
combined_results.to_excel(directory + f'random_bt_ranking.xlsx', index=False, engine='openpyxl')

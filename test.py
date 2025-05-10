import reverse_geocoder as rg
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
import multiprocessing
import pycountry  # Add this library to get country names

# Install pycountry if needed:
# pip install pycountry

# Country code to country name mapping
def get_country_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except (AttributeError, KeyError):
        return country_code  # Return the code if lookup fails

# This is crucial for multiprocessing to work correctly
def main():
    # Load data more efficiently (chunksize helps with memory)
    print("Loading transactions data...")
    transactions_df = pd.read_json('data/transactions.json', lines=True)
    print(f"Transactions data loaded: {transactions_df.shape[0]:,} rows")
    
    # Apply the function to add country information
    print("Starting country extraction process...")
    transactions_df = add_country_from_coordinates(transactions_df)
    
    # Drop the location column to save memory
    print("Removing original location data...")
    transactions_df = transactions_df.drop(columns=['location'])
    
    # Save transactions DataFrame to CSV file
    print("Saving results to CSV...")
    transactions_df.to_csv('data/transactions_with_countries.csv', index=False)
    print("Transactions data saved to 'data/transactions_with_countries.csv'")
    
    # Print summary statistics
    print(f"\nExtended transactions shape: {transactions_df.shape}")
    print(f"\nFirst 5 rows of extended transactions:")
    print(transactions_df.head())
    print(f"\nCountry distribution:")
    print(transactions_df['country'].value_counts().head(10))

def add_country_from_coordinates(df, batch_size=1000):
    """
    Add country column to dataframe based on latitude and longitude coordinates
    Using batching for memory efficiency and progress tracking
    Fixed to work with reverse_geocoder's multiprocessing
    """
    print("Extracting countries from coordinates...")
    
    # Pre-allocate the country column
    df['country'] = None
    
    # Extract coordinates once and store in lists for better performance
    print("Extracting coordinates...")
    coords_list = []
    valid_indices = []
    
    for idx, loc in enumerate(df['location']):
        if isinstance(loc, dict) and 'lat' in loc and 'long' in loc:
            lat, long = loc['lat'], loc['long']
            if lat is not None and long is not None:
                coords_list.append((lat, long))
                valid_indices.append(idx)
    
    total_coords = len(coords_list)
    print(f"Found {total_coords:,} valid coordinates")
    
    if total_coords > 0:
        # Process in batches to prevent memory issues
        country_results = [None] * total_coords
        
        # Disable multiprocessing in reverse_geocoder by using mode=1
        print("Processing coordinates in batches...")
        for i in tqdm(range(0, total_coords, batch_size)):
            batch_end = min(i + batch_size, total_coords)
            batch_indices = range(i, batch_end)
            
            # Get country for each coordinate in the batch
            for j, idx in enumerate(batch_indices):
                # Use mode=1 to disable multiprocessing
                result = rg.search(coords_list[idx], mode=1)
                # Get the country code and convert to country name
                country_code = result[0]['cc']
                country_name = get_country_name(country_code)
                country_results[idx] = country_name
                
                # Optional: print progress occasionally
                if j % (batch_size // 10) == 0 and j > 0:
                    print(f"Processed {i+j:,} coordinates...")
        
        # Assign countries back to the dataframe
        print("Assigning countries to dataframe...")
        for idx, country in zip(valid_indices, country_results):
            df.at[idx, 'country'] = country
    
    print(f"Countries extracted. Found {df['country'].notna().sum():,} locations.")
    return df

if __name__ == "__main__":
    # This is critical for multiprocessing to work properly
    multiprocessing.freeze_support()
    main()
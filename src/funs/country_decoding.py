import reverse_geocoder as rg
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
import pycountry
    
# Country code to country name mapping
def get_country_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except (AttributeError, KeyError):
        return country_code  # Return the code if lookup fails

def add_country_from_coordinates(df, batch_size=1000):
    """
    Add country column to dataframe based on latitude and longitude coordinates
    Using batching for memory efficiency and progress tracking
    Fixed to work with reverse_geocoder's multiprocessing
    """
    print("Extracting countries from coordinates...")
    
    # Pre-allocate the country column
    df['transaction_country'] = None
    
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
        
        # Assign countries back to the dataframe
        print("Assigning countries to dataframe...")
        for idx, country in zip(valid_indices, country_results):
            df.at[idx, 'transaction_country'] = country

    print(f"Countries extracted. Found {df['transaction_country'].notna().sum():,} locations.")
    return df

# This should be in your main script, not in the module
def main():
    """Main function to be called when running the script directly"""
    # Load transactions data
    print("Loading transactions data...")
    transactions_df = pd.read_json('../data/transactions.json', lines=True)
    print(f"Transactions data loaded: {transactions_df.shape[0]:,} rows")
    
    # Add country information
    transactions_df = add_country_from_coordinates(transactions_df)
    
    # Drop the location column to save memory
    transactions_df = transactions_df.drop(columns=['location'])
    
    # Save the results
    transactions_df.to_csv('../data/transactions_with_countries.csv', index=False)
    print("Results saved to transactions_with_countries.csv")
    
    return transactions_df

# Use this pattern in your main script
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
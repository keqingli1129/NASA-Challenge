from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table
import pandas as pd
from collections import Counter
import numpy as np
import re

# Function to convert string to integer for sorting
def convert_to_int(x):
    try:
        return (0, int(x))  # Tuple with 0 for numbers
    except ValueError:
        return (1, x)       # Tuple with 1 for strings
    
def mapping_hipparcos_catalog_nasaconfirmed():
    # Load Hipparcos catalog data (with RA, Dec)

    hip_data = Table.read('hipparcos_catalog.fits')

    # Load exoplanet host data with coordinates
    exo_hosts = Table.read('exo_hosts.csv', format='csv', comment='#')

    # Get all hip_name values
    hip_names = exo_hosts['hip_name']

    # Count distinct values (excluding None/nan)
    hip_name_counts = Counter([str(name) for name in hip_names if name])# Create mask for non-null hip_names
    mask = (exo_hosts['hip_name'] != '') & ~exo_hosts['hip_name'].mask
    valid_hip_names = exo_hosts[mask]

    # Clean names by removing 'HIP' prefix
    clean_names = set()
    for name in valid_hip_names['hip_name']:
        if name and not isinstance(name, np.ma.core.MaskedConstant):
            # Extract numeric part after 'HIP', ignoring suffixes
            match = re.search(r'HIP\s*(\d+)', name)
            if match:
                clean_name = match.group(1)  # Just the number as string
                clean_names.add(clean_name)

    # Print results
    print(f"Total number of distinct HIP IDs: {len(clean_names)}")

    # Print sorted list
    print("\nSorted HIP IDs (without 'HIP' prefix):")
    for name in sorted(clean_names, key=convert_to_int):
        print(name)
    # # Print results
    # print("Distinct hip_names and their counts:")
    # for name, count in hip_name_counts.most_common():
    #     print(f"{name}: {count}")

    # # Print total number of distinct valueship_name_counts
    # print(f"\nTotal number of distinct hip_names: {len(hip_name_counts)}")
    # # exo_hosts_df = pd.read_csv('exo_hosts.csv')
    # # exo_hosts = Table.from_pandas(exo_hosts_df)

    # # Create SkyCoord objects for both catalogs

    # # Convert sexagesimal strings to Angle objects (hours/minutes/seconds for RA, degrees/arcmin/arcsec for Dec)
    # # RA is in hours:minutes:seconds, so specify unit='hourangle'
    # ra_angles = Angle(hip_data['RAhms'], unit='hourangle')

    # # Dec is in degrees:arcmin:arcsec, so specify unit='deg'
    # dec_angles = Angle(hip_data['DEdms'], unit='deg')

    # # Create SkyCoord object
    # hip_coords = SkyCoord(ra=ra_angles, dec=dec_angles, frame='icrs')

    # print(hip_coords[:5])
    # exo_coords = SkyCoord(ra=exo_hosts['ra']*u.degree, dec=exo_hosts['dec']*u.degree)

    # # Cross-match with a 2 arcsecond tolerance
    # idx, d2d, _ = exo_coords.match_to_catalog_sky(hip_coords)
    # max_sep = 2 * u.arcsec
    # matches = d2d < max_sep

    # # Matched Hipparcos entries
    # matched_hip = hip_data[idx[matches]]
    # matched_exo = exo_hosts[matches]
    # print(f"Found {len(matched_hip)} matches within {max_sep.to(u.arcsec)}")
    # Show records from exo_hosts where hip_name is '43587'
    # print("Exoplanet hosts with HIP 43587:")
    # mask = exo_hosts['hip_name'] == 'HIP 43587'
    # print(exo_hosts[mask]['pl_name','soltype'])

    # print("Detailed records for 55 Cnc b:")
    # mask = exo_hosts['hip_name'] == 'HIP 43587'
    # # Show more columns to identify differences
    # print(exo_hosts[mask]['pl_name', 'soltype', 'disc_year'])
    #Create mask for non-null hip_names
    # Print sorted list with proper type handling

    # Get valid hip names and remove 'HIP' prefix


# Example usage:
# combined_catalog = combine_toi_koi('TOIs.csv', 'KOIs.csv', 'combined_catalog.csv')
def combine_toi_koi(toi_csv, koi_csv, output_csv, match_radius_arcsec=2.0):
    # Load TOI and KOI catalogs
    toi = pd.read_csv(toi_csv, comment='#')
    koi = pd.read_csv(koi_csv, comment='#')
    # # RA is in hours:minutes:seconds, so specify unit='hourangle'
    ra_angles = Angle(toi['RA'], unit='hourangle')
    # Dec is in degrees:arcmin:arcsec, so specify unit='deg'
    dec_angles = Angle(toi['Dec'], unit='deg')
    # Create SkyCoord object
    toi_coords = SkyCoord(ra=ra_angles, dec=dec_angles, frame='icrs')
    # # RA is in hours:minutes:seconds, so specify unit='hourangle'
    ra_angles = Angle(koi['ra'], unit='hourangle')
    # Dec is in degrees:arcmin:arcsec, so specify unit='deg'
    dec_angles = Angle(koi['dec'], unit='deg')
    # Create SkyCoord object
    koi_coords = SkyCoord(ra=ra_angles, dec=dec_angles, frame='icrs')
    
    # Match KOI coords to TOI coords
    idx, d2d, _ = koi_coords.match_to_catalog_sky(toi_coords)
    
    # Identify KOI entries that are NOT duplicates (distance > match_radius)
    unique_mask = d2d > match_radius_arcsec * u.arcsec
    
    # KOI entries unique to KOI catalog (not matched in TOI)
    koi_unique = koi[unique_mask]
    
    # Combine TOI catalog plus unique KOIs
    combined = pd.concat([toi, koi_unique], ignore_index=True)
    
    # Write to output CSV
    combined.to_csv(output_csv, index=False)
    print(f"Combined catalog saved to {output_csv}")
    
    return combined

def main():
    # mapping_hipparcos_catalog_nasaconfirmed()
    combined_catalog = combine_toi_koi('TOIs.csv', 'KOIs.csv', 'combined_catalog.csv')
if __name__ == "__main__":
    main()
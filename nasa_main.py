from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table
import pandas as pd
from collections import Counter
import numpy as np
import re
import csv

# Function to convert string to integer for sorting
def convert_to_int(x):
    try:
        return (0, int(x))  # Tuple with 0 for numbers
    except ValueError:
        return (1, x)       # Tuple with 1 for strings
    
def mapping_hipparcos_catalog_nasaconfirmed():
    # Load Hipparcos catalog data (not used directly here but can be)
    hip_data = Table.read('hipparcos_catalog.fits')

    # Load exoplanet host data with coordinates and solution type
    exo_hosts = Table.read('exo_hosts.csv', format='csv', comment='#')

    # Extract hip_name and soltype columns
    hip_names = exo_hosts['hip_name']
    soltypes = exo_hosts['soltype']

    # Create mask for non-null, non-empty hip_name
    mask = (exo_hosts['hip_name'] != '') & (~exo_hosts['hip_name'].mask.astype(bool))
    valid_hip_hosts = exo_hosts[mask]

    # Dictionary to hold HIP IDs and corresponding solution types (may have multiple entries)
    hip_soltype_map = {}

    # Regex pattern to extract numeric part after 'HIP' prefix
    pattern = re.compile(r'HIP\s*(\d+)')

    for row in valid_hip_hosts:
        name = row['hip_name']
        soltype = row['soltype']
        if name and not isinstance(name, np.ma.core.MaskedConstant):
            match = pattern.search(name)
            if match:
                clean_name = match.group(1)  # Numeric part as string
                if clean_name in hip_soltype_map:
                    hip_soltype_map[clean_name].add(soltype)
                else:
                    hip_soltype_map[clean_name] = {soltype}

    # Sort the HIP IDs
    sorted_hips = sorted(hip_soltype_map.keys(), key=convert_to_int)

    # Print results
    print(f"Total number of distinct HIP IDs: {len(sorted_hips)}\n")
    print("Sorted HIP IDs (without 'HIP' prefix) and their solution types:")
    for hip_id in sorted_hips:
        soltype_list = ', '.join(sorted(hip_soltype_map[hip_id]))
        print(f"{hip_id}: {soltype_list}")

    # Save to CSV
    with open('hip_ids.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['HIP_ID', 'Solution_Types'])
        for hip_id in sorted_hips:
            soltype_list = ', '.join(sorted(hip_soltype_map[hip_id]))
            writer.writerow([hip_id, 'PC'])

    print(f"\nData saved to {'hip_ids.csv'}")

# Example call:
# mapping_hipparcos_catalog_nasaconfirmed()

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

def map_combined_hipparcos(match_radius_arcsec = 5):
    # Load catalogs
    combined = Table.read('combined_catalog.csv', format='csv', comment='#', encoding='utf-8')
    hip = Table.read('hipparcos_catalog.fits')
    print(hip.colnames)
    # # Clean combined catalog data
    # mask = ~(combined['RA'].mask | combined['Dec'].mask)
    # combined_clean = combined[mask]
    
    # # Clean Hipparcos data - check for null/invalid values instead of masks
    # hip_mask = (hip['RAhms'] != '') & (hip['DEdms'] != '')
    # hip_clean = hip[hip_mask]
    
    # Combined catalog coords - RA and Dec in degrees as floats
    combined_coords = SkyCoord(ra=combined['RA']*u.degree, dec=combined['Dec']*u.degree, frame='icrs')

    # Hipparcos coords - parse sexagesimal strings for RA and Dec
    hip_coords = SkyCoord(ra=hip['RAhms'], dec=hip['DEdms'], unit=(u.hourangle, u.deg), frame='icrs')
    
    # hip_coords_epoch = hip_coords.apply_space_motion(new_obstime='J2015.5')
    # Cross-match catalogs
    idx, d2d, _ = hip_coords.match_to_catalog_sky(combined_coords)
    
    # Filter matches
    match_mask = d2d < match_radius_arcsec * u.arcsec
    
    # Get matched entries
    matched_hip = hip[match_mask]
    matched_combined = combined[idx[match_mask]]
    print(f"Found {len(matched_hip)} matches within {match_radius_arcsec} arcseconds")
    print("\nMatched Hipparcos stars:")
    # Prepare a list of selected fields for output
    output_rows = []

    for hip_star, comb_star in zip(matched_hip, matched_combined):
        output_rows.append({
            'HIP': hip_star['HIP'],
            'RAhms': hip_star['RAhms'],
            'DEdms': hip_star['DEdms'],
            'TFOPWG Disposition': comb_star['TFOPWG Disposition']
        })

    # Convert to pandas DataFrame
    output_df = pd.DataFrame(output_rows)

    # Save to CSV
    output_df.to_csv('matched_hipparcos_filtered.csv', index=False)
     # Output matched Hipparcos entries to a CSV file
    # selected = matched_hip[['HIP', 'RAhms', 'DEdms', 'TFOPWG Disposition']]
    # matched_hip.write('matched_hipparcos.csv', format='csv', overwrite=True)
    print("Matched Hipparcos entries saved to matched_hipparcos.csv")
    return matched_hip

def main():
    mapping_hipparcos_catalog_nasaconfirmed()
    # combined_catalog = combine_toi_koi('TOIs.csv', 'KOIs.csv', 'combined_catalog.csv')
    # map_combined_hipparcos()
if __name__ == "__main__":
    main()
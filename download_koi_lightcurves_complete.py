import pandas as pd
import lightkurve as lk
import os
from time import sleep
import logging

# Set up logging to track progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_koi_lightcurves_complete(csv_file_path, download_dir="./koi_lightcurves", 
                                    author="Kepler", cadence="long", mission="Kepler", 
                                    get_all_available=True, specific_quarters=None, delay=1):
    """
    Download light curves for KOI targets with flexible data selection.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the KOI.csv file
    download_dir : str
        Directory where light curves will be downloaded
    author : str
        Pipeline author ("Kepler", "K2")
    cadence : str
        Cadence type ("long" or "short")
    mission : str
        Mission name ("Kepler" or "K2")
    get_all_available : bool
        If True, download ALL available quarters/sectors. If False, use specific_quarters.
    specific_quarters : list
        List of specific quarters/sectors to download (e.g., [1, 2, 3, 4])
    delay : float
        Delay between downloads in seconds
    """
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        koi_df = pd.read_csv(csv_file_path)
        logger.info(f"Successfully read CSV file with {len(koi_df)} entries")
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return {"error": f"CSV read failed: {e}"}
    
    # Initialize results tracking
    results = {
        "successful": [],
        "failed": [],
        "skipped": [],
        "total_files_downloaded": 0
    }
    
    # Iterate through each KOI entry
    for index, row in koi_df.iterrows():
        # Construct identifiers
        koi_id = f"KOI-{row['kepoi_name']}" if 'kepoi_name' in row else f"KOI-{row['KOI']}"
        kic_id = f"KIC {row['kepid']}" if 'kepid' in row else koi_id
        
        logger.info(f"Processing {koi_id} ({index + 1}/{len(koi_df)})")
        
        # Create a subdirectory for this target
        target_dir = os.path.join(download_dir, koi_id.replace(' ', '_').replace('.', '_'))
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Search for light curves - THIS IS WHERE WE GET "ALL AVAILABLE"
            if get_all_available:
                # Get ALL available data without filtering by quarter
                search_result = lk.search_lightcurve(
                    target=kic_id, 
                    author=author, 
                    cadence=cadence,
                    mission=mission
                )
            else:
                # Get specific quarters only
                search_result = lk.search_lightcurve(
                    target=kic_id, 
                    author=author, 
                    cadence=cadence,
                    mission=mission,
                    quarter=specific_quarters  # This filters to specific quarters
                )
            
            if len(search_result) == 0:
                logger.warning(f"No data found for {koi_id} (KIC: {kic_id})")
                results["failed"].append((koi_id, "No data found"))
                continue
            
            logger.info(f"Found {len(search_result)} observations for {koi_id}")
            
            # Check if we already have all these files
            expected_files = len(search_result)
            existing_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            
            if len(existing_files) >= expected_files:
                logger.info(f"Skipping {koi_id} - already has {len(existing_files)} files")
                results["skipped"].append(koi_id)
                continue
            
            # Download all light curves for this target
            light_curve_collection = search_result.download_all(download_dir=target_dir)
            
            # Count downloaded files
            downloaded_files = [f for f in os.listdir(target_dir) if f.endswith('.fits')]
            new_files = len(downloaded_files) - len(existing_files)
            
            logger.info(f"Successfully downloaded {new_files} new files for {koi_id} (total: {len(downloaded_files)})")
            results["successful"].append((koi_id, new_files))
            results["total_files_downloaded"] += new_files
            
        except Exception as e:
            error_msg = f"Failed to download {koi_id}: {str(e)}"
            logger.error(error_msg)
            results["failed"].append((koi_id, str(e)))
        
        # Add delay to be polite to the servers
        sleep(delay)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("DOWNLOAD SUMMARY:")
    logger.info(f"Successful: {len(results['successful'])} targets")
    logger.info(f"New files downloaded: {results['total_files_downloaded']}")
    logger.info(f"Failed: {len(results['failed'])} targets")
    logger.info(f"Skipped (already downloaded): {len(results['skipped'])} targets")
    logger.info("="*50)
    
    return results

# Example usage scenarios
if __name__ == "__main__":
    
    # Scenario 1: Download ALL available data for all KOIs (DEFAULT)
    results_all = download_koi_lightcurves_complete(
        csv_file_path="KOI.csv",
        download_dir="./koi_all_data",
        get_all_available=True  # This is the default
    )
    
    # Scenario 2: Download only specific quarters (e.g., first 4 quarters)
    results_specific = download_koi_lightcurves_complete(
        csv_file_path="KOI.csv",
        download_dir="./koi_q1-4_data",
        get_all_available=False,
        specific_quarters=[1, 2, 3, 4]  # Only these quarters
    )
    
    # Scenario 3: Download all K2 data instead of Kepler
    results_k2 = download_koi_lightcurves_complete(
        csv_file_path="K2_KOI.csv",
        download_dir="./k2_all_data",
        author="K2",
        mission="K2"
    )
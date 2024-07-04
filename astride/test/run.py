import os
from os.path import join, splitext, isfile, abspath
from astride.detect import Streak
from astride.utils.logger import Logger
from astropy.io import fits

def process_file(file_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    streak = Streak(file_path, output_path=output_path)
    streak.detect()
    streak.write_outputs()
    streak.plot_figures(cut_threshold=5.0)

def test():
    logger = Logger().getLogger()
    logger.info('Start.')

    input_folder = input("Enter the input folder path containing FITS or FIT files: ").strip()
    input_folder = abspath(input_folder)  # Convert to absolute path
    output_base_folder = './testoutput/'

    if not os.path.exists(input_folder):
        logger.error(f'Input folder does not exist: {input_folder}')
        return

    logger.info(f'Absolute path of the input folder: {input_folder}')
    logger.info(f'Listing files in the input folder: {input_folder}')
    
    try:
        file_names = os.listdir(input_folder)
        logger.info(f'Files and directories found: {file_names}')
    except Exception as e:
        logger.error(f'Error accessing the input folder: {e}')
        return

    # Only keep files that are FITS or FIT files
    file_names = [f for f in file_names if isfile(join(input_folder, f)) and f.lower().endswith(('.fits', '.fit'))]
    
    if not file_names:
        logger.error(f'No FITS or FIT files found in the input folder: {input_folder}')
        return
    
    for file_name in file_names:
        file_path = join(input_folder, file_name)
        
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                exposure_time = header.get('EXPOINUS', 0)  # Replace 'EXPTIME' with the correct header keyword if different

            if exposure_time > 50000:
                # Create folder for outputs of each FIT file
                base_name, _ = splitext(file_name)
                output_path = join(output_base_folder, base_name)
                
                logger.info(f'Processing file: {file_name} with output in {output_path}')
                process_file(file_path, output_path)
            else:
                logger.info(f'Skipping file: {file_name} due to exposure time {exposure_time} us')
        except Exception as e:
            logger.error(f'Error processing file {file_name}: {e}')

    logger.info('Done.')
    logger.handlers = []

if __name__ == '__main__':
    test()
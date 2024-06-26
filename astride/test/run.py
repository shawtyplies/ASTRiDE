import os
from os.path import join, splitext
from astride.detect4 import Streak
from astride.utils.logger import Logger

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

    input_folder = input("Enter the input folder path containing FITS files: ").strip()
    output_base_folder = './testoutput/'

    if not os.path.exists(input_folder):
        logger.error(f'Input folder does not exist: {input_folder}')
        return

    file_names = [f for f in os.listdir(input_folder) if f.lower().endswith('.fits')]
    
    if not file_names:
        logger.error(f'No FITS files found in the input folder: {input_folder}')
        return
    
    for file_name in file_names:
        file_path = join(input_folder, file_name)
        
        # Create folder for outputs of each FIT file
        base_name, _ = splitext(file_name)
        output_path = join(output_base_folder, base_name)
        
        logger.info(f'Processing file: {file_name} with output in {output_path}')
        process_file(file_path, output_path)

    logger.info('Done.')
    logger.handlers = []

if __name__ == '__main__':
    test()

import os
from os.path import dirname, join, splitext
from astride.detect3 import Streak
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

    module_path = dirname(__file__)

    file_names = input("Enter the fits file names separated by commas: ").split(',')
    for file_name in file_names:
        file_name = file_name.strip()
        file_path = join(module_path, '../datasets/images', file_name)
        
        # Create folder for outputs of each fits file
        base_name, _ = splitext(file_name)
        output_path = join('./testoutput/', base_name)
        
        logger.info(f'Processing file: {file_name} with output in {output_path}')
        process_file(file_path, output_path)

    logger.info('Done.')
    logger.handlers = []

if __name__ == '__main__':
    test()

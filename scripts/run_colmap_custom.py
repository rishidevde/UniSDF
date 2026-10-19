"""
Custom memory-safe COLMAP runner for large/high-resolution scenes.
"""

import os
import sys
import argparse
import subprocess


def run_colmap(basedir, match_type, max_image_size, use_gpu, num_threads):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    # Feature Extractor with memory limit options
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', os.path.join(basedir, 'images'),
            '--ImageReader.single_camera', '1',
            '--FeatureExtraction.max_image_size', str(max_image_size),
            '--FeatureExtraction.use_gpu', str(use_gpu)
    ]
    print('Running Feature Extractor...')
    print(' '.join(feature_extractor_args))
    feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True)
    logfile.write(feat_output)
    print('Features extracted successfully')

    # Matcher with memory limit options
    matcher_args = [
        'colmap', match_type, 
            '--database_path', os.path.join(basedir, 'database.db'),
            '--FeatureMatching.use_gpu', str(use_gpu),
            '--FeatureMatching.max_num_matches', '8192'
    ]
    if match_type == 'sequential_matcher':
        # Sequential matching specific options (if any) can be added here
        pass
        
    print('Running Feature Matcher...')
    print(' '.join(matcher_args))
    match_output = subprocess.check_output(matcher_args, universal_newlines=True)
    logfile.write(match_output)
    print('Features matched successfully')
    
    p = os.path.join(basedir, 'sparse', '0')
    if not os.path.exists(p):
        os.makedirs(p)
    p = os.path.join(basedir, 'model')
    if not os.path.exists(p):
        os.makedirs(p)

    # Mapper
    mapper_args = [
        'colmap', 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--output_path', os.path.join(basedir, 'model'),
            '--Mapper.num_threads', str(num_threads),
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
    ]
    print('Running Mapper...')
    print(' '.join(mapper_args))
    map_output = subprocess.check_output(mapper_args, universal_newlines=True)
    logfile.write(map_output)
    print('Sparse map created successfully')
    
    # Point Triangulator
    re_triangulator_args = [
        'colmap', 'point_triangulator',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, 'images'),
            '--input_path', os.path.join(basedir, 'model', '0'),
            '--output_path', os.path.join(basedir, 'sparse', '0'),
            '--Mapper.tri_min_angle', '10',
            '--Mapper.tri_merge_max_reproj_error', '1'
    ]
    print('Running Point Triangulator...')
    print(' '.join(re_triangulator_args))
    re_triangulator_output = subprocess.check_output(re_triangulator_args, universal_newlines=True)
    logfile.write(re_triangulator_output)
    logfile.close()
    print('Retriangulation done successfully')
    
    print('Finished running COLMAP, see {} for logs'.format(logfile_name))


def gen_poses(basedir, match_type, max_image_size, use_gpu, num_threads):
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    
    if not all([f in files_had for f in files_needed]):
        print('Need to run COLMAP')
        run_colmap(basedir, match_type, max_image_size, use_gpu, num_threads)
    else:
        print("Don't need to run COLMAP")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_type', type=str, default='exhaustive_matcher',
                        choices=['exhaustive_matcher', 'sequential_matcher'],
                        help='type of matcher used')
    parser.add_argument('--max_image_size', type=int, default=2400,
                        help='maximum image size to avoid GPU out of memory (default: 2400)')
    parser.add_argument('--use_gpu', type=int, default=1, choices=[0, 1],
                        help='whether to use GPU for feature extraction and matching')
    parser.add_argument('--num_threads', type=int, default=16,
                        help='number of threads for mapping')
    parser.add_argument('scenedir', type=str, help='input scene directory')
    args = parser.parse_args()

    # Clear previous failed mapping attempts if any
    sparse_path = os.path.join(args.scenedir, 'sparse')
    model_path = os.path.join(args.scenedir, 'model')
    db_path = os.path.join(args.scenedir, 'database.db')
    
    if os.path.exists(sparse_path):
        import shutil
        print(f"Cleaning up previous sparse folder: {sparse_path}")
        shutil.rmtree(sparse_path)
    if os.path.exists(model_path):
        import shutil
        print(f"Cleaning up previous model folder: {model_path}")
        shutil.rmtree(model_path)
    if os.path.exists(db_path):
        print(f"Cleaning up previous database: {db_path}")
        os.remove(db_path)

    gen_poses(args.scenedir, args.match_type, args.max_image_size, args.use_gpu, args.num_threads)

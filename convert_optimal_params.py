
import os
import logging
import shutil
import subprocess
from argparse import ArgumentParser

# Define a function to run the conversion
def run_conversion(source_path, colmap_executable, magick_executable, camera, no_gpu, skip_matching,
                   guided_matching, estimate_affine_shape, domain_size_pooling, resize):
    colmap_command = '"{}"'.format(colmap_executable) if colmap_executable else "colmap"
    magick_command = '"{}"'.format(magick_executable) if magick_executable else "magick"
    use_gpu = 1 if not no_gpu else 0

    if not skip_matching:
        os.makedirs(source_path + "/distorted/sparse", exist_ok=True)

        ## Feature extraction
        feat_extracton_cmd = (
            f"{colmap_command} feature_extractor "
            f"--database_path {source_path}/distorted/database.db "
            f"--image_path {source_path}/input "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {camera} "
            f"--SiftExtraction.use_gpu {use_gpu} "
            f"--SiftExtraction.estimate_affine_shape {str(estimate_affine_shape).lower()} "
            f"--SiftExtraction.domain_size_pooling {str(domain_size_pooling).lower()} "
            f"--ImageReader.default_focal_length_factor 1.2"
        )
        exit_code = os.system(feat_extracton_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            return exit_code

        ## Feature matching
        feat_matching_cmd = (
            f"{colmap_command} exhaustive_matcher "
            f"--database_path {source_path}/distorted/database.db "
            f"--SiftMatching.use_gpu {use_gpu} "
            f"--SiftMatching.guided_matching {str(guided_matching).lower()}"
        )
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            return exit_code

        ## Bundle adjustment
        mapper_cmd = (
            f"{colmap_command} mapper "
            f"--database_path {source_path}/distorted/database.db "
            f"--image_path {source_path}/input "
            f"--output_path {source_path}/distorted/sparse"
        )
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            return exit_code

    ## Image undistortion
    img_undist_cmd = (
        f"{colmap_command} image_undistorter "
        f"--image_path {source_path}/input "
        f"--input_path {source_path}/distorted/sparse/0 "
        f"--output_path {source_path} "
        f"--output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
        return exit_code

    files = os.listdir(source_path + "/sparse")
    os.makedirs(source_path + "/sparse/0", exist_ok=True)
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(source_path, "sparse", file)
        destination_file = os.path.join(source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    # Create a PLY file from the sparse model
    model_converter_cmd = (
        f"{colmap_command} model_converter "
        f"--input_path {source_path}/sparse/0 "
        f"--output_path {source_path}/sparse/0/points.ply "
        f"--output_type PLY"
    )
    exit_code = os.system(model_converter_cmd)
    if exit_code != 0:
        logging.error(f"Model conversion to PLY failed with code {exit_code}. Exiting.")
        return exit_code

    if resize:
        print("Copying and resizing...")

        # Resize images.
        os.makedirs(source_path + "/images_2", exist_ok=True)
        os.makedirs(source_path + "/images_4", exist_ok=True)
        os.makedirs(source_path + "/images_8", exist_ok=True)
        files = os.listdir(source_path + "/images")
        for file in files:
            source_file = os.path.join(source_path, "images", file)

            destination_file = os.path.join(source_path, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(f"{magick_command} mogrify -resize 50% {destination_file}")
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                return exit_code

            destination_file = os.path.join(source_path, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(f"{magick_command} mogrify -resize 25% {destination_file}")
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                return exit_code

            destination_file = os.path.join(source_path, "images_8", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(f"{magick_command} mogrify -resize 12.5% {destination_file}")
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                return exit_code

    print("Done.")
    return 0

if __name__ == '__main__':
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    parser.add_argument("--guided_matching", type=bool, default=True)
    parser.add_argument("--estimate_affine_shape", type=bool, default=True)
    parser.add_argument("--domain_size_pooling", type=bool, default=True)
    args = parser.parse_args()

    run_conversion(args.source_path, args.colmap_executable, args.magick_executable, args.camera, args.no_gpu,
                   args.skip_matching, args.guided_matching, args.estimate_affine_shape, args.domain_size_pooling, args.resize)

#!/usr/bin/env python3
"""
Convert all HEIF/HEIC files in a directory to PNG format.

Usage:
    python convert_heif2png.py <input_directory> [output_directory]
    
If output_directory is not specified, PNG files will be saved in the same directory as input files.
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import pillow_heif

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()


def convert_heif_to_png(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single HEIF/HEIC file to PNG.
    
    Args:
        input_path: Path to input HEIF/HEIC file
        output_path: Path to output PNG file
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        heif_file = pillow_heif.read_heif(str(input_path))
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image.save(str(output_path), 'PNG')
        print(f'✓ {input_path.name} -> {output_path.name}')
        return True
    except Exception as e:
        print(f'✗ Failed to convert {input_path.name}: {e}')
        return False


def convert_directory(input_dir: str, output_dir: str = None, replace: bool = False) -> None:
    """
    Convert all HEIF/HEIC files in a directory to PNG.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path (optional)
        replace: If True, replace original HEIF files with PNG files
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f'Error: Input directory "{input_dir}" does not exist.')
        sys.exit(1)
    
    if not input_path.is_dir():
        print(f'Error: "{input_dir}" is not a directory.')
        sys.exit(1)
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all HEIF/HEIC files (case-insensitive)
    heif_extensions = ['.heif', '.heic', '.HEIF', '.HEIC']
    heif_files = []
    for ext in heif_extensions:
        heif_files.extend(input_path.glob(f'*{ext}'))
    
    if not heif_files:
        print(f'No HEIF/HEIC files found in "{input_dir}"')
        return
    
    print(f'Found {len(heif_files)} HEIF/HEIC file(s) in "{input_dir}"')
    print('Converting to PNG...\n')
    
    # Sort files for consistent ordering
    heif_files = sorted(heif_files)
    
    success_count = 0
    fail_count = 0
    
    # Determine the number of digits needed for zero-padding
    num_digits = len(str(len(heif_files) - 1))
    
    for idx, heif_file in enumerate(heif_files):
        # Create PNG filename with zero-padded index
        png_filename = f'{idx:0{num_digits}d}.png'
        png_path = output_path / png_filename
        
        # Convert
        if convert_heif_to_png(heif_file, png_path):
            success_count += 1
            # Delete original HEIF file if replace mode is enabled
            if replace and output_path == input_path:
                try:
                    heif_file.unlink()
                    print(f'  Deleted original file: {heif_file.name}')
                except Exception as e:
                    print(f'  Warning: Could not delete {heif_file.name}: {e}')
        else:
            fail_count += 1
    
    print('\nConversion complete!')
    print(f'Success: {success_count}, Failed: {fail_count}')
    
    if output_dir:
        print(f'PNG files saved to: {output_path}')
    else:
        print(f'PNG files saved to: {input_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert HEIF/HEIC files to PNG format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all HEIF files in a directory (save PNG in the same directory)
  python convert_heif2png.py ./images

  # Convert and save to a different directory
  python convert_heif2png.py ./input_images ./output_images

  # Convert and replace original HEIF files with PNG
  python convert_heif2png.py ./images --replace
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing HEIF/HEIC files'
    )
    
    parser.add_argument(
        'output_dir',
        type=str,
        nargs='?',
        default=None,
        help='Output directory for PNG files (optional, defaults to input directory)'
    )
    
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace original HEIF files with PNG files (only works when output_dir is not specified)'
    )
    
    args = parser.parse_args()
    
    if args.replace and args.output_dir:
        print('Warning: --replace option is ignored when output_dir is specified.')
        args.replace = False
    
    convert_directory(args.input_dir, args.output_dir, args.replace)


if __name__ == '__main__':
    main()

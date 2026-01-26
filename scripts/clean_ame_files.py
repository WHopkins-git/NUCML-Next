#!/usr/bin/env python3
"""
AME2020/NUBASE2020 File Cleaning Script
========================================

Cleans AME2020 and NUBASE2020 data files by replacing '#' characters with '.'

The AME2020 format uses '#' to indicate estimated (non-experimental) values
in place of decimal points. For example:
- "12345#67" means "12345.67" (estimated value)
- "12345.67" means "12345.67" (experimental value)

This script creates cleaned versions of all AME files where '#' → '.'

Input files (downloaded from https://www-nds.iaea.org/amdc/):
- mass_1.mas20.txt
- rct1.mas20.txt
- rct2_1.mas20.txt
- nubase_4.mas20.txt
- covariance.mas20.txt (optional)

Output files (cleaned versions):
- mass_1_clean.mas20.txt
- rct1_clean.mas20.txt
- rct2_1_clean.mas20.txt
- nubase_4_clean.mas20.txt
- covariance_clean.mas20.txt (if input exists)

Usage:
    python scripts/clean_ame_files.py --input-dir data/ --output-dir data/

    # Or just use default (cleans in-place with _clean suffix):
    python scripts/clean_ame_files.py
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_ame_file(input_path: Path, output_path: Path) -> Tuple[int, int]:
    """
    Clean a single AME file by replacing '#' with '.'.

    Args:
        input_path: Path to original AME file
        output_path: Path to save cleaned file

    Returns:
        Tuple of (lines_processed, replacements_made)
    """
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return (0, 0)

    logger.info(f"Cleaning {input_path.name}...")

    lines_processed = 0
    replacements_made = 0

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # Count '#' characters in this line
                hash_count = line.count('#')
                replacements_made += hash_count

                # Replace '#' with '.'
                cleaned_line = line.replace('#', '.')

                # Write cleaned line
                outfile.write(cleaned_line)
                lines_processed += 1

    logger.info(f"  ✓ Processed {lines_processed:,} lines, made {replacements_made:,} replacements")
    logger.info(f"  ✓ Saved to {output_path}")

    return (lines_processed, replacements_made)


def main():
    """Main cleaning script."""
    parser = argparse.ArgumentParser(
        description='Clean AME2020/NUBASE2020 files by replacing # with .',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data',
        help='Directory containing original AME files (default: data/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for cleaned files (default: same as input-dir)'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_clean',
        help='Suffix to add to cleaned filenames (default: _clean)'
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("AME2020/NUBASE2020 File Cleaning")
    logger.info("=" * 80)
    logger.info(f"Input directory:  {input_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Filename suffix:  {args.suffix}")
    logger.info("")

    # Files to clean
    ame_files = [
        'mass_1.mas20.txt',
        'rct1.mas20.txt',
        'rct2_1.mas20.txt',
        'nubase_4.mas20.txt',
        'covariance.mas20.txt',  # Optional, large file
    ]

    total_lines = 0
    total_replacements = 0
    files_cleaned = 0

    for filename in ame_files:
        # Construct input/output paths
        input_path = input_dir / filename

        # Generate output filename (e.g., mass_1.mas20.txt → mass_1_clean.mas20.txt)
        # Insert suffix before first extension (.mas20.txt → _clean.mas20.txt)
        if '.mas20.txt' in filename:
            base_name = filename.replace('.mas20.txt', '')
            output_filename = f"{base_name}{args.suffix}.mas20.txt"
        elif filename.endswith('.txt'):
            base_name = filename[:-4]  # Remove .txt
            output_filename = f"{base_name}{args.suffix}.txt"
        else:
            output_filename = f"{filename}{args.suffix}"

        output_path = output_dir / output_filename

        # Clean the file
        lines, replacements = clean_ame_file(input_path, output_path)

        if lines > 0:
            total_lines += lines
            total_replacements += replacements
            files_cleaned += 1

        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("Cleaning Summary")
    logger.info("=" * 80)
    logger.info(f"Files cleaned:       {files_cleaned}")
    logger.info(f"Total lines:         {total_lines:,}")
    logger.info(f"Total replacements:  {total_replacements:,}")
    logger.info("")

    if files_cleaned == 0:
        logger.warning("No files were cleaned. Check that AME files exist in input directory.")
        logger.info("")
        logger.info("To download AME2020 files:")
        logger.info("  cd data/")
        logger.info("  wget https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt")
        logger.info("  wget https://www-nds.iaea.org/amdc/ame2020/rct1.mas20.txt")
        logger.info("  wget https://www-nds.iaea.org/amdc/ame2020/rct2_1.mas20.txt")
        logger.info("  wget https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt")
        logger.info("")
        return 1

    logger.info("✓ Cleaning complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. The enrichment code will automatically use cleaned files")
    logger.info("  2. Run your notebooks normally - they'll load the cleaned data")
    logger.info("  3. Original files are preserved (not modified)")
    logger.info("")

    return 0


if __name__ == '__main__':
    exit(main())

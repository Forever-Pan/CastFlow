#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to Parquet Converter Script

This script converts CSV files to Parquet format.
Supports single file conversion or batch processing of multiple files.

Usage:
    # Convert single file with default settings
    python csv_to_parquet.py --input input.csv
    
    # Convert with custom output path
    python csv_to_parquet.py --input input.csv --output output.parquet
    
    # Convert with specific compression
    python csv_to_parquet.py --input input.csv --compression gzip
    
    # Batch convert all CSV files in directory
    python csv_to_parquet.py --input_dir ./datasets --output_dir ./datasets/parquet
    
    # Convert with all options
    python csv_to_parquet.py --input input.csv --output output.parquet \\
        --compression snappy --engine pyarrow --index
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def convert_csv_to_parquet(
    input_path: str,
    output_path: Optional[str] = None,
    compression: str = "snappy",
    index: bool = False,
    engine: str = "pyarrow",
    encoding: str = "utf-8",
    sep: str = ",",
    header: Optional[int] = 0,
    **read_csv_kwargs,
) -> str:
    """
    Convert a CSV file to Parquet format.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to the output Parquet file (optional, auto-generated if not provided)
        compression: Compression codec (default: 'snappy', options: 'snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'uncompressed')
        index: Whether to write the index (default: False)
        engine: Engine to use for writing (default: 'pyarrow', options: 'pyarrow', 'fastparquet')
        encoding: CSV file encoding (default: 'utf-8')
        sep: CSV delimiter (default: ',')
        header: Row number to use as column names (default: 0, None for no header)
        **read_csv_kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        Path to the output Parquet file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        # Auto-generate output path
        output_path = input_path.with_suffix(".parquet")
    else:
        output_path = Path(output_path)

    print(f"Reading CSV file: {input_path}")
    try:
        read_params = {
            'encoding': encoding,
            'sep': sep,
            'header': header,
            **read_csv_kwargs
        }
        # Remove None values to use pandas defaults
        read_params = {k: v for k, v in read_params.items() if v is not None}
        df = pd.read_csv(input_path, **read_params)
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Column names: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    print(f"\nConverting to Parquet: {output_path}")
    try:
        df.to_parquet(
            output_path,
            compression=compression,
            index=index,
            engine=engine,
        )
        
        # Get file sizes
        input_size = input_path.stat().st_size / (1024 * 1024)  # MB
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
        
        print(f"  - Input size: {input_size:.2f} MB")
        print(f"  - Output size: {output_size:.2f} MB")
        print(f"  - Compression ratio: {compression_ratio:.1f}%")
        print(f"  - Compression: {compression}")
        print(f"  - Engine: {engine}")
        
    except Exception as e:
        raise ValueError(f"Failed to write Parquet file: {e}")

    print(f"\n✓ Conversion completed successfully!")
    return str(output_path)


def batch_convert(
    input_dir: str, 
    output_dir: Optional[str] = None, 
    pattern: str = "*.csv",
    recursive: bool = False,
    **kwargs
):
    """
    Batch convert all CSV files in a directory to Parquet format.

    Args:
        input_dir: Directory containing CSV files
        output_dir: Output directory (optional, uses input_dir if not provided)
        pattern: File pattern to match (default: "*.csv")
        recursive: Whether to search subdirectories recursively (default: False)
        **kwargs: Additional arguments passed to convert_csv_to_parquet
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    if recursive:
        csv_files = list(input_dir.rglob(pattern))
    else:
        csv_files = list(input_dir.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching pattern '{pattern}'")
        return

    print(f"Found {len(csv_files)} CSV file(s) to convert\n")
    
    for csv_file in csv_files:
        output_file = output_dir / csv_file.with_suffix(".parquet").name
        try:
            convert_csv_to_parquet(csv_file, output_file, **kwargs)
            print()
        except Exception as e:
            print(f"✗ Failed to convert {csv_file.name}: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (output auto-generated)
  python csv_to_parquet.py --input input.csv
  
  # Convert with custom output path
  python csv_to_parquet.py --input input.csv --output output.parquet
  
  # Convert with gzip compression
  python csv_to_parquet.py --input input.csv --compression gzip
  
  # Batch convert all CSV files in directory
  python csv_to_parquet.py --input_dir ./datasets --output_dir ./datasets/parquet
  
  # Batch convert recursively with custom pattern
  python csv_to_parquet.py --input_dir ./datasets --recursive --pattern "*.csv"
  
  # Convert with custom CSV options
  python csv_to_parquet.py --input input.csv --encoding utf-8 --sep ","
        """,
    )

    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        default="./datasets/SFT_RL_bank/grok_merge.csv",
        help="Input CSV file path (for single file conversion)",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Input directory path (for batch conversion)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./datasets/SFT_RL_bank/grok_merge.parquet",
        help="Output Parquet file path (for single file) or directory (for batch). "
             "If not specified, output will be auto-generated.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch conversion (alternative to --output)",
    )

    # CSV reading options
    csv_group = parser.add_argument_group("CSV Reading Options")
    csv_group.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="CSV file encoding (default: utf-8)",
    )
    csv_group.add_argument(
        "--sep",
        type=str,
        default=",",
        help="CSV delimiter (default: ',')",
    )
    csv_group.add_argument(
        "--header",
        type=int,
        default=0,
        nargs="?",
        const=0,
        help="Row number to use as column names (default: 0, use --no-header for None)",
    )
    csv_group.add_argument(
        "--no-header",
        action="store_const",
        const=None,
        dest="header",
        help="No header row in CSV file",
    )

    # Parquet writing options
    parquet_group = parser.add_argument_group("Parquet Writing Options")
    parquet_group.add_argument(
        "--compression",
        "-c",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "zstd", "lz4", "uncompressed"],
        help="Compression codec (default: snappy)",
    )
    parquet_group.add_argument(
        "--index",
        action="store_true",
        help="Write DataFrame index to parquet file",
    )
    parquet_group.add_argument(
        "--engine",
        type=str,
        default="pyarrow",
        choices=["pyarrow", "fastparquet"],
        help="Parquet engine to use (default: pyarrow)",
    )

    # Batch processing options
    batch_group = parser.add_argument_group("Batch Processing Options")
    batch_group.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match for batch conversion (default: *.csv)",
    )
    batch_group.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search subdirectories recursively",
    )

    args = parser.parse_args()

    try:
        # Determine if batch or single file mode
        if args.input_dir:
            # Batch mode
            output_dir = args.output_dir or args.output
            batch_convert(
                args.input_dir,
                output_dir,
                pattern=args.pattern,
                recursive=args.recursive,
                compression=args.compression,
                index=args.index,
                engine=args.engine,
                encoding=args.encoding,
                sep=args.sep,
                header=args.header,
            )
        else:
            # Single file mode
            convert_csv_to_parquet(
                args.input,
                args.output,
                compression=args.compression,
                index=args.index,
                engine=args.engine,
                encoding=args.encoding,
                sep=args.sep,
                header=args.header,
            )
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

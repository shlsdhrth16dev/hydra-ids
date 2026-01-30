"""
CSV Merger for ML Data Pipelines

This module provides memory-efficient CSV merging functionality with:
- Chunked processing to minimize memory footprint
- Comprehensive error handling and logging
- Data validation and schema consistency checks
- Progress tracking and detailed reporting
- Optional sampling for memory-constrained environments

Optimized for the CICIDS network intrusion detection dataset.

Author: ML Engineering Team
Date: 2026-01-30
"""

import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime
import sys
from tqdm import tqdm


# Ensure log directory exists before configuring logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'merge_operation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Data class to store per-file metadata."""
    file_name: str
    rows: int
    columns: int
    size_mb: float
    processing_time: float


class CSVMerger:
    """
    Memory-efficient CSV merger optimized for large datasets.
    
    Features:
    - Chunked processing to minimize memory usage
    - Schema validation across files
    - Progress tracking with visual feedback
    - Comprehensive logging and error handling
    - Optional sampling mode
    - Detailed merge reports
    """
    
    def __init__(
        self,
        input_dir: Path = Path("data/raw"),
        output_file: Path = Path("data/processed/cicids_merged.csv"),
        chunk_size: int = 50000,
        sample_size: Optional[int] = None,
        validate_schema: bool = True
    ):
        """
        Initialize the CSV merger.
        
        Args:
            input_dir: Directory containing CSV files to merge
            output_file: Path to save merged CSV file
            chunk_size: Number of rows to process at once
            sample_size: Optional row limit per file (None = use all data)
            validate_schema: Whether to validate schema consistency
        """
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.chunk_size = chunk_size
        self.sample_size = sample_size
        self.validate_schema = validate_schema
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self.file_infos: List[FileInfo] = []
        self.total_rows_merged = 0
        self.expected_columns: Optional[List[str]] = None
        
        logger.info("Initialized CSVMerger")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Chunk size: {self.chunk_size:,}")
        if self.sample_size:
            logger.info(f"Sample mode: {self.sample_size:,} rows per file")
    
    def get_csv_files(self) -> List[Path]:
        """Get sorted list of CSV files to merge."""
        csv_files = sorted(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")
        
        logger.info(f"Found {len(csv_files)} CSV file(s) to merge")
        for file in csv_files:
            size_mb = file.stat().st_size / 1024 / 1024
            logger.info(f"  - {file.name} ({size_mb:.2f} MB)")
        
        return csv_files
    
    def validate_file_schema(self, file_path: Path, columns: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that a file's schema matches the expected schema.
        
        Args:
            file_path: Path to CSV file
            columns: Column names from the file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.expected_columns is None:
            # First file - set as reference
            self.expected_columns = columns
            logger.info(f"Reference schema set from {file_path.name}: {len(columns)} columns")
            return True, None
        
        # Compare with reference schema
        if set(columns) != set(self.expected_columns):
            missing = set(self.expected_columns) - set(columns)
            extra = set(columns) - set(self.expected_columns)
            
            error_msg = f"Schema mismatch in {file_path.name}:\n"
            if missing:
                error_msg += f"  Missing columns: {missing}\n"
            if extra:
                error_msg += f"  Extra columns: {extra}\n"
            
            return False, error_msg
        
        # Check column order
        if columns != self.expected_columns:
            logger.warning(f"{file_path.name} has different column order (will be reordered)")
        
        return True, None
    
    def merge_with_chunks(self, csv_files: List[Path]) -> Dict:
        """
        Merge CSV files using memory-efficient chunked processing.
        
        Args:
            csv_files: List of CSV file paths to merge
            
        Returns:
            Dictionary containing merge statistics
        """
        start_time = datetime.now()
        first_file = True
        
        logger.info("=" * 80)
        logger.info("Starting merge process with chunked processing")
        logger.info("=" * 80)
        
        # Calculate total files for progress bar
        total_files = len(csv_files)
        
        for file_idx, file_path in enumerate(csv_files, 1):
            file_start_time = datetime.now()
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            
            logger.info(f"\n[{file_idx}/{total_files}] Processing {file_path.name} ({file_size_mb:.2f} MB)")
            
            try:
                # First, validate schema by reading header
                header_df = pd.read_csv(file_path, nrows=0)
                columns = header_df.columns.tolist()
                
                if self.validate_schema:
                    is_valid, error_msg = self.validate_file_schema(file_path, columns)
                    if not is_valid:
                        raise ValueError(error_msg)
                
                # Process file in chunks
                rows_processed = 0
                chunk_iterator = pd.read_csv(
                    file_path,
                    chunksize=self.chunk_size,
                    nrows=self.sample_size
                )
                
                # Count total rows for progress bar (approximate if sampling)
                if self.sample_size:
                    estimated_chunks = (self.sample_size // self.chunk_size) + 1
                else:
                    # Quick row count
                    with open(file_path, 'r', encoding='utf-8') as f:
                        estimated_rows = sum(1 for _ in f) - 1  # Exclude header
                    estimated_chunks = (estimated_rows // self.chunk_size) + 1
                
                # Process chunks with progress bar
                with tqdm(total=estimated_chunks, 
                         desc=f"Processing {file_path.name}", 
                         unit="chunk",
                         leave=False) as pbar:
                    
                    for chunk in chunk_iterator:
                        # Reorder columns to match expected schema if needed
                        if self.expected_columns and list(chunk.columns) != self.expected_columns:
                            chunk = chunk[self.expected_columns]
                        
                        # Write chunk to output file
                        mode = 'w' if first_file else 'a'
                        header = first_file
                        
                        chunk.to_csv(
                            self.output_file,
                            mode=mode,
                            header=header,
                            index=False,
                            encoding='utf-8'
                        )
                        
                        first_file = False
                        rows_processed += len(chunk)
                        pbar.update(1)
                
                # Store file metadata
                file_processing_time = (datetime.now() - file_start_time).total_seconds()
                file_info = FileInfo(
                    file_name=file_path.name,
                    rows=rows_processed,
                    columns=len(columns),
                    size_mb=file_size_mb,
                    processing_time=file_processing_time
                )
                self.file_infos.append(file_info)
                self.total_rows_merged += rows_processed
                
                logger.info(f"✓ Processed {rows_processed:,} rows in {file_processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {file_path.name}: {str(e)}", exc_info=True)
                raise
        
        total_processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate statistics
        stats = {
            "total_files": len(csv_files),
            "total_rows": self.total_rows_merged,
            "total_columns": len(self.expected_columns) if self.expected_columns else 0,
            "total_processing_time_seconds": round(total_processing_time, 2),
            "output_file": str(self.output_file),
            "output_size_mb": round(self.output_file.stat().st_size / 1024 / 1024, 2)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("MERGE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total files merged:    {stats['total_files']}")
        logger.info(f"Total rows:            {stats['total_rows']:,}")
        logger.info(f"Total columns:         {stats['total_columns']}")
        logger.info(f"Processing time:       {stats['total_processing_time_seconds']:.2f}s")
        logger.info(f"Output file:           {stats['output_file']}")
        logger.info(f"Output size:           {stats['output_size_mb']:.2f} MB")
        logger.info(f"Avg speed:             {(stats['total_rows'] / stats['total_processing_time_seconds']):,.0f} rows/sec")
        logger.info("=" * 80)
        
        return stats
    
    def generate_report(self, stats: Dict) -> Dict:
        """
        Generate comprehensive merge report.
        
        Args:
            stats: Statistics dictionary from merge operation
            
        Returns:
            Complete report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": stats,
            "files_processed": [
                {
                    "file_name": info.file_name,
                    "rows": info.rows,
                    "columns": info.columns,
                    "size_mb": round(info.size_mb, 2),
                    "processing_time_seconds": round(info.processing_time, 2)
                }
                for info in self.file_infos
            ],
            "configuration": {
                "chunk_size": self.chunk_size,
                "sample_size": self.sample_size,
                "validate_schema": self.validate_schema
            }
        }
        
        return report
    
    def save_report(self, report: Dict, output_path: Path = LOG_DIR / "merge_report.json"):
        """Save merge report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    def merge(self) -> Dict:
        """
        Execute the full merge process.
        
        Returns:
            Dictionary containing merge report
        """
        try:
            # Get CSV files
            csv_files = self.get_csv_files()
            
            # Merge files
            stats = self.merge_with_chunks(csv_files)
            
            # Generate and save report
            report = self.generate_report(stats)
            self.save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Merge failed: {str(e)}", exc_info=True)
            raise


def main():
    """Main execution function."""
    try:
        # Initialize merger with optimized settings
        merger = CSVMerger(
            input_dir=Path("data/raw"),
            output_file=Path("data/processed/cicids_merged.csv"),
            chunk_size=50000,
            sample_size=None,  # Use all data (set to e.g., 100000 for testing)
            validate_schema=True
        )
        
        # Execute merge
        report = merger.merge()
        
        # Success
        logger.info("\n✅ Merge operation completed successfully!")
        logger.info(f"Merged dataset saved to: {merger.output_file}")
        logger.info(f"Logs saved to: {LOG_DIR / 'merge_operation.log'}")
        logger.info(f"Report saved to: {LOG_DIR / 'merge_report.json'}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

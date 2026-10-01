#!/usr/bin/env python3
"""
Parse PKLot parking lot dataset into FiftyOne format.

This script converts the PKLot dataset with XML annotations into a FiftyOne
dataset, preserving parking space polygons, occupancy status, and metadata.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import fiftyone as fo
import fiftyone.core.labels as fol
from PIL import Image


def parse_xml_annotation(xml_path: str, image_width: int, image_height: int) -> Tuple[str, List[fol.Polyline]]:
    """
    Parse a PKLot XML annotation file and extract parking space polylines.
    
    Args:
        xml_path: Path to the XML annotation file
        image_width: Width of the corresponding image for normalization
        image_height: Height of the corresponding image for normalization
    
    Returns:
        Tuple of (parking_lot_id, list of Polyline objects)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get parking lot ID from root element
    parking_lot_id = root.get('id')
    
    polylines = []
    
    # Iterate through each parking space
    for space in root.findall('space'):
        space_id = int(space.get('id'))
        
        # Handle cases where occupied attribute might be missing
        occupied_str = space.get('occupied')
        if occupied_str is not None:
            occupied = int(occupied_str)
            # Map occupied status to string
            occupancy_status = "occupied" if occupied == 1 else "not occupied"
        else:
            # Default to "unknown" if occupied attribute is missing
            occupancy_status = "unknown"
        
        # Extract contour points
        contour = space.find('contour')
        points = []
        
        for point in contour.findall('point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            
            # Normalize coordinates to [0, 1] range
            x_norm = x / image_width
            y_norm = y / image_height
            
            points.append([x_norm, y_norm])
        
        # Create polyline for this parking space
        polyline = fol.Polyline(
            label="parking_space",
            points=[points],  # List of lists, one polygon per parking space
            index=space_id,
            closed=True,
            filled=True,
            occupancy_status=occupancy_status,
            space_id=space_id
        )
        
        polylines.append(polyline)
    
    return parking_lot_id, polylines


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Extract date and full timestamp from the filename.
    
    Args:
        filename: Image/XML filename (e.g., "2012-09-12_06_05_16.jpg")
    
    Returns:
        Dictionary with extracted metadata
    """
    # Remove extension to get the timestamp string
    base_name = os.path.splitext(filename)[0]  # "2012-09-12_06_05_16"
    
    # Parse the full timestamp
    parking_timestamp = datetime.strptime(base_name, "%Y-%m-%d_%H_%M_%S")
    
    # Extract just the date for the date field
    date = parking_timestamp.date()
    
    return {
        "date": date,
        "parking_timestamp": parking_timestamp
    }


def iter_pklot_files(dataset_root: str):
    """
    Generator that yields tuples of (image_path, xml_path, parking_lot, weather) 
    for all valid image-annotation pairs in the PKLot dataset.
    
    Args:
        dataset_root: Root directory of the PKLot dataset
    
    Yields:
        Tuple of (image_path, xml_path, parking_lot, weather)
    """
    # Define the PKLot subdirectory
    pklot_dir = os.path.join(dataset_root, "PKLot", "PKLot")
    
    # Define parking lots and weather conditions
    parking_lots = ["PUCPR", "UFPR04", "UFPR05"]
    weather_conditions = ["Sunny", "Cloudy", "Rainy"]
    
    for parking_lot in parking_lots:
        lot_dir = os.path.join(pklot_dir, parking_lot)
        
        if not os.path.exists(lot_dir):
            print(f"Parking lot directory not found: {lot_dir}")
            continue
        
        for weather in weather_conditions:
            weather_dir = os.path.join(lot_dir, weather)
            
            if not os.path.exists(weather_dir):
                continue  # Skip silently if weather dir doesn't exist
            
            # Walk through all subdirectories in the weather directory
            for root, dirs, files in os.walk(weather_dir):
                # Filter for jpg files
                jpg_files = [f for f in files if f.endswith('.jpg')]
                
                for jpg_file in jpg_files:
                    image_path = os.path.join(root, jpg_file)
                    xml_file = jpg_file.replace('.jpg', '.xml')
                    xml_path = os.path.join(root, xml_file)
                    
                    # Check if corresponding XML exists
                    if not os.path.exists(xml_path):
                        print(f"XML annotation not found for {jpg_file}")
                        continue
                    
                    yield image_path, xml_path, parking_lot, weather


def process_pklot_dataset(dataset_root: str, dataset_name: str = "PKLot") -> fo.Dataset:
    """
    Process the entire PKLot dataset and create a FiftyOne dataset.
    
    Args:
        dataset_root: Root directory of the PKLot dataset
        dataset_name: Name for the FiftyOne dataset
    
    Returns:
        FiftyOne Dataset object
    """
    # Create or load the dataset
    dataset = fo.Dataset(name=dataset_name, overwrite=True, persistent=True)
    
    # Collect all samples in a list first for better performance
    samples = []
    sample_count = 0
    
    # Iterate through all image-annotation pairs using the generator
    for image_path, xml_path, parking_lot, weather in iter_pklot_files(dataset_root):
        # Get image dimensions
        width, height = get_image_dimensions(image_path)
        
        # Parse XML annotation
        source_id, polylines = parse_xml_annotation(xml_path, width, height)
        
        # Parse filename metadata
        filename = os.path.basename(image_path)
        metadata = parse_filename_metadata(filename)
        
        # Create FiftyOne sample
        sample = fo.Sample(
            filepath=image_path,
            source=source_id,
            weather=fol.Classification(label=weather.lower()),
            date=metadata["date"],
            parking_timestamp=metadata["parking_timestamp"],
            parking_spaces=fol.Polylines(polylines=polylines)
        )
        
        # Add sample to list
        samples.append(sample)
        sample_count += 1
        
        # Print progress every 100 samples
        if sample_count % 100 == 0:
            print(f"Processed {sample_count} samples...")
    
    # Add all samples to dataset at once for better performance
    print(f"\nAdding {len(samples)} samples to dataset...")
    dataset.add_samples(samples)
    
    # Compute metadata for all samples (image dimensions, etc.)
    print("Computing metadata...")
    dataset.compute_metadata()
    
    # Add dynamic sample fields for better organization
    print("Adding dynamic sample fields...")
    dataset.add_dynamic_sample_fields()
    
    print(f"\nDataset creation complete!")
    print(f"Total samples: {len(dataset)}")
    print(f"Parking lots: {dataset.distinct('source')}")
    print(f"Weather conditions: {dataset.distinct('weather.label')}")
    
    return dataset


def main():
    """Main function to run the PKLot parser."""
    # Set the path to your PKLot dataset
    dataset_root = "/Users/harpreetsahota/workspace/parking-lot-database-fo"
    
    # Process the dataset
    print("Starting PKLot dataset parsing...")
    dataset = process_pklot_dataset(dataset_root)

if __name__ == "__main__":
    main()

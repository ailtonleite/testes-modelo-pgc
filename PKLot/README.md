---
annotations_creators: []
language: en
size_categories:
- 10K<n<100K
task_categories:
- object-detection
task_ids: []
pretty_name: PKLot
tags:
- fiftyone
- image
- image-classification
dataset_summary: >




  This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 12416
  samples.


  ## Installation


  If you haven't already, install FiftyOne:


  ```bash

  pip install -U fiftyone

  ```


  ## Usage


  ```python

  import fiftyone as fo

  from fiftyone.utils.huggingface import load_from_hub


  # Load the dataset

  # Note: other available arguments include 'max_samples', etc

  dataset = load_from_hub("Voxel51/PKLot")


  # Launch the App

  session = fo.launch_app(dataset)

  ```
license: cc
---

# Dataset Card for PKLot

![image/png](pklot-mq.gif)


PKLot is a robust dataset for parking lot classification containing 12,416 images captured from three different parking lots (PUCPR, UFPR04, UFPR05) under various weather conditions (sunny, cloudy, rainy). Each image includes detailed annotations for individual parking spaces with occupancy status, resulting in approximately 695,900 segmented parking space instances.

This is a [FiftyOne](https://github.com/voxel51/fiftyone) dataset with 12,416 samples.

## Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone
```

## Usage

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load the dataset
# Note: other available arguments include 'max_samples', etc
dataset = load_from_hub("Voxel51/PKLot")

# Launch the App
session = fo.launch_app(dataset)
```

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

The PKLot dataset is a comprehensive parking lot classification dataset designed for computer vision research in parking space detection and occupancy classification. The dataset contains:

- **12,416 high-resolution images** (1280×720 pixels)
- **3 different parking lots**: PUCPR (Pontifícia Universidade Católica do Paraná), UFPR04, and UFPR05 (Federal University of Paraná)
- **3 weather conditions**: Sunny, Cloudy, and Rainy
- **Time-series data**: Images captured at 5-minute intervals throughout different days
- **~695,900 parking space instances**: Each image contains 45-100 annotated parking spaces
- **Rich annotations**: Each parking space includes polygon boundaries and occupancy status

The dataset is particularly valuable for:
- Parking space detection algorithms
- Occupancy classification models
- Temporal analysis of parking patterns
- Weather-robust computer vision systems
- Smart city and intelligent transportation system research

- **Curated by:** Paulo R. L. de Almeida, Luiz S. Oliveira, Alceu S. Britto Jr., Eunelson J. Silva Jr., Alessandro L. Koerich
- **Funded by [optional]:** Federal University of Paraná (UFPR) and Pontifícia Universidade Católica do Paraná (PUCPR)
- **Shared by [optional]:** Vision, Robotics and Imaging Laboratory (VRI) - UFPR
- **Language(s) (NLP):** Not applicable (computer vision dataset)
- **License:** [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/)

### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- **Repository:** [PKLot Official Page](http://web.inf.ufpr.br/vri/databases/parking-lot-database/)
- **Paper:** [Almeida et al., "PKLot – A robust dataset for parking lot classification", Expert Systems with Applications, 2015](http://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf)
- **Download:** [PKLot.tar.gz (4.6GB)](http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz)

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

The PKLot dataset is intended for:

1. **Parking Space Detection**: Training and evaluating algorithms to detect individual parking spaces in aerial/surveillance imagery
2. **Occupancy Classification**: Developing models to classify parking spaces as occupied or vacant
3. **Temporal Analysis**: Studying parking patterns over time and predicting future occupancy
4. **Weather Robustness**: Testing computer vision models under different weather conditions
5. **Smart Parking Systems**: Developing real-time parking availability systems
6. **Benchmark Dataset**: Comparing performance of different parking detection algorithms

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

This dataset should not be used for:

- Identifying individuals or vehicles (images are not high-resolution enough for identification)
- Real-time commercial applications without proper validation
- Training models for different parking lot layouts without additional data
- Applications requiring night-time or low-light conditions (dataset only contains daylight images)

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

### FiftyOne Dataset Fields

Each sample in the FiftyOne dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `filepath` | string | Path to the image file |
| `source` | string | Parking lot identifier (`pucpr`, `ufpr04`, `ufpr05`) |
| `weather` | Classification | Weather condition label (`sunny`, `cloudy`, `rainy`) |
| `date` | date | Date of image capture (YYYY-MM-DD) |
| `parking_timestamp` | datetime | Full timestamp of capture (YYYY-MM-DD HH:MM:SS) |
| `parking_spaces` | Polylines | Collection of parking space polygons |

### Parking Space Annotations (Polylines)

Each parking space polyline contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `label` | string | Always "parking_space" |
| `points` | list | Normalized polygon vertices [[x,y], ...] in [0,1] range |
| `index` | int | Unique parking space ID (1-100) |
| `closed` | bool | True (parking spaces are closed polygons) |
| `filled` | bool | True (for visualization as filled polygons) |
| `occupancy_status` | string | "occupied", "not occupied", or "unknown" |
| `space_id` | int | Parking space identifier |

### Dataset Statistics

- **Total Samples**: 12,416 images
- **Parking Lots Distribution**:
  - PUCPR: ~4,474 images
  - UFPR04: ~3,791 images  
  - UFPR05: ~4,152 images
- **Weather Distribution**:
  - Sunny: ~50% of images
  - Cloudy: ~35% of images
  - Rainy: ~15% of images
- **Temporal Coverage**: September 2012 - April 2013
- **Capture Frequency**: 5-minute intervals

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

The PKLot dataset was created to address the lack of robust, publicly available datasets for parking lot classification research. Key motivations included:

1. **Standardized Benchmark**: Providing a common dataset for comparing parking detection algorithms
2. **Real-World Conditions**: Capturing diverse weather conditions and lighting variations
3. **Temporal Dynamics**: Understanding parking patterns over time
4. **Scale**: Offering sufficient data for training deep learning models
5. **Reproducible Research**: Enabling researchers to compare results on the same dataset

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

The data collection process involved:

1. **Camera Setup**: Fixed surveillance cameras installed at three parking lots
2. **Capture Protocol**: Automatic image capture every 5 minutes during daylight hours
3. **Weather Diversity**: Deliberate collection across different weather conditions
4. **Time Period**: Data collected from September 2012 to April 2013
5. **Image Resolution**: All images captured at 1280×720 pixels
6. **Quality Control**: Manual verification of image quality and weather labels

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

The data was produced by researchers at:
- Federal University of Paraná (UFPR), Brazil
- Pontifícia Universidade Católica do Paraná (PUCPR), Brazil
- Vision, Robotics and Imaging Laboratory (VRI)

### Annotations

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

The annotation process consisted of:

1. **Parking Space Delineation**: Manual marking of parking space boundaries using rotated rectangles and polygons
2. **Occupancy Labeling**: Binary classification (0=vacant, 1=occupied) for each parking space
3. **XML Format**: Annotations stored in XML files with both rotated rectangle and contour representations
4. **Consistency**: Same parking space IDs maintained across all images from the same parking lot
5. **Validation**: Cross-checking of annotations for accuracy

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

Annotations were created by the research team at the Vision, Robotics and Imaging Laboratory (VRI) at UFPR, with quality control and validation performed by multiple team members.

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

The dataset contains surveillance imagery of parking lots but:
- Images are taken from elevated positions at resolution insufficient for personal identification
- No license plates or individual features are distinguishable
- Focus is on parking space occupancy, not vehicle or person identification
- The dataset complies with privacy regulations for public space surveillance

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Known Limitations

1. **Geographic Bias**: All data from two universities in Curitiba, Brazil
2. **Temporal Bias**: Limited to daylight hours (approximately 6 AM to 7 PM)
3. **Seasonal Bias**: Data from September 2012 to April 2013 only
4. **Weather Distribution**: Unbalanced weather conditions (more sunny than rainy days)
5. **Parking Lot Types**: Only university parking lots, may not generalize to other environments
6. **Camera Angles**: Fixed camera positions, limited viewpoint diversity

### Technical Limitations

- No night-time or low-light conditions
- No snow or extreme weather conditions
- Fixed parking space layouts (no dynamic spaces)
- Resolution limitations for fine-grained vehicle classification

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be aware that:

1. **Generalization**: Models trained on this dataset may need adaptation for different geographic locations or parking lot types
2. **Lighting Conditions**: Additional data may be needed for 24-hour operation systems
3. **Real-time Deployment**: Validation on target deployment environment is essential
4. **Privacy Considerations**: Ensure compliance with local regulations when deploying models
5. **Weather Robustness**: Test model performance across all weather conditions in the dataset

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@article{almeida2015pklot,
  title={PKLot--A robust dataset for parking lot classification},
  author={Almeida, Paulo and Oliveira, Luiz S and Silva Jr, Eunelson and Britto Jr, Alceu and Koerich, Alessandro},
  journal={Expert Systems with Applications},
  volume={42},
  number={11},
  pages={4937--4949},
  year={2015},
  publisher={Elsevier}
}
```

**APA:**

Almeida, P. R., Oliveira, L. S., Britto Jr, A. S., Silva Jr, E. J., & Koerich, A. L. (2015). PKLot--A robust dataset for parking lot classification. Expert Systems with Applications, 42(11), 4937-4949.

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

- **Parking Space**: Individual parking slot/bay in a parking lot
- **Occupancy Status**: Binary classification of whether a parking space contains a vehicle
- **Polyline**: Closed polygon defining the boundary of a parking space
- **Rotated Rectangle**: Bounding box with rotation angle for non-axis-aligned parking spaces
- **Normalized Coordinates**: Coordinates scaled to [0,1] range relative to image dimensions

## More Information

For more information about the dataset, visit the [official PKLot page](http://web.inf.ufpr.br/vri/databases/parking-lot-database/) or read the [original paper](http://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf).

## Dataset Card Authors

- Harpreet Sahota (FiftyOne integration and dataset card)
- Original dataset by Paulo R. L. de Almeida et al.

## Dataset Card Contact

For questions about the original PKLot dataset, please contact the Vision, Robotics and Imaging Laboratory at UFPR.
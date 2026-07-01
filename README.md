# OpenRouter Photo Processing

Extract deduplicated OpenStreetMap business tags from photos using AI-powered image analysis.

## Description

This tool analyzes photos to extract business information in OpenStreetMap (OSM) key-value format. It uses the OpenRouter API to intelligently identify and extract structured business data from images, including details like:

- Business name and type
- Address components
- Phone numbers
- Website URLs
- Opening hours
- And more...

All results are deduplicated and formatted according to OpenStreetMap standards.

## Requirements

- Python 3.7+
- Valid OpenRouter API key

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenRouter API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENROUTER_API_KEY=your_key_here`

## Usage

Run the script with an image file or folder:

```bash
python extract_business_osm.py path/to/image.jpg
python extract_business_osm.py path/to/folder/
```

The tool will analyze the images and output OpenStreetMap-formatted business information.

## Dependencies

See `requirements.txt` for full list:
- `openrouter` - API client for OpenRouter
- `python-dotenv` - Environment variable management
- `exifread` - EXIF metadata reading

## Author

Andrew MacKinnon

# Educational Video Generator

An AI-powered tool to generate educational videos with high-quality animations from text prompts.

## Features

- **AI Script Generation**: Creates structured educational scripts with scenes and narration
- **Audio Generation**: Converts script to narrated audio with adjustable speed
- **Manim Animation**: Creates mathematical animations based on script content
- **Video Production**: Combines audio and animations into a polished educational video
- **Performance Optimized**: 
  - Parallel processing for faster rendering
  - Caching system to avoid regenerating content
  - Hardware acceleration where available
- **Customizable**:
  - Adjustable audio speed
  - Multiple detail levels and animation styles
  - Various quality levels for rendering

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install ffmpeg (required for audio/video processing):
   - macOS: `brew install ffmpeg`
   - Ubuntu: `apt-get install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

Basic usage:
```bash
python generate_video.py "quantum computing"
```

Advanced options:
```bash
python generate_video.py "quantum computing" --detail high --style detailed --quality medium --audio-speed 1.3 --parallel
```

### Options

- `--detail {low,medium,high}`: Level of detail in the script (default: high)
- `--style {minimal,standard,detailed}`: Visual style of the animations (default: detailed)
- `--voice {standard,slow}`: Base voice speed for narration (default: standard)
- `--audio-speed AUDIO_SPEED`: Adjust audio playback speed multiplier (default: 1.0)
- `--quality {low,medium,high}`: Rendering quality (default: low)
- `--api-key API_KEY`: OpenAI API key (alternatively set OPENAI_API_KEY env variable)
- `--parallel`: Enable parallel rendering for faster processing
- `--no-cache`: Disable caching (always regenerate content)
- `--fast-model`: Use faster but simpler AI model for script generation
- `--output OUTPUT`: Custom output filename

## Output

The final video will be saved to `output/[topic].mp4`

## Requirements

- Python 3.8 or higher
- FFmpeg
- OpenAI API key (set as environment variable OPENAI_API_KEY or passed with --api-key)

## Production Recommendations

For production use with high-quality output:
```bash
python generate_video.py "your topic" --detail high --style detailed --quality high --parallel --audio-speed 1.1
```

For faster development and testing:
```bash
python generate_video.py "your topic" --detail low --quality low --fast-model
```
#!/usr/bin/env python3
"""
CLI tool to generate educational videos with enhanced animations and content
"""

import os
import sys
import json
import subprocess
import time
import shutil
import requests
import traceback
import multiprocessing
import concurrent.futures
import atexit
import psutil
from pathlib import Path
from tempfile import NamedTemporaryFile
import openai
from gtts import gTTS
import argparse
import signal
from tqdm import tqdm
import numpy as np

# Configuration
OUTPUT_DIR = Path("output")
SCRIPT_DIR = OUTPUT_DIR / "script"
AUDIO_DIR = OUTPUT_DIR / "audio"
ANIMATION_DIR = OUTPUT_DIR / "animation"
VIDEO_DIR = OUTPUT_DIR
TEMP_DIR = OUTPUT_DIR / "temp"
# API_KEY removed: do not hardcode secrets in source code

# Global variables for cleanup
temp_files = []

# Create directories if they don't exist
for directory in [OUTPUT_DIR, SCRIPT_DIR, AUDIO_DIR, ANIMATION_DIR, VIDEO_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True)

def register_temp_file(filepath):
    """Register a temporary file for cleanup."""
    global temp_files
    temp_files.append(filepath)

def cleanup_temp_files():
    """Clean up temporary files."""
    global temp_files
    for file in temp_files:
        try:
            if os.path.exists(file):
                os.unlink(file)
        except Exception:
            pass

# Register the cleanup function to run on exit
atexit.register(cleanup_temp_files)

def signal_handler(sig, frame):
    """Handle keyboard interrupts and other signals."""
    print("\n\nInterrupted. Cleaning up...")
    cleanup_temp_files()
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_system_info():
    """Get system resource information."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    cpu_count = os.cpu_count()
    return {
        'memory_total': mem.total,
        'memory_available': mem.available,
        'memory_percent': mem.percent,
        'disk_total': disk.total,
        'disk_free': disk.free,
        'disk_percent': disk.percent,
        'cpu_count': cpu_count
    }

def log_progress(message):
    """Print a formatted progress message."""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] 🔄 {message}")

def log_success(message):
    """Print a formatted success message."""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] ✅ {message}")

def log_error(message):
    """Print a formatted error message."""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] ❌ {message}")

def log_warning(message):
    """Print a formatted warning message."""
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{current_time}] ⚠️ {message}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    dependencies = {
        "ffmpeg": "ffmpeg -version",
        "manim": f"{sys.executable} -m manim --version"
    }
    
    missing = []
    for dep, cmd in dependencies.items():
        try:
            subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            missing.append(dep)
    
    if missing:
        log_error(f"Missing dependencies: {', '.join(missing)}")
        if "ffmpeg" in missing:
            print("Install ffmpeg: https://ffmpeg.org/download.html")
        if "manim" in missing:
            print(f"Install manim: {sys.executable} -m pip install manim")
        return False
    return True

def generate_script(topic, detail_level="high", use_cache=True, fast_model=False):
    """Generate a script for the given topic using OpenAI, with caching for faster repeat runs."""
    try:
        # Check for cached script first
        script_file = SCRIPT_DIR / f"{topic.replace(' ', '_')}.json"
        if use_cache and script_file.exists():
            log_progress(f"Using cached script for '{topic}'")
            with open(script_file, 'r') as f:
                script = json.load(f)
            log_success(f"Loaded cached script with {len(script)} scenes")
            return script
            
        log_progress(f"Generating script for topic: '{topic}' with {detail_level} detail level")
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Base prompt with scene structure
        base_prompt = f"Create an educational script about '{topic}'."
        
        # Adjust detail level based on parameter
        if detail_level == "high":
            scene_count = "6-8"
            detail_instruction = "Include detailed explanations with examples, analogies, and connections to real-world applications."
        elif detail_level == "medium":
            scene_count = "4-6"
            detail_instruction = "Include clear explanations with some examples and applications."
        else:  # low
            scene_count = "3-4"
            detail_instruction = "Focus on fundamental concepts with simple explanations."
        
        system_message = """You are an expert educational content creator specializing in creating engaging, informative scripts
for educational animations. Your scripts should be clear, accurate, and well-structured with a natural flow between scenes.
Each scene should build on previous content while introducing new concepts in a logical progression."""
        
        user_message = f"""{base_prompt} {detail_instruction}
Format the response as a JSON array of scenes with the structure:
[{{
  "title": "Scene Title",
  "narration": "What to say in this scene (1-3 paragraphs)",
  "visual_description": "Detailed description of what to show visually (be specific about animations, transitions, and visual elements)",
  "duration": "Estimated duration in seconds",
  "key_points": ["List of key points to emphasize in this scene"]
}}]

Include {scene_count} scenes that build on each other.
Make the visual descriptions very specific for animators, with clear guidance on what elements should appear and how they should animate.
Include transitions between scenes for a cohesive flow.
"""
        
        log_progress("Sending request to OpenAI for script generation...")
        start_time = time.time()
        
        # Choose model based on speed preference
        model = "gpt-3.5-turbo" if fast_model else "gpt-4"
        log_progress(f"Using {model} for script generation")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        
        generation_time = time.time() - start_time
        script_text = response.choices[0].message.content
        
        # Extract just the JSON part if there's any additional text
        script_text = script_text.strip()
        if "```json" in script_text:
            script_text = script_text.split("```json")[1].split("```")[0]
        elif "```" in script_text:
            script_text = script_text.split("```")[1].split("```")[0]
        
        script = json.loads(script_text)
        
        # Save the script to a file
        with open(script_file, 'w') as f:
            json.dump(script, f, indent=2)
        
        log_success(f"Generated script with {len(script)} scenes using {model} in {generation_time:.2f} seconds")
        return script
    except Exception as e:
        log_error(f"Error generating script: {e}")
        # Use a basic fallback script if generation fails
        log_progress("Using fallback script template...")
        fallback_script = [
            {
                "title": f"Introduction to {topic.title()}",
                "narration": f"Let's explore the fascinating concept of {topic}.",
                "visual_description": f"Title screen showing '{topic.title()}' with a simple icon that represents the topic. The title should animate in with a fade-in effect.",
                "duration": "10",
                "key_points": ["Introduction to the topic", "Setting expectations"]
            },
            {
                "title": f"Key Concepts of {topic.title()}",
                "narration": f"There are several important aspects to understand about {topic}. These fundamental principles help us grasp the full scope of the subject.",
                "visual_description": "Animated bullet points appearing one by one with icons next to each point. Use smooth transitions between points.",
                "duration": "15",
                "key_points": ["Core principles", "Fundamental concepts"]
            },
            {
                "title": "Applications",
                "narration": f"{topic.title()} has many practical applications in the real world. Let's look at how this concept applies in various contexts.",
                "visual_description": "Split screen showing 3-4 different application scenarios with simple animations demonstrating each use case.",
                "duration": "15",
                "key_points": ["Real-world applications", "Practical uses"]
            },
            {
                "title": "Conclusion",
                "narration": f"In summary, {topic} is a fascinating subject with much to explore. The concepts we've covered today provide a foundation for deeper understanding.",
                "visual_description": "Animated recap of key points with icons from previous scenes floating in and arranging into a cohesive diagram. End with a final title card and a subtle call to action.",
                "duration": "10",
                "key_points": ["Summary of key points", "Closing thoughts"]
            }
        ]
        
        # Save the fallback script to a file
        script_file = SCRIPT_DIR / f"{topic.replace(' ', '_')}.json"
        with open(script_file, 'w') as f:
            json.dump(fallback_script, f, indent=2)
            
        log_success(f"Created fallback script with {len(fallback_script)} scenes")
        return fallback_script

# ... rest of the code remains unchanged ...

def generate_audio(script, topic, voice_type="standard", audio_speed=1.0):
    """Generate audio for each scene's narration using gTTS with speed control and enhanced quality."""
    log_progress("Generating audio narration...")
    audio_files = []
    audio_dir = AUDIO_DIR / topic.replace(' ', '_')
    audio_dir.mkdir(exist_ok=True)
    
    # Voice parameters
    if voice_type == "slow":
        slow_option = True
    else:
        slow_option = False
    
    total_scenes = len(script)
    processed_scenes = 0
    
    def generate_audio_for_scene(i, scene):
        """Generate audio for a single scene with enhanced quality and timing."""
        scene_title = scene.get("title", f"Scene {i+1}")
        raw_audio_file = audio_dir / f"scene_{i}_raw.mp3"
        processed_audio_file = audio_dir / f"scene_{i}.mp3"
        temp_adjusted_file = audio_dir / f"scene_{i}_adjusted.mp3"
        
        # Break narration into sentences for more natural pauses
        narration = scene["narration"]
        
        # Generate the audio file
        try:
            tts = gTTS(text=narration, lang="en", slow=slow_option)
            tts.save(str(raw_audio_file))
            
            # Create a more natural pace with slight pauses after sentences
            # This helps with the synchronization
            try:
                # First, adjust speed if needed
                if audio_speed != 1.0:
                    # Use ffmpeg to adjust audio speed
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(raw_audio_file),
                        "-filter:a", f"atempo={audio_speed}",
                        "-vn", str(temp_adjusted_file)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    # Use the speed-adjusted version for further processing
                    input_file = str(temp_adjusted_file)
                else:
                    input_file = str(raw_audio_file)
                
                # Now add subtle enhancements to make it sound more natural
                # Apply a slight bass boost and dynamic compression to improve clarity
                subprocess.run([
                    "ffmpeg", "-y", "-i", input_file,
                    "-af", "equalizer=f=100:width_type=o:width=2:g=1.5, " + 
                           "equalizer=f=400:width_type=o:width=2:g=0.5, " + 
                           "acompressor=threshold=0.089:ratio=9:attack=200:release=1000:makeup=2",
                    "-c:a", "libmp3lame", "-q:a", "2",  # Higher quality encoding
                    str(processed_audio_file)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Remove temporary files
                if os.path.exists(str(raw_audio_file)):
                    os.unlink(str(raw_audio_file))
                if os.path.exists(str(temp_adjusted_file)):
                    os.unlink(str(temp_adjusted_file))
                
                return processed_audio_file
                
            except Exception as e:
                log_warning(f"Audio enhancement failed for scene {i+1}, using basic audio: {e}")
                if audio_speed != 1.0:
                    # Fallback to basic speed adjustment
                    subprocess.run([
                        "ffmpeg", "-y", "-i", str(raw_audio_file),
                        "-filter:a", f"atempo={audio_speed}",
                        "-vn", str(processed_audio_file)
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Remove raw audio
                    os.replace(str(processed_audio_file), str(processed_audio_file))
                else:
                    # Just use the raw file
                    os.replace(str(raw_audio_file), str(processed_audio_file))
                
                return processed_audio_file
        except Exception as e:
            log_error(f"Error generating audio for scene {i+1}: {e}")
            # As a last resort, create a silent audio file with appropriate duration
            try:
                # Create a silent audio file based on estimated duration from scene text
                # Roughly 150 words per minute for narration
                word_count = len(narration.split())
                estimated_duration = max(3, (word_count / 150) * 60)  # At least 3 seconds
                
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
                    "-t", str(estimated_duration),
                    "-c:a", "libmp3lame", "-q:a", "2",
                    str(processed_audio_file)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                log_warning(f"Created silent fallback audio for scene {i+1} with duration {estimated_duration:.1f}s")
                return processed_audio_file
            except:
                log_error(f"Failed to create even silent audio for scene {i+1}")
                return None
    
    # Use parallel processing to generate audio files
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), total_scenes)) as executor:
        # Submit all tasks and create a future-to-index mapping
        future_to_index = {
            executor.submit(generate_audio_for_scene, i, scene): i 
            for i, scene in enumerate(script)
        }
        
        for future in concurrent.futures.as_completed(future_to_index):
            processed_scenes += 1
            scene_index = future_to_index[future]
            try:
                audio_file = future.result()
                if audio_file:
                    audio_files.append(audio_file)
                    scene = script[scene_index]
                    
                    # Get audio duration for better timing
                    try:
                        probe_result = subprocess.run([
                            "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_file)
                        ], capture_output=True, text=True, check=True)
                        
                        duration = float(probe_result.stdout.strip())
                        duration_msg = f" ({duration:.1f}s)"
                    except:
                        duration_msg = ""
                    
                    log_progress(f"Generated audio for scene {scene_index+1}/{total_scenes}: "
                                f"{scene.get('title', f'Scene {scene_index+1}')}"
                                f"{duration_msg} ({processed_scenes}/{total_scenes})")
            except Exception as e:
                log_error(f"Error processing audio for scene {scene_index+1}: {e}")
    
    # Sort audio_files by scene index to maintain correct order
    audio_files = [f for f in audio_files if f]  # Filter out any None values
    audio_files.sort(key=lambda f: int(f.stem.split('_')[1]))
    
    log_success(f"Generated {len(audio_files)} audio files")
    return audio_files

# ... rest of the code remains unchanged ...

def main():
    parser = argparse.ArgumentParser(description="Generate educational videos with OpenAI")
    parser.add_argument("topic", help="The topic for the educational video")
    parser.add_argument("--detail", choices=["low", "medium", "high"], default="high",
                        help="Level of detail in the script (default: high)")
    parser.add_argument("--style", choices=["minimal", "standard", "detailed"], default="detailed",
                        help="Visual style of the animations (default: detailed)")
    parser.add_argument("--voice", choices=["standard", "slow"], default="standard",
                        help="Voice speed for narration (default: standard)")
    parser.add_argument("--audio-speed", type=float, default=1.0,
                        help="Adjust audio playback speed (e.g., 1.5 for 50%% faster, 0.8 for 20%% slower)")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="low",
                        help="Rendering quality (default: low for faster results)")
    parser.add_argument("--api-key", help="OpenAI API key (alternatively set OPENAI_API_KEY env variable)")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel rendering of animation scenes")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of script and animations")
    parser.add_argument("--fast-model", action="store_true", help="Use faster but simpler AI model for script generation")
    parser.add_argument("--output", help="Custom output filename (without extension)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    topic = args.topic.lower()
    use_cache = not args.no_cache
    
    # Start timing the entire process
    start_time = time.time()
    
    log_progress(f"Starting video generation for topic: '{topic}'")
    log_progress(f"Using detail level: {args.detail}, style: {args.style}, voice: {args.voice}, quality: {args.quality}")
    log_progress(f"Audio speed: {args.audio_speed}x, Parallel: {args.parallel}, Cache: {use_cache}")
    
    # Step 1: Generate the script
    script_start = time.time()
    script = generate_script(topic, detail_level=args.detail, use_cache=use_cache, fast_model=args.fast_model)
    script_time = time.time() - script_start
    log_progress(f"Script generation completed in {script_time:.2f} seconds")
    
    # Step 2: Generate the audio narration
    audio_start = time.time()
    audio_files = generate_audio(script, topic, voice_type=args.voice, audio_speed=args.audio_speed)
    audio_time = time.time() - audio_start
    log_progress(f"Audio generation completed in {audio_time:.2f} seconds")
    
    # Step 3: Generate and render Manim animations
    animation_start = time.time()
    manim_file = generate_manim_code(script, topic, style=args.style)
    
    success = render_manim_animations(manim_file, topic, quality=args.quality, use_cache=use_cache, parallel=args.parallel)
    animation_time = time.time() - animation_start
    log_progress(f"Animation generation completed in {animation_time:.2f} seconds")
    
    if not success:
        log_error("Failed to render Manim animations. Using fallback animations.")
    
    # Step 4: Combine audio and video
    video_start = time.time()
    # Use custom output name if provided
    if args.output:
        output_path = VIDEO_DIR / f"{args.output}.mp4"
        # Create a temporary symlink to ensure our file handling works correctly
        original_path = VIDEO_DIR / f"{topic.replace(' ', '_')}.mp4"
        final_video = combine_audio_and_video(audio_files, topic)
        if final_video and final_video.exists():
            shutil.copy(final_video, output_path)
            final_video = output_path
    else:
        final_video = combine_audio_and_video(audio_files, topic)
    
    video_time = time.time() - video_start
    log_progress(f"Video combination completed in {video_time:.2f} seconds")
    
    total_time = time.time() - start_time
    
    if final_video:
        log_success(f"🎬 Video generation complete! Final video is at: {final_video}")
        log_success(f"Total process completed in {total_time:.2f} seconds")
        
        # Print performance summary
        log_progress("Performance summary:")
        log_progress(f"Script generation: {script_time:.2f}s ({script_time/total_time*100:.1f}%)")
        log_progress(f"Audio generation: {audio_time:.2f}s ({audio_time/total_time*100:.1f}%)")
        log_progress(f"Animation generation: {animation_time:.2f}s ({animation_time/total_time*100:.1f}%)")
        log_progress(f"Video combination: {video_time:.2f}s ({video_time/total_time*100:.1f}%)")
    else:
        log_error("Failed to create final video.")

if __name__ == "__main__":
    main() 

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
API_KEY = "ubefmowbkdyegfojfify38ryuhfjhef"


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

def generate_manim_code(script, topic, style="detailed"):
    """Generate Manim code for visualizing the script using OpenAI."""
    log_progress("Generating Manim animation code...")
    animation_dir = ANIMATION_DIR / topic.replace(' ', '_')
    animation_dir.mkdir(exist_ok=True)
    
    # Generate Manim Python code file
    manim_file = animation_dir / f"{topic.replace(' ', '_')}_animation.py"
    
    try:
        log_progress("Using OpenAI for animation code generation...")
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        system_prompt = """You are an expert in creating educational animations with the Manim library.
Generate a SINGLE Python file using Manim to create a professional, high-quality educational video animation.

Key Requirements:
1. Create ONE main scene class called 'VideoScene' that contains all animations
2. Use 'from manim import *' for imports
3. Each animation must be polished and professional looking:
   - Use smooth transitions between elements
   - Implement proper spacing and layout
   - Use color gradients and highlights for emphasis
   - Add subtle motion to keep visuals engaging
   - Use scale and position animations effectively
4. For text elements:
   - Use Text() with appropriate font sizes (36-48 for titles, 24-32 for content)
   - Implement proper text animations (Write, FadeIn with shift)
   - Use color to emphasize important points
   - Add subtle animations to keep text engaging
5. For bullet points and lists:
   - Create custom bullet points using circles/dots
   - Animate each point sequentially
   - Add subtle animations to maintain interest
   - Use proper spacing and alignment
6. For diagrams and visuals:
   - Create clean, professional-looking shapes
   - Use animations to build diagrams piece by piece
   - Add motion to keep diagrams engaging
   - Implement proper color schemes
7. Timing and Pacing:
   - Add appropriate wait times between animations (2-3 seconds for reading)
   - Use smooth transitions between sections
   - Sync animations with typical narration speed
   - Include proper pauses for emphasis
8. Scene Structure:
   - Start with an engaging title sequence
   - Transition smoothly between topics
   - Use clear section headers
   - End with a professional conclusion

Example structure for high-quality animations:
```python
from manim import *

class VideoScene(Scene):
    def construct(self):
        # Title sequence
        title = Text("Topic", font_size=48, color=BLUE)
        subtitle = Text("Subtitle", font_size=36, color=BLUE_B)
        subtitle.next_to(title, DOWN)
        self.play(
            Write(title, run_time=1.5),
            FadeIn(subtitle, shift=UP*0.5)
        )
        self.wait(2)

        # Transition to content
        self.play(
            title.animate.scale(0.8).to_edge(UP),
            FadeOut(subtitle)
        )

        # Create engaging bullet points
        points = VGroup()
        for i, text in enumerate(["Point 1", "Point 2"]):
            dot = Dot(color=BLUE)
            content = Text(text, font_size=28)
            content.next_to(dot, RIGHT, buff=0.2)
            point = VGroup(dot, content)
            point.move_to(LEFT*2 + DOWN*(i-1))
            points.add(point)
            self.play(
                Create(dot),
                Write(content),
                run_time=1
            )
            self.wait(0.5)

        # Add visual element with animation
        visual = Circle(radius=2, color=BLUE)
        visual.next_to(points, RIGHT, buff=1)
        self.play(
            Create(visual),
            points.animate.scale(0.9)
        )
        self.wait(2)
```

Make the code completely self-contained and ready to run.
REMEMBER: Focus on creating PROFESSIONAL and ENGAGING animations that maintain viewer interest."""

        # Format the script in a way that's clear for the API
        script_description = json.dumps(script, indent=2)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a professional Manim animation for a video about '{topic}' with this script:\n\n{script_description}\n\nGenerate complete, runnable Manim code with a SINGLE VideoScene class that implements all scenes in sequence. Focus on creating ENGAGING and PROFESSIONAL animations that maintain viewer interest. Use smooth transitions, proper timing, and visual appeal. Create bullet points manually using dots and text. Use color gradients, motion, and proper spacing to keep the visuals interesting."}
            ],
            temperature=0.7,
        )
        
        manim_code = response.choices[0].message.content
        
        # Extract just the code part if there's any additional text
        if "```python" in manim_code:
            manim_code = manim_code.split("```python")[1].split("```")[0]
        elif "```" in manim_code:
            manim_code = manim_code.split("```")[1].split("```")[0]
        
        # Basic validation and correction
        manim_code = manim_code.strip()
        
        # Ensure correct imports
        if "from manimlib import" in manim_code:
            manim_code = manim_code.replace("from manimlib import", "from manim import")
        
        if "from manim import" not in manim_code:
            manim_code = "from manim import *\n\n" + manim_code
            
        with open(manim_file, 'w') as f:
            f.write(manim_code)
        
        log_success(f"Generated Manim code with OpenAI and saved to {manim_file}")
        return manim_file
    except Exception as e:
        log_error(f"Error generating Manim code with OpenAI: {e}")
        create_fallback_manim_code(manim_file, topic, script)
        log_success(f"Created fallback Manim code and saved to {manim_file}")
        return manim_file

def create_fallback_manim_code(manim_file, topic, script):
    """Create a simple but visually appealing fallback Manim code that renders quickly."""
    log_progress("Creating optimized fallback animation code...")
    
    basic_manim_code = f"""from manim import *

class VideoScene(Scene):
    def construct(self):
        # Title
        title = Text("{topic.title()}", font_size=48, color=BLUE)
        subtitle = Text("Educational Video", font_size=32, color=LIGHT_GREY)
        subtitle.next_to(title, DOWN)
        
        self.play(
            Write(title),
            FadeIn(subtitle)
        )
        self.wait(1)
        
        # Process each scene
        for scene in {script}:
            # Clear previous
            self.clear()
            
            # Scene title
            scene_title = Text(scene["title"], font_size=36, color=BLUE)
            self.play(Write(scene_title))
            scene_title.to_edge(UP)
            
            # Key points with nice formatting
            points = scene.get("key_points", [])
            if points:
                bullet_points = VGroup()
                for i, point in enumerate(points):
                    bullet = Dot(color=YELLOW)
                    text = Text(point, font_size=24, color=WHITE)
                    bullet_point = VGroup(bullet, text)
                    text.next_to(bullet, RIGHT, buff=0.2)
                    bullet_point.move_to(ORIGIN + UP * (1.5 - i))
                    bullet_points.add(bullet_point)
                
                self.play(
                    *[FadeIn(point) for point in bullet_points],
                    lag_ratio=0.5
                )
            
            # Simple visual element based on topic
            visual = Circle(radius=2, color=BLUE)
            if "neural" in "{topic.lower()}":
                # Create a simple neural network visualization
                nodes = VGroup(*[Dot() for _ in range(6)])
                nodes.arrange_in_grid(rows=2, cols=3)
                edges = VGroup()
                for i, n1 in enumerate(nodes):
                    for j, n2 in enumerate(nodes):
                        if i < j and abs(i-j) <= 3:
                            edge = Line(n1.get_center(), n2.get_center(), color=BLUE_A)
                            edges.add(edge)
                visual = VGroup(edges, nodes)
            
            visual.scale(0.5)
            visual.to_edge(RIGHT)
            
            self.play(Create(visual))
            
            # Wait based on scene duration
            duration = float(scene.get("duration", 10))
            self.wait(max(1, duration - 2))  # Subtract animation time
        
        # Final scene
        self.clear()
        final_text = Text("Thank you for watching!", font_size=36, color=BLUE)
        self.play(Write(final_text))
        self.wait(1)

if __name__ == "__main__":
    VideoScene().render()
"""
    
    with open(manim_file, 'w') as f:
        f.write(basic_manim_code)
    
    log_success("Created optimized Manim code for faster rendering")

def render_manim_animations(manim_file, topic, quality="low", use_cache=False, parallel=False):
    """Render the Manim animations with maximum hardware utilization."""
    try:
        # Get system info for optimization
        cpu_count = os.cpu_count()
        mem = psutil.virtual_memory()
        
        # Quality settings with hardware optimization
        quality_flag = "-ql"  # Default to low quality for speed
        if quality == "medium":
            quality_flag = "-qm"
        elif quality == "high":
            quality_flag = "-qh"
        
        # Ensure output directory exists
        output_dir = ANIMATION_DIR / topic.replace(' ', '_')
        output_dir.mkdir(exist_ok=True)
        
        # Get the absolute paths
        manim_file_abs = os.path.abspath(manim_file)
        output_dir_abs = os.path.abspath(output_dir)
        
        # Get the animation file name without extension
        anim_file_name = os.path.splitext(os.path.basename(manim_file))[0]
        
        # Build optimized command with only supported flags
        cmd = [
            sys.executable,
            "-m", "manim",
            str(manim_file_abs),
            "VideoScene",  # Specify the scene class to render
            quality_flag,
            "--format=mp4",          # Force MP4 output
            "--output_file", topic.replace(' ', '_'),  # Set output filename
            "--media_dir", str(output_dir_abs),  # Explicitly set media directory
            "--log_to_file",  # Log to file instead of stdout
            "-v", "DEBUG",    # Increase verbosity for debugging
        ]
        
        if not use_cache:
            cmd.append("--disable_caching")
        
        log_progress(f"Rendering with hardware acceleration (CPUs: {cpu_count}, Memory: {mem.available / (1024*1024*1024):.1f}GB available)")
        log_progress(f"Running Manim command: {' '.join(cmd)}")
        
        # Set process priority to high
        try:
            p = psutil.Process(os.getpid())
            if os.name == 'nt':  # Windows
                p.nice(psutil.HIGH_PRIORITY_CLASS)
            else:  # Unix-like
                os.nice(-10)  # Higher priority
        except:
            pass
            
        # Run Manim with output capture
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log the output for debugging
        if result.stdout:
            log_progress("Manim stdout:")
            log_progress(result.stdout)
        if result.stderr:
            log_progress("Manim stderr:")
            log_progress(result.stderr)
            
        if result.returncode != 0:
            log_error(f"Error rendering Manim animations: {result.stderr}")
            return False
        
        # Find the output video using standard Manim output paths
        target_video = ANIMATION_DIR / f"{topic.replace(' ', '_')}.mp4"
        
        # Check common Manim output locations with more detailed logging
        possible_paths = [
            target_video,
            # New Manim output structure (most likely location based on logs)
            output_dir / "videos" / anim_file_name / "480p15" / f"{topic.replace(' ', '_')}.mp4",
            output_dir / "videos" / anim_file_name / "720p30" / f"{topic.replace(' ', '_')}.mp4",
            output_dir / "videos" / anim_file_name / "1080p60" / f"{topic.replace(' ', '_')}.mp4",
            # Alternative locations
            output_dir / "videos" / "VideoScene" / "480p15" / f"{topic.replace(' ', '_')}.mp4",
            output_dir / "videos" / "VideoScene" / "720p30" / f"{topic.replace(' ', '_')}.mp4",
            output_dir / "videos" / "VideoScene" / "1080p60" / f"{topic.replace(' ', '_')}.mp4",
            # Legacy locations
            Path("media") / "videos" / "VideoScene" / "480p15" / "VideoScene.mp4",
            Path("media") / "videos" / "VideoScene" / "720p30" / "VideoScene.mp4",
            Path("media") / "videos" / "VideoScene" / "1080p60" / "VideoScene.mp4",
        ]
        
        # Log all possible paths we're checking
        log_progress("Checking for rendered video in the following locations:")
        for path in possible_paths:
            log_progress(f"- {path}")
            if path.exists():
                log_progress(f"Found video at: {path}")
                # Move to standard location if needed
                if path != target_video:
                    os.makedirs(os.path.dirname(target_video), exist_ok=True)
                    shutil.copy(str(path), str(target_video))
                    log_success(f"Copied video to final location: {target_video}")
                return True
        
        # If we get here, we didn't find the video
        log_error("No rendered video found in any of the expected locations")
        # List contents of output directory for debugging
        log_progress("Contents of output directory:")
        for root, dirs, files in os.walk(output_dir):
            for name in files:
                if name.endswith('.mp4'):  # Only show video files
                    log_progress(f"- {os.path.join(root, name)}")
        
        return False
    
    except Exception as e:
        log_error(f"Unexpected error in render_manim_animations: {e}")
        traceback.print_exc()  # Print full traceback for debugging
        return False

def combine_audio_and_video(audio_files, topic, scene_durations=None):
    """Combine audio files with the animated scenes using hardware acceleration."""
    log_progress("Combining audio and video files with hardware acceleration...")
    
    if not audio_files:
        log_error("No audio files provided for combining")
        return None
    
    try:
        # Get system info for optimization
        cpu_count = os.cpu_count()
        mem = psutil.virtual_memory()
        
        # Determine hardware acceleration codec based on platform
        if sys.platform == "darwin":  # macOS
            hw_accel = ["-c:v", "h264_videotoolbox"]
        elif sys.platform == "win32":  # Windows
            hw_accel = ["-c:v", "h264_nvenc"]  # NVIDIA GPU
        else:  # Linux
            hw_accel = ["-c:v", "h264_vaapi"]  # Intel GPU
            
        # Create optimized combined audio first
        combined_audio = AUDIO_DIR / f"{topic.replace(' ', '_')}_combined.wav"
            
        # Use concat demuxer for more efficient audio combining
        concat_file = TEMP_DIR / "concat.txt"
        with open(concat_file, 'w') as f:
            for audio in audio_files:
                f.write(f"file '{audio.absolute()}'\n")
            
        # Combine audio with optimized settings
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:a", "pcm_s16le",  # Use high-quality audio codec
            "-ar", "44100",       # Standard audio sample rate
            "-ac", "2",           # Stereo audio
            "-b:a", "192k",       # High audio bitrate
            str(combined_audio)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
        # Find the animation video - use the actual path where Manim saved it
        video_path = ANIMATION_DIR / topic.replace(' ', '_') / "videos" / f"{topic.replace(' ', '_')}_animation" / "480p15" / f"{topic.replace(' ', '_')}.mp4"
        if not video_path.exists():
            video_path = ANIMATION_DIR / f"{topic.replace(' ', '_')}.mp4"
            
        if not video_path.exists():
            log_error(f"Animation video not found at {video_path}")
            return None
            
        # Output path
        output_file = VIDEO_DIR / f"{topic.replace(' ', '_')}.mp4"
        
        # Get video duration
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        video_duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Get audio duration
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(combined_audio)
        ]
        audio_duration = float(subprocess.check_output(probe_cmd).decode().strip())
        
        # Adjust video speed if needed to match audio duration
        speed_factor = audio_duration / video_duration
        temp_video = TEMP_DIR / "temp_video.mp4"
        
        if abs(1 - speed_factor) > 0.1:  # If difference is more than 10%
            log_progress(f"Adjusting video speed by factor {speed_factor:.2f} to match audio duration")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                *hw_accel,
                "-an",  # Remove any existing audio
                str(temp_video)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            video_input = str(temp_video)
        else:
            video_input = str(video_path)
        
        # Combine video and audio with hardware acceleration
        cmd = [
            "ffmpeg", "-y",
            "-i", video_input,
            "-i", str(combined_audio),
            *hw_accel,
            "-c:a", "aac",
            "-b:a", "192k",       # High audio bitrate
            "-shortest",          # Use the shortest stream length
            "-af", "apad",       # Pad audio with silence if needed
            "-preset", "slow",    # Better quality encoding
            "-movflags", "+faststart",  # Enable streaming optimization
            "-threads", str(cpu_count),
            str(output_file)
        ]
        
        # Set process priority to high
        try:
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
        except:
            pass
                
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        
        # Clean up temporary files
        os.unlink(concat_file)
        os.unlink(combined_audio)
        if temp_video.exists():
            os.unlink(temp_video)
        
        log_success(f"Successfully created final video: {output_file}")
        return output_file
        
    except subprocess.CalledProcessError as e:
        log_error(f"Error combining audio and video: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        log_error(f"Unexpected error in combine_audio_and_video: {e}")
        traceback.print_exc()
        return None

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

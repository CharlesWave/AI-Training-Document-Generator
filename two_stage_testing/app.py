from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
import os
import json
import cv2
import time
import socket
import traceback
import logging
import base64
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from ai_service import generate_training_document, get_file_state
from prompts_Two_Stage import stage_1_prompt100, stage_2_prompt100
from pdf_generator import generate_pdf
import gc
import requests
import io
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change from DEBUG to INFO for the root logger
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set app logger to INFO level

# Set higher log level for ALL HTTP-related modules to suppress debug messages
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('http').setLevel(logging.WARNING)
logging.getLogger('connectionpool').setLevel(logging.WARNING)

# Add specific custom logging for the three key events
def log_key_event(event_type, message):
    """Log key events that the user is interested in"""
    logger.info(f"KEY EVENT - {event_type}: {message}")

# Configure specific modules to show only warnings or errors
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Add three custom loggers for the specific events we want to track
gemini_upload_logger = logging.getLogger('gemini_upload')
gemini_upload_logger.setLevel(logging.INFO)

gemini_process_logger = logging.getLogger('gemini_process')
gemini_process_logger.setLevel(logging.INFO)

gpt4o_logger = logging.getLogger('gpt4o')
gpt4o_logger.setLevel(logging.INFO)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SCREENSHOTS_FOLDER'] = 'screenshots'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit
app.secret_key = os.urandom(24)  # For session management
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Increase socket timeout to avoid write operation timeouts
socket.setdefaulttimeout(1200)  # 20 minutes (increased from 10)

# Increase Flask's request timeout
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SCREENSHOTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_screenshots(video_path, timestamps, details):
    """
    Extract screenshots from the video at the specified timestamps.
    Returns a list of paths to the generated screenshots.
    """
    screenshot_paths = []
    
    try:
        logger.info(f"Starting screenshot extraction for {len(timestamps)} timestamp lists...")
        start_time = time.time()
        
        # Create a single VideoCapture object to reuse
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return [[] for _ in range(len(details))]
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video properties: Duration={duration:.2f}s, FPS={fps:.2f}, Total frames={total_frames}, Resolution={width}x{height}")
        
        # Use multi-processing if the video has many timestamp lists
        use_multithreading = len(timestamps) > 10 and os.cpu_count() > 1
        
        # Determine the total number of screenshots to extract
        total_screenshots = sum(len(ts_list) for ts_list in timestamps)
        screenshot_count = 0
        
        # Process each detail's timestamps
        for i, timestamp_list in enumerate(details):
            detail_screenshots = []
            
            # Check if timestamp list is empty
            if not timestamp_list or i >= len(timestamps):
                logger.info(f"No timestamps provided for detail #{i+1}: '{details[i][:50]}...'")
                screenshot_paths.append([])  # Add an empty list for this detail
                continue
            
            # Get the timestamps for this detail
            current_timestamps = timestamps[i]
            
            for timestamp in current_timestamps:
                try:
                    # Convert timestamp string (e.g., "1:30") to seconds
                    minutes, seconds = timestamp.split(':')
                    time_in_seconds = int(minutes) * 60 + float(seconds)
                    
                    # Skip if timestamp is beyond video duration
                    if duration > 0 and time_in_seconds > duration:
                        logger.warning(f"Timestamp {timestamp} exceeds video duration of {duration:.2f}s")
                        continue
                    
                    # Set the frame position
                    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
                    
                    # Read the frame
                    success, frame = cap.read()
                    if success:
                        # Generate filename
                        screenshot_filename = f"detail_{i+1}_timestamp_{timestamp.replace(':', '_')}.png"
                        screenshot_path = os.path.join(app.config['SCREENSHOTS_FOLDER'], screenshot_filename)
                        
                        # Save the frame with original quality, no compression or modification
                        cv2.imwrite(screenshot_path, frame)
                        detail_screenshots.append(screenshot_path)
                        
                        # Update progress
                        screenshot_count += 1
                        if screenshot_count % 10 == 0 or screenshot_count == total_screenshots:
                            logger.info(f"Extracted {screenshot_count}/{total_screenshots} screenshots ({screenshot_count/total_screenshots*100:.1f}%)")
                    else:
                        logger.warning(f"Failed to capture screenshot at timestamp {timestamp} for detail #{i+1}")
                
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                    continue
            
            screenshot_paths.append(detail_screenshots)
            
            # Clean up memory periodically
            if i % 5 == 4:  # Increased frequency of cleanup
                gc.collect()
                
                # If processing many screenshots, periodically recreate the VideoCapture
                # to avoid memory leaks in OpenCV
                if i > 0 and i % 20 == 19 and len(timestamps) > 30:
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Could not reopen video file {video_path}")
                        break
        
        # Release the video capture
        cap.release()
        
        # Verify that we have the same number of screenshot lists as details
        if len(screenshot_paths) != len(details):
            logger.warning(f"Mismatch between number of details ({len(details)}) and screenshot lists ({len(screenshot_paths)})")
            # Pad with empty lists to match details length
            while len(screenshot_paths) < len(details):
                screenshot_paths.append([])
                
        logger.info(f"Screenshot extraction completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in extract_screenshots: {str(e)}")
        logger.error(traceback.format_exc())
        # If there was an error, ensure we return at least empty lists
        if len(screenshot_paths) < len(details):
            # Pad with empty lists to match details length
            screenshot_paths.extend([[] for _ in range(len(details) - len(screenshot_paths))])
    
    return screenshot_paths

def extract_screenshots_with_folder(video_path, timestamps, details, screenshot_folder_path):
    """
    Extract screenshots from the video at the specified timestamps and save them to the specified folder.
    Returns a list of paths to the generated screenshots.
    
    Args:
        video_path (str): Path to the video file
        timestamps (list): List of timestamp lists for each detail
        details (list): List of knowledge points or details
        screenshot_folder_path (str): Path to the folder where screenshots should be saved
        
    Returns:
        list: List of lists, where each inner list contains paths to screenshots for a detail
    """
    screenshot_paths = []
    
    try:
        logger.info(f"Starting screenshot extraction for {len(timestamps)} timestamp lists...")
        start_time = time.time()
        
        # Create a single VideoCapture object to reuse
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file {video_path}")
            return [[] for _ in range(len(details))]
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Video properties: Duration={duration:.2f}s, FPS={fps:.2f}, Total frames={total_frames}, Resolution={width}x{height}")
        
        # Determine the total number of screenshots to extract
        total_screenshots = sum(len(ts_list) for ts_list in timestamps)
        screenshot_count = 0
        
        # Process each detail's timestamps
        for i, timestamp_list in enumerate(details):
            detail_screenshots = []
            
            # Check if timestamp list is empty
            if not timestamp_list or i >= len(timestamps):
                logger.info(f"No timestamps provided for detail #{i+1}: '{details[i][:50]}...'")
                screenshot_paths.append([])  # Add an empty list for this detail
                continue
            
            # Get the timestamps for this detail
            current_timestamps = timestamps[i]
            
            for timestamp in current_timestamps:
                try:
                    # Convert timestamp string (e.g., "1:30") to seconds
                    minutes, seconds = timestamp.split(':')
                    time_in_seconds = int(minutes) * 60 + float(seconds)
                    
                    # Skip if timestamp is beyond video duration
                    if duration > 0 and time_in_seconds > duration:
                        logger.warning(f"Timestamp {timestamp} exceeds video duration of {duration:.2f}s")
                        continue
                    
                    # Set the frame position
                    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000)
                    
                    # Read the frame
                    success, frame = cap.read()
                    if success:
                        # Generate filename
                        screenshot_filename = f"detail_{i+1}_timestamp_{timestamp.replace(':', '_')}.png"
                        screenshot_path = os.path.join(screenshot_folder_path, screenshot_filename)
                        
                        # Save the frame with original quality, no compression or modification
                        cv2.imwrite(screenshot_path, frame)
                        detail_screenshots.append(screenshot_path)
                        
                        # Update progress
                        screenshot_count += 1
                        if screenshot_count % 10 == 0 or screenshot_count == total_screenshots:
                            logger.info(f"Extracted {screenshot_count}/{total_screenshots} screenshots ({screenshot_count/total_screenshots*100:.1f}%)")
                    else:
                        logger.warning(f"Failed to capture screenshot at timestamp {timestamp} for detail #{i+1}")
                
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                    continue
            
            screenshot_paths.append(detail_screenshots)
            
            # Clean up memory periodically
            if i % 5 == 4:  # Increased frequency of cleanup
                gc.collect()
                
                # If processing many screenshots, periodically recreate the VideoCapture
                # to avoid memory leaks in OpenCV
                if i > 0 and i % 20 == 19 and len(timestamps) > 30:
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        logger.error(f"Could not reopen video file {video_path}")
                        break
        
        # Release the video capture
        cap.release()
        
        # Verify that we have the same number of screenshot lists as details
        if len(screenshot_paths) != len(details):
            logger.warning(f"Mismatch between number of details ({len(details)}) and screenshot lists ({len(screenshot_paths)})")
            # Pad with empty lists to match details length
            while len(screenshot_paths) < len(details):
                screenshot_paths.append([])
                
        logger.info(f"Screenshot extraction completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in extract_screenshots_with_folder: {str(e)}")
        logger.error(traceback.format_exc())
        # If there was an error, ensure we return at least empty lists
        if len(screenshot_paths) < len(details):
            # Pad with empty lists to match details length
            screenshot_paths.extend([[] for _ in range(len(details) - len(screenshot_paths))])
    
    return screenshot_paths

def image_to_base64(image_path):
    """Convert an image file to base64 encoded string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

def gpt4o_select_screenshots(knowledge_point, screenshots, summary):
    """
    Use GPT-4o to select relevant screenshots and provide captions
    
    Args:
        knowledge_point (str): The knowledge point text
        screenshots (list): List of paths to screenshots
        summary (dict): Summary from stage 1
        
    Returns:
        list: List of dictionaries with selection results
    """
    try:
        logger.info(f"Starting GPT-4o processing for knowledge point: {knowledge_point[:50]}...")
        start_time = time.time()
        
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log_key_event("GPT4O_API_KEY_MISSING", "OpenAI API key not found in environment variables")
            logger.error("OpenAI API key not found in environment variables")
            return []
        logger.info(f"OpenAI API key found with length: {len(api_key)}")
        
        # Format the system prompt with the summary
        system_prompt = stage_2_prompt100.replace("{{summary_from_stage_1}}", json.dumps(summary))
        
        # If there are no screenshots, return a default response
        if not screenshots:
            log_key_event("GPT4O_NO_SCREENSHOTS", "No screenshots provided for this knowledge point")
            logger.warning("No screenshots provided for this knowledge point")
            return [{
                "screenshot_index": 0,
                "selected": False,
                "caption_or_reason": "No screenshot provided"
            }]
        
        log_key_event("GPT4O_PROCESSING_SCREENSHOTS", f"Processing {len(screenshots)} screenshots for knowledge point")
        logger.info(f"Processing {len(screenshots)} screenshots")
        
        # Verify that all screenshot paths exist and are accessible
        valid_screenshots = []
        for i, path in enumerate(screenshots):
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                valid_screenshots.append(path)
            else:
                logger.error(f"Screenshot path does not exist: {path}")
        
        logger.info(f"Found {len(valid_screenshots)} valid screenshots out of {len(screenshots)}")
        
        if len(valid_screenshots) == 0:
            log_key_event("GPT4O_NO_VALID_SCREENSHOTS", "No valid screenshots found for this knowledge point")
            logger.error("No valid screenshots found for this knowledge point")
            return [{
                "screenshot_index": 0,
                "selected": False,
                "caption_or_reason": "No valid screenshots found"
            }]
        
        # Prepare the user message with base64 encoded images
        image_content = [
            {
                "type": "text",
                "text": f"Knowledge point: {knowledge_point}\nPlease select the most relevant screenshots and provide captions."
            }
        ]
        
        # Track the total size of encoded images
        total_image_size = 0
        successful_images = 0
        
        for i, screenshot_path in enumerate(valid_screenshots):
            try:
                # Log the image file size
                file_size = os.path.getsize(screenshot_path)
                logger.debug(f"Processing screenshot {i+1}/{len(valid_screenshots)}: {screenshot_path} ({file_size/1024:.1f} KB)")
                
                base64_image = image_to_base64(screenshot_path)
                if base64_image:
                    image_size = len(base64_image)
                    total_image_size += image_size
                    successful_images += 1
                    
                    logger.debug(f"Screenshot {i+1} encoded successfully: {image_size/1024:.1f} KB as base64")
                    image_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                else:
                    logger.warning(f"Failed to encode screenshot {i+1}")
            except Exception as e:
                logger.error(f"Error processing screenshot {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
        
        if successful_images == 0:
            logger.error("Failed to process any screenshots")
            return [{
                "screenshot_index": 0,
                "selected": False,
                "caption_or_reason": "Error processing screenshots"
            }]
            
        logger.info(f"Successfully encoded {successful_images}/{len(valid_screenshots)} screenshots. Total size: {total_image_size/1024/1024:.2f} MB")
        
        # Make the API request
        log_key_event("GPT4O_SENDING_REQUEST", "Sending request to OpenAI API (GPT-4o)")
        logger.info("Sending request to OpenAI API (GPT-4o)...")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": image_content}
            ],
            "max_tokens": 3000
        }
        
        # Log request details
        logger.debug(f"Request details: URL={url}, Model=gpt-4o, Max tokens=3000")
        logger.debug(f"Request headers: {headers.keys()}")
        logger.debug(f"Message count: {len(data['messages'])}")
        logger.debug(f"Content items in user message: {len(image_content)}")
        
        # Add retry logic for OpenAI API
        max_retries = 3
        retry_count = 0
        retry_delay = 5  # seconds
        
        while retry_count < max_retries:
            try:
                logger.info(f"Sending HTTP request to OpenAI (attempt {retry_count+1}/{max_retries})...")
                request_start = time.time()
                
                # Add logging before the actual request
                log_key_event("GPT4O_ATTEMPT", f"Attempt {retry_count+1}/{max_retries} to call GPT-4o API")
                
                response = requests.post(url, headers=headers, json=data, timeout=120)  # 2 minute timeout
                
                request_time = time.time() - request_start
                logger.info(f"OpenAI API response received in {request_time:.2f} seconds with status code: {response.status_code}")
                
                # Check for error status codes that should trigger a retry
                if response.status_code in [429, 500, 502, 503, 504]:
                    log_key_event("GPT4O_ERROR", f"Received error status code: {response.status_code}")
                    logger.warning(f"Received retryable status code: {response.status_code}")
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        logger.info(f"Retrying in {retry_delay} seconds (attempt {retry_count+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Failed after {max_retries} attempts with status code: {response.status_code}")
                        return []
                
                # If we get here, the request was successful
                log_key_event("GPT4O_SUCCESS", f"Successfully received response with status: {response.status_code}")
                response.raise_for_status()
                break
                
            except requests.exceptions.Timeout:
                log_key_event("GPT4O_TIMEOUT", f"Request timed out after 120 seconds (attempt {retry_count+1}/{max_retries})")
                logger.error(f"Request to OpenAI API timed out after 120 seconds (attempt {retry_count+1}/{max_retries})")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    logger.info(f"Retrying in {retry_delay} seconds (attempt {retry_count+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed after {max_retries} attempts due to timeout")
                    return []
                    
            except requests.exceptions.RequestException as e:
                log_key_event("GPT4O_ERROR", f"API request failed: {str(e)}")
                logger.error(f"Request to OpenAI API failed: {str(e)} (attempt {retry_count+1}/{max_retries})")
                
                # Determine if this error is retryable
                retryable_error = False
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code in [429, 500, 502, 503, 504]:
                        retryable_error = True
                
                if retryable_error and retry_count < max_retries - 1:
                    retry_count += 1
                    logger.info(f"Retrying in {retry_delay} seconds (attempt {retry_count+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed after {retry_count+1} attempts due to non-retryable error or max retries reached")
                    return []
        
        # If we get here, we have a successful response
        try:
            # Extract and parse the response
            response_data = response.json()
            logger.debug(f"Response data keys: {list(response_data.keys())}")
            
            if 'choices' not in response_data or len(response_data['choices']) == 0:
                logger.error(f"No choices in response: {response_data}")
                return []
                
            content = response_data['choices'][0]['message']['content']
            logger.debug(f"Response content length: {len(content)} chars")
            logger.debug(f"Response content preview: {content[:200]}...")
            
            # Extract the JSON list from the response
            try:
                # Find JSON content (possibly wrapped in markdown code blocks)
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    logger.debug(f"Extracted JSON content: {json_content[:200]}...")
                    
                    selections = json.loads(json_content)
                    logger.info(f"Successfully parsed {len(selections)} selection results")
                    return selections
                else:
                    logger.error("Could not find JSON content in GPT-4o response")
                    logger.error(f"Full response content: {content}")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing GPT-4o JSON response: {str(e)}")
                logger.error(f"Response content: {content}")
                return []
        except Exception as e:
            logger.error(f"Error processing OpenAI API response: {str(e)}")
            logger.error(f"Raw response: {response.text[:500]}")
            return []
            
        total_time = time.time() - start_time
        logger.info(f"Total GPT-4o processing took {total_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Error in gpt4o_select_screenshots: {str(e)}")
        logger.error(traceback.format_exc())
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    
    file = request.files['video']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Generate unique job ID
        job_id = int(time.time())
        
        # Check if user wants to include JSON
        include_json = 'include_json' in request.form
        
        # Store in session for later use
        session['include_json'] = include_json
        session['file_path'] = file_path
        session['job_id'] = job_id
        session['processing_stage'] = 'stage1'  # Start with stage 1
        
        # Process asynchronously in a real app, but for PoC we'll do it synchronously
        return redirect(url_for('process_video', job_id=job_id))
    
    return redirect(request.url)

@app.route('/process/<int:job_id>')
def process_video(job_id):
    file_path = session.get('file_path')
    if not file_path:
        return redirect(url_for('index'))
        
    # Redirect to the processing page with the initial status
    return render_template('processing.html', 
                         message="Your video is being processed by AI...",
                         job_id=job_id,
                         stage="stage1",
                         progress=0)

@app.route('/stage1/<int:job_id>')
def process_stage1(job_id):
    """Execute Stage 1: Gemini 2.0 Flash processing"""
    try:
        file_path = session.get('file_path')
        logger.info(f"Starting Stage 1 processing for file: {file_path}, job_id: {job_id}")
        
        # Call Gemini API to get training document JSON
        response = generate_training_document(stage_1_prompt100, file_path)
        
        # Check if the response indicates the file is still processing
        if isinstance(response, dict) and response.get('status') == 'processing':
            logger.info(f"File still processing: {response.get('file_name')}")
            # Store file name in session for checking status later
            session['file_name'] = response.get('file_name')
            session['processing_stage'] = 'video_upload'  # Indicate we're in video upload stage
            
            return json.dumps({
                'status': 'processing',
                'message': 'Your video is still being uploaded and processed...',
                'stage': 'video_upload',
                'progress': 10
            })
        
        # Store the raw response
        session['stage1_response'] = response
        
        try:
            response_data = json.loads(response)
            logger.info(f"Successfully parsed JSON response with keys: {list(response_data.keys())}")
            
            # Save the JSON response
            json_filename = f"stage1_response_{job_id}.json"
            json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json_file.write(response)
                
            logger.info(f"Saved stage 1 response to {json_path}")
            
            # Dynamically identify keys in the response
            summary_key = None
            points_key = None
            timestamps_key = None
            
            # Find the keys based on their structure
            for key, value in response_data.items():
                if key == 'Summary':
                    summary_key = key
                elif key == 'Timestamps':
                    timestamps_key = key
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    # This is likely the list of text points (knowledge_points, etc.)
                    points_key = key
            
            # If points_key not found, try to find by alternative names
            if not points_key:
                for possible_key in ['knowledge_points', 'Knowledge Points', 'Details', 'Points']:
                    if possible_key in response_data:
                        points_key = possible_key
                        break
            
            # If timestamps_key not found, try to find by structure
            if not timestamps_key:
                for key, value in response_data.items():
                    if isinstance(value, list) and all(isinstance(item, list) for item in value):
                        # Check if this looks like timestamps (values contain ":")
                        if any(any(':' in str(x) for x in sublist) for sublist in value if sublist):
                            timestamps_key = key
                            break
            
            logger.info(f"Stage 1 - Identified keys: summary_key={summary_key}, points_key={points_key}, timestamps_key={timestamps_key}")
            
            # Check that we found all required keys
            if not summary_key or not points_key or not timestamps_key:
                missing_keys = []
                if not summary_key: missing_keys.append("Summary")
                if not points_key: missing_keys.append("knowledge points")
                if not timestamps_key: missing_keys.append("Timestamps")
                
                error_msg = f"Could not identify required data structure in response: {missing_keys}"
                logger.error(error_msg)
                return json.dumps({
                    'status': 'error',
                    'message': error_msg
                })
            
            # Store keys in session for later use
            session['summary_key'] = summary_key
            session['points_key'] = points_key
            session['timestamps_key'] = timestamps_key
            
            # Extract timestamps and points
            text_points = response_data[points_key]
            timestamps = response_data[timestamps_key]
            summary = response_data[summary_key]
            
            # Make sure the lengths match
            if len(text_points) != len(timestamps):
                error_msg = f"Mismatch between number of points ({len(text_points)}) and timestamp lists ({len(timestamps)})"
                logger.error(error_msg)
                return json.dumps({
                    'status': 'error',
                    'message': error_msg
                })
            
            # Create a folder for this job's screenshots
            screenshot_folder = f"stage_1_screenshot_{job_id}"
            screenshot_folder_path = os.path.join(app.config['SCREENSHOTS_FOLDER'], screenshot_folder)
            os.makedirs(screenshot_folder_path, exist_ok=True)
            logger.info(f"Using screenshot folder: {screenshot_folder_path}")
            
            # Store the screenshot folder path in session for later use
            session['current_screenshot_folder'] = screenshot_folder_path
            
            # Extract screenshots from the video using the timestamped folder
            logger.info(f"Extracting screenshots from video using {points_key} and {timestamps_key}...")
            screenshots = extract_screenshots_with_folder(file_path, timestamps, text_points, screenshot_folder_path)
            
            # Store the screenshots, text points, and summary in session
            # We'll store the paths in a structured way for stage 2
            knowledge_data = []
            for i, point in enumerate(text_points):
                knowledge_data.append({
                    'point': point,
                    'screenshots': screenshots[i] if i < len(screenshots) else []
                })
            
            # Save the knowledge data to a file (might be too large for session)
            knowledge_data_filename = f"knowledge_data_{job_id}.json"
            knowledge_data_path = os.path.join(app.config['UPLOAD_FOLDER'], knowledge_data_filename)
            with open(knowledge_data_path, 'w', encoding='utf-8') as data_file:
                json.dump(knowledge_data, data_file)
                
            session['knowledge_data_path'] = knowledge_data_path
            session['summary'] = summary
            session['total_points'] = len(text_points)
            session['current_point_index'] = 0
            session['processing_stage'] = 'stage2'
            
            # Explicitly save the session to ensure it's persisted
            session.modified = True
            
            logger.info(f"Stage 1 complete. Moving to Stage 2 with {len(text_points)} knowledge points.")
            logger.info(f"Session variables set: knowledge_data_path={session.get('knowledge_data_path')}, total_points={session.get('total_points')}, current_point_index={session.get('current_point_index')}")
            
            # Return success to move to stage 2
            return json.dumps({
                'status': 'success',
                'stage': 'stage2',
                'message': 'Stage 1 complete. Starting Stage 2 processing...',
                'progress': 50,
                'total_points': len(text_points)
            })
            
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {str(json_err)}")
            logger.error(f"Invalid JSON response (first 500 chars): {response[:500]}")
            return json.dumps({
                'status': 'error',
                'message': f"JSON parsing error: {str(json_err)}"
            })
            
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Error in process_stage1: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        return json.dumps({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/stage2/<int:job_id>')
def process_stage2(job_id):
    """Execute Stage 2: GPT-4o processing for one knowledge point"""
    try:
        # Get data from session
        logger.info(f"Starting Stage 2 processing for job_id: {job_id}")
        knowledge_data_path = session.get('knowledge_data_path')
        current_point_index = session.get('current_point_index', 0)
        total_points = session.get('total_points', 0)
        
        log_key_event("STAGE2_STARTED", f"Starting Stage 2 for job_id={job_id}, point {current_point_index+1}/{total_points}")
        logger.info(f"Session data: knowledge_data_path={knowledge_data_path}, current_point_index={current_point_index}, total_points={total_points}")
        
        if not knowledge_data_path:
            log_key_event("STAGE2_ERROR", "knowledge_data_path not found in session")
            logger.error("knowledge_data_path not found in session")
            return json.dumps({
                'status': 'error',
                'message': "Session data lost. Please try again."
            })
            
        if not os.path.exists(knowledge_data_path):
            log_key_event("STAGE2_ERROR", f"Knowledge data file not found: {knowledge_data_path}")
            logger.error(f"Knowledge data file not found: {knowledge_data_path}")
            return json.dumps({
                'status': 'error',
                'message': f"Knowledge data file not found: {os.path.basename(knowledge_data_path)}"
            })
            
        logger.info(f"Current point index: {current_point_index}/{total_points}")
        
        # Verify OpenAI API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            log_key_event("STAGE2_ERROR", "OpenAI API key not found in environment variables")
            logger.error("OpenAI API key not found in environment variables")
            return json.dumps({
                'status': 'error',
                'message': "OpenAI API key not configured. Please check your environment variables."
            })
        logger.info(f"OpenAI API key is configured (length: {len(api_key)})")
            
        summary = session.get('summary')
        if not summary:
            logger.warning("Summary not found in session, using empty dict")
            summary = {}
        
        # Load knowledge data from file
        try:
            with open(knowledge_data_path, 'r', encoding='utf-8') as data_file:
                knowledge_data = json.load(data_file)
            logger.info(f"Loaded knowledge data with {len(knowledge_data)} points")
            
            # Log a sample of the first knowledge point for debugging
            if knowledge_data and len(knowledge_data) > 0:
                first_point = knowledge_data[0]
                logger.info(f"First knowledge point: {first_point['point'][:100]}...")
                logger.info(f"First point has {len(first_point.get('screenshots', []))} screenshots")
                
                # Log the point we're about to process
                if current_point_index < len(knowledge_data):
                    current_point = knowledge_data[current_point_index]
                    logger.info(f"About to process point {current_point_index}: {current_point['point'][:100]}...")
                    logger.info(f"Point {current_point_index} has {len(current_point.get('screenshots', []))} screenshots")
                else:
                    log_key_event("STAGE2_ERROR", f"Invalid current_point_index: {current_point_index} is out of range")
                    logger.error(f"Invalid current_point_index: {current_point_index} is out of range for knowledge_data length {len(knowledge_data)}")
            else:
                log_key_event("STAGE2_ERROR", "Knowledge data is empty or invalid")
                logger.error("Knowledge data is empty or invalid")
                return json.dumps({
                    'status': 'error',
                    'message': "Knowledge data is empty or invalid."
                })
        except Exception as e:
            log_key_event("STAGE2_ERROR", f"Error loading knowledge data: {str(e)}")
            logger.error(f"Error loading knowledge data: {str(e)}")
            return json.dumps({
                'status': 'error',
                'message': f"Error loading knowledge data: {str(e)}"
            })
        
        # Explicitly update session to prevent session loss
        session.modified = True
        logger.info("Session marked as modified to prevent data loss")
        
        # Check if we've processed all points
        if current_point_index >= total_points:
            # Move to the final document generation
            log_key_event("STAGE2_COMPLETE", "All knowledge points processed. Moving to document generation.")
            logger.info("All knowledge points processed. Moving to document generation.")
            session['processing_stage'] = 'document_generation'
            session.modified = True
            
            return json.dumps({
                'status': 'success',
                'stage': 'document_generation',
                'message': 'Stage 2 complete. Generating final document...',
                'progress': 90
            })
        
        # Process the current knowledge point
        try:
            current_data = knowledge_data[current_point_index]
            knowledge_point = current_data['point']
            screenshots = current_data.get('screenshots', [])
            
            log_key_event("STAGE2_PROCESSING_POINT", f"Processing knowledge point {current_point_index+1}/{total_points}")
            logger.info(f"Processing knowledge point {current_point_index+1}/{total_points}: {knowledge_point[:50]}...")
            logger.info(f"Number of screenshots for this point: {len(screenshots)}")
            
            # Verify screenshots exist
            valid_screenshots = []
            for i, screenshot_path in enumerate(screenshots):
                if os.path.exists(screenshot_path):
                    valid_screenshots.append(screenshot_path)
                else:
                    logger.error(f"Screenshot {i} does not exist: {screenshot_path}")
            logger.info(f"Valid screenshots: {len(valid_screenshots)}/{len(screenshots)}")
            
            # Call GPT-4o to select screenshots and provide captions
            log_key_event("STAGE2_CALLING_GPT4O", f"Calling GPT-4o API for point {current_point_index+1}/{total_points}")
            logger.info("About to call GPT-4o API...")
            selection_results = gpt4o_select_screenshots(knowledge_point, screenshots, summary)
            
            if selection_results:
                log_key_event("STAGE2_GPT4O_SUCCESS", f"GPT-4o API returned {len(selection_results)} results")
                logger.info(f"GPT-4o API call complete, received {len(selection_results)} results")
            else:
                log_key_event("STAGE2_GPT4O_NO_RESULTS", "No selection results returned from GPT-4o")
                logger.warning("No selection results returned from GPT-4o")
                # Continue anyway with empty results
                selection_results = []
            
            # Store the results in the knowledge data
            knowledge_data[current_point_index]['selection_results'] = selection_results
            
            # Update the knowledge data file
            with open(knowledge_data_path, 'w', encoding='utf-8') as data_file:
                json.dump(knowledge_data, data_file)
            
            # Increment the current point index
            session['current_point_index'] = current_point_index + 1
            session.modified = True
            log_key_event("STAGE2_POINT_COMPLETED", f"Updated current_point_index to {current_point_index + 1}")
            logger.info(f"Updated current_point_index to {current_point_index + 1}")
            
            # Calculate progress
            progress = int(55 + (current_point_index + 1) / total_points * 35)
            
            # Return progress information
            return json.dumps({
                'status': 'processing',
                'stage': 'stage2',
                'message': f'Processing knowledge point {current_point_index+1}/{total_points}...',
                'progress': progress,
                'current_point': current_point_index + 1,
                'total_points': total_points
            })
        except IndexError:
            log_key_event("STAGE2_ERROR", f"Index error accessing knowledge point {current_point_index}")
            logger.error(f"Index error accessing knowledge point {current_point_index}. Total points: {len(knowledge_data)}")
            return json.dumps({
                'status': 'error',
                'message': f"Error accessing knowledge point {current_point_index+1}"
            })
        
    except Exception as e:
        stack_trace = traceback.format_exc()
        log_key_event("STAGE2_ERROR", f"Unhandled error: {str(e)}")
        logger.error(f"Error in process_stage2: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        return json.dumps({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/generate_document/<int:job_id>')
def generate_document(job_id):
    """Generate the final document with selected screenshots"""
    try:
        # Get data from session
        knowledge_data_path = session.get('knowledge_data_path')
        include_json = session.get('include_json', False)
        
        # Get screenshot folder path from session
        screenshot_folder_path = session.get('current_screenshot_folder')
        if not screenshot_folder_path:
            logger.warning("No screenshot folder path found in session, using default screenshots folder")
            screenshot_folder_path = app.config['SCREENSHOTS_FOLDER']
        
        logger.info(f"Using screenshot folder: {screenshot_folder_path}")
        
        # Load knowledge data from file
        with open(knowledge_data_path, 'r', encoding='utf-8') as data_file:
            knowledge_data = json.load(data_file)
        
        # Prepare data for PDF generation
        summary = session.get('summary')
        
        # Extract points and selected screenshots with captions
        points = []
        screenshots_selected = []
        captions = []
        
        for item in knowledge_data:
            points.append(item['point'])
            
            # Filter selected screenshots and their captions
            point_screenshots = []
            point_captions = []
            
            # Check if selection_results exist
            if 'selection_results' in item and item['selection_results']:
                all_screenshots = item['screenshots']
                selection_results = item['selection_results']
                
                # Go through each selection result
                for result in selection_results:
                    if 'selected' in result and result['selected'] and 'screenshot_index' in result:
                        index = result['screenshot_index']
                        if index < len(all_screenshots):
                            screenshot_path = all_screenshots[index]
                            # Verify the screenshot exists
                            if os.path.exists(screenshot_path):
                                point_screenshots.append(screenshot_path)
                                
                                # Add caption
                                caption = result.get('caption_or_reason', '')
                                point_captions.append(caption)
                            else:
                                logger.warning(f"Selected screenshot does not exist: {screenshot_path}")
            
            screenshots_selected.append(point_screenshots)
            captions.append(point_captions)
        
        # Create a response data structure for PDF generation
        response_data = {
            'Summary': summary,
            'Knowledge Points': points
        }
        
        # Generate PDF
        logger.info("Generating PDF document...")
        pdf_path = generate_pdf(response_data, screenshots_selected, captions)
        logger.info(f"PDF generated successfully at: {pdf_path}")
        
        # Store the path in session for download
        session['pdf_path'] = pdf_path
        
        # If JSON was requested, prepare that as well
        json_path = None
        if include_json:
            # Create a complete document JSON with all data
            document_data = {
                'Summary': summary,
                'Knowledge Points': points,
                'Processing Results': knowledge_data,
                'Screenshot Folder': screenshot_folder_path  # Add screenshot folder path to JSON
            }
            
            json_filename = f"training_document_{job_id}.json"
            json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(document_data, json_file, indent=2)
                
            session['json_path'] = json_path
        
        # Return success with redirect to results page
        session['processing_stage'] = 'complete'
        return json.dumps({
            'status': 'complete',
            'message': 'Document generation complete!',
            'progress': 100,
            'redirect': url_for('show_results', job_id=job_id)
        })
        
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Error in generate_document: {str(e)}")
        logger.error(f"Stack trace: {stack_trace}")
        return json.dumps({
            'status': 'error',
            'message': f"Error: {str(e)}"
        })

@app.route('/results/<int:job_id>')
def show_results(job_id):
    """Show results page with links to downloads"""
    include_json = session.get('include_json', False)
    pdf_path = session.get('pdf_path')
    json_path = session.get('json_path') if include_json else None
    
    if not pdf_path:
        return redirect(url_for('index'))
        
    return render_template('results.html', 
                          include_json=include_json,
                          job_id=job_id)

@app.route('/check_status/<int:job_id>')
def check_status(job_id):
    """Endpoint to check processing status and move through stages"""
    processing_stage = session.get('processing_stage', 'stage1')
    logger.info(f"CHECK STATUS: job_id={job_id}, current processing_stage={processing_stage}")
    
    # Log specific important session variables
    logger.info(f"Session data: processing_stage={processing_stage}, "
                f"current_point_index={session.get('current_point_index')}, "
                f"total_points={session.get('total_points')}, "
                f"knowledge_data_path={session.get('knowledge_data_path')}, "
                f"has_summary={bool(session.get('summary'))}")
    
    # Check if expected session data is missing but job_id exists
    # This can happen if session cookies are lost
    if processing_stage == 'stage1' and not session.get('file_path') and os.path.exists(os.path.join(app.config['SCREENSHOTS_FOLDER'], f"stage_1_screenshot_{job_id}")):
        logger.warning(f"Session data may be lost for job_id={job_id}, but screenshot folder exists")
        log_key_event("SESSION_RECOVERY_ATTEMPT", f"Attempting to recover lost session for job_id={job_id}")
        
        # Try to recover by checking if knowledge_data file exists
        knowledge_data_path = os.path.join(app.config['UPLOAD_FOLDER'], f"knowledge_data_{job_id}.json")
        if os.path.exists(knowledge_data_path):
            logger.info(f"Found knowledge data file for job_id={job_id}, recovering session")
            session['knowledge_data_path'] = knowledge_data_path
            session['current_screenshot_folder'] = os.path.join(app.config['SCREENSHOTS_FOLDER'], f"stage_1_screenshot_{job_id}")
            
            # Load knowledge data to restore session information
            try:
                with open(knowledge_data_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
                    session['total_points'] = len(knowledge_data)
                    session['current_point_index'] = 0  # Restart from beginning
                    session['processing_stage'] = 'stage2'
                    session.modified = True
                    log_key_event("SESSION_RECOVERED", f"Successfully recovered session for job_id={job_id}")
                    logger.info(f"Recovered session for job_id={job_id}, moving to stage2")
                    
                    return json.dumps({
                        'status': 'processing',
                        'stage': 'stage2',
                        'message': 'Continuing with screenshot selection...',
                        'progress': 50,
                        'total_points': len(knowledge_data)
                    })
            except Exception as e:
                log_key_event("SESSION_RECOVERY_FAILED", f"Failed to recover session: {str(e)}")
                logger.error(f"Failed to recover session: {str(e)}")
    
    # Handle file upload processing stage
    if processing_stage == 'video_upload':
        file_name = session.get('file_name')
        if not file_name:
            return json.dumps({'status': 'error', 'message': 'No file in progress'})
            
        current_state = get_file_state(file_name)
        logger.debug(f"File state: {current_state}")
        
        if current_state == 'ACTIVE':
            # The file is now active, update the processing stage
            log_key_event("STAGE_TRANSITION", "Moving from video_upload to stage1")
            session['processing_stage'] = 'stage1'
            return json.dumps({
                'status': 'processing', 
                'stage': 'stage1',
                'message': 'Video processing complete. Starting analysis...',
                'progress': 30
            })
        elif current_state == 'FAILED':
            log_key_event("PROCESSING_FAILED", f"File processing failed: {file_name}")
            return json.dumps({'status': 'error', 'message': 'File processing failed'})
        else:
            return json.dumps({
                'status': 'processing', 
                'stage': 'video_upload',
                'message': 'Your video is still being processed by AI...',
                'progress': 15
            })
    
    # Handle stage 1 processing
    elif processing_stage == 'stage1':
        log_key_event("STAGE_EXECUTION", "Executing Stage 1 processing")
        logger.info(f"Redirecting to Stage 1 processing for job_id={job_id}")
        return process_stage1(job_id)
    
    # Handle stage 2 processing
    elif processing_stage == 'stage2':
        log_key_event("STAGE_EXECUTION", "Executing Stage 2 processing")
        logger.info(f"Redirecting to Stage 2 processing for job_id={job_id}")
        return process_stage2(job_id)
    
    # Handle document generation
    elif processing_stage == 'document_generation':
        log_key_event("STAGE_EXECUTION", "Executing final document generation")
        logger.info(f"Redirecting to document generation for job_id={job_id}")
        return generate_document(job_id)
    
    # Handle completed process
    elif processing_stage == 'complete':
        log_key_event("PROCESSING_COMPLETE", f"Processing complete for job_id={job_id}")
        return json.dumps({
            'status': 'complete',
            'message': 'Processing complete!',
            'progress': 100,
            'redirect': url_for('show_results', job_id=job_id)
        })
    
    # Default response for unknown stage
    logger.warning(f"Unknown processing stage: {processing_stage}")
    log_key_event("UNKNOWN_STAGE", f"Unknown processing stage: {processing_stage}")
    return json.dumps({
        'status': 'unknown',
        'message': f'Unknown processing stage: {processing_stage}',
        'progress': 0
    })

@app.route('/download/pdf/<int:job_id>')
def download_pdf(job_id):
    """Download the generated PDF file"""
    try:
        pdf_path = session.get('pdf_path')
        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return render_template('error.html', error="PDF file not found. Please try again.")
        
        return send_file(pdf_path, as_attachment=True, download_name=f'training_document_{job_id}.pdf')
    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        return render_template('error.html', error=f"Error downloading PDF: {str(e)}")

@app.route('/download/json/<int:job_id>')
def download_json(job_id):
    """Download the raw JSON data"""
    try:
        json_path = session.get('json_path')
        if not json_path or not os.path.exists(json_path):
            logger.error(f"JSON file not found: {json_path}")
            return render_template('error.html', error="JSON file not found. Please try again.")
        
        return send_file(json_path, as_attachment=True, download_name=f'training_document_data_{job_id}.json')
    except Exception as e:
        logger.error(f"Error downloading JSON: {str(e)}")
        return render_template('error.html', error=f"Error downloading JSON: {str(e)}")

@app.route('/download/stage2_data/<int:job_id>')
def download_stage2_data(job_id):
    """Download the Stage 2 knowledge data with processing results so far"""
    try:
        # Get the knowledge data path from session
        knowledge_data_path = session.get('knowledge_data_path')
        
        if not knowledge_data_path or not os.path.exists(knowledge_data_path):
            logger.error(f"Stage 2 knowledge data file not found: {knowledge_data_path}")
            return render_template('error.html', error="Stage 2 knowledge data file not found. Processing may not have reached Stage 2 yet.")
        
        # Log that someone is downloading the intermediate results
        log_key_event("STAGE2_DATA_DOWNLOAD", f"User downloaded intermediate Stage 2 data for job_id={job_id}")
        logger.info(f"Serving intermediate Stage 2 results from: {knowledge_data_path}")
        
        return send_file(knowledge_data_path, as_attachment=True, download_name=f'stage2_data_{job_id}.json')
    except Exception as e:
        logger.error(f"Error downloading Stage 2 data: {str(e)}")
        return render_template('error.html', error=f"Error downloading Stage 2 data: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True) 
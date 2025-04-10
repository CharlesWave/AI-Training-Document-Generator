import base64
import os
import time
import logging
import traceback
import json
from google import genai
from google.genai import types

# Configure module logger
logger = logging.getLogger(__name__)

# Set higher log level for HTTP-related modules to suppress debug messages
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

# Custom loggers for key events
gemini_upload_logger = logging.getLogger('gemini_upload')
gemini_process_logger = logging.getLogger('gemini_process')

# Define log_key_event function for consistent formatting
def log_key_event(event_type, message):
    """Log key events that the user is interested in"""
    logger.info(f"KEY EVENT - {event_type}: {message}")

# Initialize the Gemini client once at the module level
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def get_file_state(file_name):
    """
    Retrieves the state of an uploaded file using its name.

    Args:
        file_name (str): The name of the uploaded file.

    Returns:
        str: The state of the file (e.g., "ACTIVE", "PROCESSING", "FAILED"), or None if an error occurs.
    """
    try:
        logger.debug(f"Checking state of file: {file_name}")
        # Use the correct method to get file state with the file name
        file_object = client.files.get(name=file_name)
        logger.debug(f"File state: {file_object.state}")
        return file_object.state

    except Exception as e:
        logger.error(f"Error getting file state: {e}")
        logger.error(traceback.format_exc())
        return None


def generate_training_document(system_prompt, user_prompt, video_path):

    try:
        logger.info(f"Starting training document generation for video: {video_path}")
        
        # Upload the file
        logger.info("Uploading file to Gemini API...")
        try:
            uploaded_file = client.files.upload(file=video_path)
            # Extract the file name (not URI) to use with get_file_state
            file_name = uploaded_file.name
            log_key_event("GEMINI_UPLOAD", f"File uploaded successfully. Name: {file_name}, URI: {uploaded_file.uri}")
            gemini_upload_logger.info(f"SUCCESSFUL UPLOAD: {file_name}")
        except Exception as upload_error:
            log_key_event("GEMINI_UPLOAD_FAILED", f"File upload failed: {str(upload_error)}")
            gemini_upload_logger.error(f"FAILED UPLOAD: {str(upload_error)}")
            raise
        
        # Check if the file is active before proceeding
        max_attempts = 10
        attempts = 0
        wait_time = 10  # seconds
        
        while attempts < max_attempts:
            file_state = get_file_state(file_name)
            logger.info(f"File state: {file_state}, attempt {attempts+1}/{max_attempts}")
            
            if file_state == "ACTIVE":
                log_key_event("GEMINI_FILE_ACTIVE", f"File {file_name} is now active and ready for processing")
                break
            elif file_state == "FAILED":
                error_msg = f"File upload failed with state: {file_state}"
                log_key_event("GEMINI_FILE_FAILED", error_msg)
                logger.error(error_msg)
                raise Exception(error_msg)
            elif file_state == "PROCESSING" or file_state == "PENDING":
                # If we're on the last attempt and still processing, return a special value
                if attempts == max_attempts - 1:
                    logger.warning(f"File still processing after {max_attempts} attempts, returning processing status")
                    return {"status": "processing", "message": "Video is still being processed", "file_name": file_name}
            
            # Wait before checking again
            logger.debug(f"Waiting {wait_time} seconds before checking file state again...")
            time.sleep(wait_time)
            attempts += 1
        
        # If we've exceeded max attempts and file is still not active
        current_state = get_file_state(file_name)
        if attempts >= max_attempts and current_state != "ACTIVE":
            error_msg = f"File did not become active after {max_attempts} attempts, current state: {current_state}"
            log_key_event("GEMINI_FILE_TIMEOUT", error_msg)
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Now that the file is active, proceed with document generation
        logger.info("File is active, proceeding with document generation")
        files = [uploaded_file]
        model = "gemini-2.0-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=user_prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            system_instruction=[
                types.Part.from_text(text=system_prompt),
            ],
        )

        # Add retry logic for API calls
        api_max_attempts = 3
        api_attempt = 0
        api_wait_time = 10
        
        while api_attempt < api_max_attempts:
            try:
                logger.info(f"Calling Gemini API with model: {model} (attempt {api_attempt+1}/{api_max_attempts})")
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                log_key_event("GEMINI_PROCESSING_COMPLETE", f"Successfully generated content from video {file_name}")
                gemini_process_logger.info(f"SUCCESSFUL PROCESSING: {file_name}")
                # If we got here without an exception, break out of the retry loop
                break
            except Exception as api_error:
                # Check if it's a 500 or 503 error
                error_str = str(api_error)
                is_server_error = False
                
                if '500' in error_str or '503' in error_str or 'UNAVAILABLE' in error_str:
                    is_server_error = True
                    logger.warning(f"Received server error: {error_str}")
                
                # If it's not a server error or we've reached max attempts, re-raise
                if not is_server_error or api_attempt >= api_max_attempts - 1:
                    log_key_event("GEMINI_PROCESSING_FAILED", f"Failed to process video: {error_str}")
                    gemini_process_logger.error(f"FAILED PROCESSING: {error_str}")
                    logger.error(f"Failed to call Gemini API after {api_attempt+1} attempts: {error_str}")
                    raise
                
                # Wait before retrying
                logger.info(f"Waiting {api_wait_time} seconds before retrying Gemini API call...")
                time.sleep(api_wait_time)
                api_attempt += 1
        
        # Validate that the response contains text and is valid JSON
        if not hasattr(response, 'text') or not response.text:
            error_msg = "Empty response received from Gemini API"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        # Clean the response - remove everything before the first { and after the last }
        raw_text = response.text
        logger.debug(f"Original raw response (first 100 chars): {raw_text[:100]}...")
        
        # Find the first opening brace and last closing brace
        first_brace_index = raw_text.find('{')
        last_brace_index = raw_text.rfind('}')
        
        if first_brace_index == -1 or last_brace_index == -1:
            error_msg = "Response does not contain valid JSON structure (missing braces)"
            logger.error(error_msg)
            logger.error(f"Raw response: {raw_text}")
            raise Exception(error_msg)
            
        # Extract just the JSON part
        json_text = raw_text[first_brace_index:last_brace_index + 1]
        
        # Log the cleaned response for debugging
        logger.debug(f"Cleaned JSON response (truncated): {json_text[:200]}...")
        
        # Try to parse as JSON to validate before returning
        try:
            json.loads(json_text)
            logger.info("Successfully validated response as valid JSON")
        except json.JSONDecodeError as e:
            logger.error(f"Response is not valid JSON after cleaning: {e}")
            logger.error(f"Invalid JSON response (first 500 chars): {json_text[:500]}")
            # Return the cleaned text anyway, so the caller can handle the error
            return json_text
        
        # Return the cleaned JSON text
        return json_text
        
    except Exception as e:
        logger.error(f"Error in generate_training_document: {e}")
        logger.error(traceback.format_exc())
        raise

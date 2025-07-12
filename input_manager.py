import cv2
import os
import glob
from pathlib import Path
from typing import Generator, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputManager:
    """
    Manages input sources for DualCast eSports commentary system.
    
    Supports three input types:
    - Video files (mp4, avi, mov, etc.)
    - Image folders (jpg, png, etc.)
    - Webcam feed
    """
    
    def __init__(self, 
                 source: Union[str, int], 
                 resize_dims: Optional[Tuple[int, int]] = None,
                 image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        Initialize InputManager.
        
        Args:
            source: Input source - video file path, image folder path, or webcam index (0, 1, etc.)
            resize_dims: Optional tuple (width, height) to resize frames
            image_extensions: Supported image file extensions
        """
        self.source = source
        self.resize_dims = resize_dims
        self.image_extensions = image_extensions
        self.source_type = self._detect_source_type()
        self.cap = None
        self.image_files = []
        self.current_image_index = 0
        
        logger.info(f"Initialized InputManager with source: {source}, type: {self.source_type}")
    
    def _detect_source_type(self) -> str:
        """Detect the type of input source."""
        if isinstance(self.source, int):
            return "webcam"
        elif isinstance(self.source, str):
            if os.path.isfile(self.source):
                return "video"
            elif os.path.isdir(self.source):
                return "images"
            else:
                raise ValueError(f"Invalid source path: {self.source}")
        else:
            raise ValueError(f"Unsupported source type: {type(self.source)}")
    
    def _setup_video_capture(self) -> bool:
        """Setup video capture for video files or webcam."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Log video properties for debugging
            if self.source_type == "video":
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"Video properties - FPS: {fps}, Frames: {frame_count}, "
                           f"Resolution: {width}x{height}")
            
            return True
        except Exception as e:
            logger.error(f"Error setting up video capture: {e}")
            return False
    
    def _setup_image_folder(self) -> bool:
        """Setup image file list from folder."""
        try:
            pattern_list = []
            for ext in self.image_extensions:
                pattern_list.extend(glob.glob(os.path.join(self.source, f"*{ext}")))
                pattern_list.extend(glob.glob(os.path.join(self.source, f"*{ext.upper()}")))
            
            self.image_files = sorted(pattern_list)
            
            if not self.image_files:
                logger.error(f"No image files found in {self.source}")
                return False
            
            logger.info(f"Found {len(self.image_files)} image files")
            return True
        except Exception as e:
            logger.error(f"Error setting up image folder: {e}")
            return False
    
    def _resize_frame(self, frame):
        """Resize frame if resize_dims is specified."""
        if self.resize_dims is not None:
            try:
                frame = cv2.resize(frame, self.resize_dims)
            except Exception as e:
                logger.warning(f"Failed to resize frame: {e}")
        return frame
    
    def read_frame(self) -> Generator[Optional[tuple], None, None]:
        """
        Generator that yields frames from the input source.
        
        Yields:
            tuple: (frame, metadata) where frame is numpy array and metadata is dict
                   Returns (None, None) when stream ends or error occurs
        """
        # Setup based on source type
        if self.source_type in ["video", "webcam"]:
            if not self._setup_video_capture():
                yield None, None
                return
        elif self.source_type == "images":
            if not self._setup_image_folder():
                yield None, None
                return
        
        try:
            frame_count = 0
            
            while True:
                frame = None
                metadata = {"frame_number": frame_count, "source_type": self.source_type}
                
                if self.source_type in ["video", "webcam"]:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.info("End of video stream reached")
                        break
                    
                    # Add video-specific metadata
                    if self.source_type == "video":
                        metadata["timestamp"] = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    
                elif self.source_type == "images":
                    if self.current_image_index >= len(self.image_files):
                        logger.info("All images processed")
                        break
                    
                    image_path = self.image_files[self.current_image_index]
                    frame = cv2.imread(image_path)
                    
                    if frame is None:
                        logger.warning(f"Failed to load image: {image_path}")
                        self.current_image_index += 1
                        continue
                    
                    # Add image-specific metadata
                    metadata["image_path"] = image_path
                    metadata["image_name"] = os.path.basename(image_path)
                    self.current_image_index += 1
                
                if frame is not None:
                    # Resize frame if requested
                    frame = self._resize_frame(frame)
                    
                    # Add frame dimensions to metadata
                    metadata["height"], metadata["width"] = frame.shape[:2]
                    metadata["channels"] = frame.shape[2] if len(frame.shape) > 2 else 1
                    
                    yield frame, metadata
                    frame_count += 1
                else:
                    logger.warning(f"Failed to read frame {frame_count}")
                    break
                    
        except Exception as e:
            logger.error(f"Error during frame reading: {e}")
        finally:
            self.cleanup()
        
        # Signal end of stream
        yield None, None
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Video capture released")
    
    def get_source_info(self) -> dict:
        """Get information about the input source."""
        info = {
            "source": self.source,
            "source_type": self.source_type,
            "resize_dims": self.resize_dims
        }
        
        if self.source_type == "video" and self.cap is not None:
            info.update({
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            })
        elif self.source_type == "images":
            info.update({
                "image_count": len(self.image_files),
                "supported_extensions": self.image_extensions
            })
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Video file
    print("=== Testing Video File ===")
    video_path = "sample_video.mp4"  # Replace with actual video path
    
    try:
        with InputManager(video_path, resize_dims=(640, 480)) as input_mgr:
            print(f"Source info: {input_mgr.get_source_info()}")
            
            frame_generator = input_mgr.read_frame()
            for i, (frame, metadata) in enumerate(frame_generator):
                if frame is None:
                    print("End of video stream")
                    break
                print(f"Frame {i}: {metadata}")
                if i >= 5:  # Process only first 5 frames for demo
                    break
    except Exception as e:
        print(f"Video test failed: {e}")
    
    # Example 2: Image folder
    print("\n=== Testing Image Folder ===")
    image_folder = "sample_images/"  # Replace with actual folder path
    
    try:
        with InputManager(image_folder, resize_dims=(640, 480)) as input_mgr:
            print(f"Source info: {input_mgr.get_source_info()}")
            
            frame_generator = input_mgr.read_frame()
            for i, (frame, metadata) in enumerate(frame_generator):
                if frame is None:
                    print("All images processed")
                    break
                print(f"Image {i}: {metadata}")
                if i >= 3:  # Process only first 3 images for demo
                    break
    except Exception as e:
        print(f"Image folder test failed: {e}")
    
    # Example 3: Webcam
    print("\n=== Testing Webcam ===")
    try:
        with InputManager(0, resize_dims=(640, 480)) as input_mgr:
            print(f"Source info: {input_mgr.get_source_info()}")
            
            frame_generator = input_mgr.read_frame()
            for i, (frame, metadata) in enumerate(frame_generator):
                if frame is None:
                    print("Webcam stream ended")
                    break
                print(f"Webcam frame {i}: {metadata}")
                if i >= 3:  # Process only first 3 frames for demo
                    break
    except Exception as e:
        print(f"Webcam test failed: {e}")
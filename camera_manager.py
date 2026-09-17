import cv2
from database import insert_camera, get_cameras, get_camera, delete_camera


def add_camera(name, source_type, source_url=None, device_index=0):
    """
    Register a new camera source.

    Args:
        name: Display name for the camera
        source_type: One of 'webcam', 'ip_stream', 'rtsp'
        source_url: URL for IP/RTSP cameras (None for webcam)
        device_index: Device index for local webcam (default 0)

    Returns:
        camera_id (int)
    """
    return insert_camera(name, source_type, source_url, device_index)


def remove_camera(camera_id):
    """Remove a camera source by ID."""
    delete_camera(camera_id)


def list_cameras():
    """Get all registered cameras."""
    return get_cameras()


def get_camera_source(camera_id):
    """
    Get the OpenCV-compatible source for a camera.

    Returns:
        int (device index) or str (URL), or None if camera not found.
    """
    camera = get_camera(camera_id)
    if camera is None:
        return None

    if camera["source_type"] == "webcam":
        return camera["device_index"]
    else:
        return camera["source_url"]


def test_camera_source(source):
    """
    Test if a camera source can be opened.

    Args:
        source: int (device index) or str (URL)

    Returns:
        tuple (success: bool, message: str)
    """
    try:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return True, "Camera source is working."
            else:
                return False, "Camera opened but could not read a frame."
        else:
            return False, "Could not open camera source."
    except Exception as e:
        return False, f"Error testing camera: {str(e)}"

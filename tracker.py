import time
from collections import defaultdict
from config import BACKGROUND_HISTORY_FRAMES, WASTE_PERSISTENCE_FRAMES


class TrackedObject:
    """Represents a tracked person or waste object across frames."""

    def __init__(self, track_id, class_name, box, confidence, first_seen_frame):
        self.track_id = track_id
        self.class_name = class_name
        self.box = box
        self.confidence = confidence
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = first_seen_frame
        self.frames_seen = 1
        self.is_new = True  # True if appeared after monitoring started
        self.associated_person_id = None  # For waste: which person was nearby

    def update(self, box, confidence, frame_number):
        self.box = box
        self.confidence = confidence
        self.last_seen_frame = frame_number
        self.frames_seen += 1
        # Bug 14 fix: transition is_new to False after enough frames
        if self.is_new and self.frames_seen >= WASTE_PERSISTENCE_FRAMES:
            self.is_new = False

    def center(self):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def is_stale(self, current_frame, max_missing=30):
        """Check if this object hasn't been seen for too many frames."""
        return (current_frame - self.last_seen_frame) > max_missing


class ObjectTracker:
    """
    Manages tracked objects and performs temporal before/after analysis.
    
    Uses track IDs from YOLOv8's built-in ByteTrack to maintain identity
    across frames. Determines which waste objects are NEW (appeared after
    a person was present) vs PRE-EXISTING.
    """

    def __init__(self):
        self.persons = {}       # track_id -> TrackedObject
        self.waste_objects = {}  # track_id -> TrackedObject
        self.frame_count = 0
        self.startup_frames = BACKGROUND_HISTORY_FRAMES  # Grace period
        self.startup_waste_ids = set()  # Waste IDs seen during startup

        # Region history: maps grid cells to sets of waste track_ids
        # Used to determine if waste was pre-existing
        self.region_waste_history = defaultdict(set)

    def update(self, detections):
        """
        Update tracker with new detections from detector.

        Args:
            detections: dict from detector.detect() with "persons" and "waste" keys

        Returns:
            dict with:
                - "persons": dict of track_id -> TrackedObject
                - "waste": dict of track_id -> TrackedObject
                - "new_waste": list of TrackedObject (waste that appeared AFTER startup)
                - "pre_existing_waste": list of TrackedObject (waste present from the start)
                - "frame_count": int
        """
        self.frame_count += 1
        current_person_ids = set()
        current_waste_ids = set()

        # Update persons
        for det in detections["persons"]:
            tid = det["track_id"]
            if tid == -1:
                continue
            current_person_ids.add(tid)

            if tid in self.persons:
                self.persons[tid].update(det["box"], det["confidence"], self.frame_count)
            else:
                self.persons[tid] = TrackedObject(
                    tid, "person", det["box"], det["confidence"], self.frame_count
                )

        # Update waste objects
        for det in detections["waste"]:
            tid = det["track_id"]
            if tid == -1:
                continue
            current_waste_ids.add(tid)

            if tid in self.waste_objects:
                self.waste_objects[tid].update(det["box"], det["confidence"], self.frame_count)
            else:
                obj = TrackedObject(
                    tid, det["class_name"], det["box"], det["confidence"], self.frame_count
                )
                # If we're still in the startup phase, mark as pre-existing
                if self.frame_count <= self.startup_frames:
                    obj.is_new = False
                    self.startup_waste_ids.add(tid)
                else:
                    obj.is_new = True
                self.waste_objects[tid] = obj

                # Record region
                region = self._get_region(det["box"])
                self.region_waste_history[region].add(tid)

        # Clean up stale tracks
        stale_persons = [
            tid for tid, obj in self.persons.items()
            if obj.is_stale(self.frame_count)
        ]
        for tid in stale_persons:
            del self.persons[tid]

        stale_waste = [
            tid for tid, obj in self.waste_objects.items()
            if obj.is_stale(self.frame_count)
        ]
        for tid in stale_waste:
            del self.waste_objects[tid]

        # Classify waste
        new_waste = []
        pre_existing_waste = []
        for tid, obj in self.waste_objects.items():
            if obj.is_new and tid not in self.startup_waste_ids:
                new_waste.append(obj)
            else:
                pre_existing_waste.append(obj)

        # Pass through cars from detector (no tracking); analyzer uses for car-litter
        cars = detections.get("cars", [])

        return {
            "persons": self.persons,
            "waste": self.waste_objects,
            "new_waste": new_waste,
            "pre_existing_waste": pre_existing_waste,
            "frame_count": self.frame_count,
            "cars": cars,
        }

    def is_waste_persistent(self, waste_obj):
        """
        Check if a waste object has persisted long enough to be considered dumped.
        It must have been seen for at least WASTE_PERSISTENCE_FRAMES.
        """
        return waste_obj.frames_seen >= WASTE_PERSISTENCE_FRAMES

    def find_nearby_person(self, waste_obj, max_distance=None):
        """
        Find the closest person to a waste object.

        Returns:
            TrackedObject of the nearest person, or None if no person is nearby.
        """
        from config import PROXIMITY_THRESHOLD
        if max_distance is None:
            max_distance = PROXIMITY_THRESHOLD

        waste_center = waste_obj.center()
        closest_person = None
        closest_dist = float("inf")

        for tid, person in self.persons.items():
            person_center = person.center()
            dist = (
                (waste_center[0] - person_center[0]) ** 2 +
                (waste_center[1] - person_center[1]) ** 2
            ) ** 0.5

            if dist < closest_dist:
                closest_dist = dist
                closest_person = person

        if closest_person and closest_dist <= max_distance:
            return closest_person
        return None

    def reset(self):
        """Reset all tracking state."""
        self.persons.clear()
        self.waste_objects.clear()
        self.frame_count = 0
        self.startup_waste_ids.clear()
        self.region_waste_history.clear()

    def _get_region(self, box, grid_size=100):
        """Map a bounding box to a grid region for spatial indexing."""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return (cx // grid_size, cy // grid_size)

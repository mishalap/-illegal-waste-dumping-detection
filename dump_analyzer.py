import time
import math
import numpy as np
from collections import defaultdict
from config import (
    PROXIMITY_THRESHOLD, WEIGHT_DETECTION, WEIGHT_TEMPORAL, WEIGHT_ACCUMULATION,
    WASTE_PERSISTENCE_FRAMES, DEFAULT_SAFE_ZONES, CAR_PROXIMITY_THRESHOLD,
    CAR_LITTER_WASTE_CLASS_NAMES, CAR_LITTER_MAX_WASTE_AGE_FRAMES,
    # Same incident logic as video upload (one event = one incident)
    VIDEO_STATIONARY_THRESHOLD, VIDEO_PERSON_NEAR_FRAMES,
    VIDEO_PERSON_FAR_CONSECUTIVE_FRAMES, VIDEO_PERSON_LEFT_FRAMES,
    VIDEO_WASTE_STATIONARY_NEAR_PERSON_FRAMES,
    VIDEO_COOLDOWN_SECONDS, VIDEO_COOLDOWN_REGION_GRID_SIZE,
    VIDEO_CAR_LITTER_ACCUMULATION_FRAMES,
)


INCIDENT_TYPE_PERSON_DUMP = "person_dump"
INCIDENT_TYPE_CAR_LITTER = "car_litter"


class DumpEvent:
    """Tracks the accumulation of evidence for a potential dump event using a state machine."""

    incident_type = INCIDENT_TYPE_PERSON_DUMP

    def __init__(self, waste_track_id, person_track_id, initial_center):
        self.waste_track_id = waste_track_id
        self.person_track_id = person_track_id
        self.accumulated_frames = 0
        self.detection_confidences = []
        self.temporal_scores = []
        self.first_detected = time.time()
        
        # State Machine Variables
        self.person_is_near = True
        self.person_was_near_count = 0
        self.person_has_left = False
        self.frames_since_left = 0
        self.person_far_consecutive_frames = 0  # must reach threshold before we set person_has_left
        self.initial_waste_center = initial_center
        self.is_stationary = True
        self.confirmed = False
        # Waste must be seen stationary (not moving with person) while person was near = "placed", not "carried"
        self.waste_stationary_near_person_count = 0

    def update_state(self, is_near, current_center, detection_conf, temporal_score):
        """Update the state machine with current frame information."""
        self.person_is_near = is_near

        # 1. How much did the waste move from its reference position?
        dist_moved = math.sqrt(
            (current_center[0] - self.initial_waste_center[0])**2 +
            (current_center[1] - self.initial_waste_center[1])**2
        )
        if dist_moved > VIDEO_STATIONARY_THRESHOLD:
            self.is_stationary = False

        # 2. While person is near: did waste stay put (placed) or move (carried)?
        if is_near:
            self.person_was_near_count += 1
            self.frames_since_left = 0
            self.person_has_left = False
            self.person_far_consecutive_frames = 0
            if dist_moved <= VIDEO_STATIONARY_THRESHOLD:
                self.waste_stationary_near_person_count += 1
            else:
                self.waste_stationary_near_person_count = 0  # waste moving = being carried
            self.accumulated_frames += 1
            self.detection_confidences.append(detection_conf)
            self.temporal_scores.append(temporal_score)
        else:
            # 3. Person is far: require CONSECUTIVE far frames before we consider "person left"
            self.person_far_consecutive_frames += 1
            if self.person_was_near_count >= VIDEO_PERSON_NEAR_FRAMES:
                if self.person_far_consecutive_frames >= VIDEO_PERSON_FAR_CONSECUTIVE_FRAMES:
                    if not self.person_has_left:
                        self.person_has_left = True
                        self.frames_since_left = 0
                    self.frames_since_left += 1

    def reset(self):
        """Reset accumulation if conditions stop being met."""
        self.accumulated_frames = 0
        self.person_was_near_count = 0
        self.frames_since_left = 0
        self.person_has_left = False
        self.person_far_consecutive_frames = 0
        self.waste_stationary_near_person_count = 0
        self.detection_confidences.clear()
        self.temporal_scores.clear()

    def is_confirmed(self):
        """
        Confirm ONLY if (same logic as video upload):
        - Person was near for enough frames
        - Waste was seen stationary (not moving with person) while person was near = placed, not carried
        - Person has been far for enough consecutive frames, then left for VIDEO_PERSON_LEFT_FRAMES
        - Waste has remained stationary after person left
        """
        return (
            self.person_has_left and
            self.frames_since_left >= VIDEO_PERSON_LEFT_FRAMES and
            self.is_stationary and
            self.waste_stationary_near_person_count >= VIDEO_WASTE_STATIONARY_NEAR_PERSON_FRAMES
        )

    def final_confidence(self):
        """Calculate weighted final confidence score."""
        if not self.detection_confidences:
            return 0.0

        avg_detection = sum(self.detection_confidences) / len(self.detection_confidences)
        avg_temporal = sum(self.temporal_scores) / len(self.temporal_scores) if self.temporal_scores else 0.0
        # For simplicity, keep similar scoring but ensure it reflects the state
        score = (
            WEIGHT_DETECTION * avg_detection +
            WEIGHT_TEMPORAL * avg_temporal +
            WEIGHT_ACCUMULATION * 1.0  # Already passed accumulation threshold
        )
        return round(min(score, 1.0), 3)


class CarLitterEvent:
    """Tracks waste near a vehicle for car garbage-throwing detection."""

    incident_type = INCIDENT_TYPE_CAR_LITTER

    def __init__(self, waste_track_id, car_index, initial_center):
        self.waste_track_id = waste_track_id
        self.car_index = car_index
        self.accumulated_frames = 0
        self.detection_confidences = []
        self.initial_center = initial_center
        self.confirmed = False
        self.first_detected = time.time()

    def update(self, current_center, detection_conf):
        self.accumulated_frames += 1
        self.detection_confidences.append(detection_conf)

    def reset(self):
        """Reset evidence so car-near accumulation must be consecutive."""
        self.accumulated_frames = 0
        self.detection_confidences.clear()

    def is_confirmed(self):
        return self.accumulated_frames >= VIDEO_CAR_LITTER_ACCUMULATION_FRAMES

    def final_confidence(self):
        if not self.detection_confidences:
            return 0.0
        avg = sum(self.detection_confidences) / len(self.detection_confidences)
        acc_factor = min(self.accumulated_frames / VIDEO_CAR_LITTER_ACCUMULATION_FRAMES, 1.0)
        return round(min(avg * 0.7 + acc_factor * 0.3, 1.0), 3)


class DumpAnalyzer:
    """
    Core intelligence module that combines detection, tracking, and
    temporal data to determine if an illegal dumping event has occurred.

    Logic: 
    1. Person near new waste for N frames (STATIONARY check starts).
    2. Person moves away from waste.
    3. Waste remains stationary for M frames after person left.
    4. Trigger!
    """

    def __init__(self, safe_zones=None):
        self.active_events = {}  # key: (waste_track_id, person_track_id) -> DumpEvent
        self.car_litter_events = {}  # key: waste_track_id -> CarLitterEvent
        self.cooldown_log = {}   # key: region_key -> last_incident_time
        self.safe_zones = safe_zones or DEFAULT_SAFE_ZONES

    def analyze(self, tracked_data):
        """
        Analyze current frame data to detect dump events.

        Args:
            tracked_data: dict from ObjectTracker.update() with persons, new_waste, etc.

        Returns:
            list of confirmed DumpEvent objects (may be empty)
        """
        confirmed_events = []
        new_waste = tracked_data["new_waste"]
        persons = tracked_data["persons"]

        # Track waste visibility in this frame.
        active_new_waste_ids = {w.track_id for w in new_waste}
        active_all_waste_ids = set(tracked_data.get("waste", {}).keys())

        # Check each new waste object
        for waste_obj in new_waste:
            # Skip if in a safe zone
            if self._is_in_safe_zone(waste_obj.box):
                continue

            # Find nearby person
            closest_person = None
            closest_dist = float("inf")
            for pid, person in persons.items():
                dist = self._distance(waste_obj.center(), person.center())
                if dist < closest_dist:
                    closest_dist = dist
                    closest_person = person

            is_near = (closest_person is not None and closest_dist <= PROXIMITY_THRESHOLD)

            # Get or create event
            # We track by waste ID and the person who was MOST RECENTLY near it
            # To handle multiple people, we iterate active events later
            event_key = waste_obj.track_id
            
            if is_near:
                person_id = closest_person.track_id
                if event_key not in self.active_events:
                    self.active_events[event_key] = DumpEvent(
                        waste_obj.track_id, person_id, waste_obj.center()
                    )
                
                event = self.active_events[event_key]
                # If a DIFFERENT person is now nearer, we update the person_track_id?
                # Actually, ByteTrack IDs should be stable. Let's keep the first one 
                # or the closest one.
                event.person_track_id = person_id 
                event.update_state(True, waste_obj.center(), waste_obj.confidence, self._temporal_score(waste_obj))
            elif event_key in self.active_events:
                # Person left
                event = self.active_events[event_key]
                event.update_state(False, waste_obj.center(), 0, 0) # Confidences don't matter as much once left

            # Check if confirmed
            if event_key in self.active_events:
                event = self.active_events[event_key]
                if event.is_confirmed() and not event.confirmed:
                    # Final stationary verification (waste must exist and not have moved)
                    if event.is_stationary:
                        # Check cooldown
                        region_key = self._region_key(waste_obj.box)
                        if not self._is_in_cooldown(region_key):
                            event.confirmed = True
                            self.cooldown_log[region_key] = time.time()
                            confirmed_events.append(event)
                
                # If waste moved too much while person was near OR after, reset/invalidate
                if not event.is_stationary:
                    self.active_events[event_key].reset()
                    self.active_events[event_key].is_stationary = True # Reset stationary flag for next attempt
                    self.active_events[event_key].initial_waste_center = waste_obj.center()

        # Clean up events for waste that is NO LONGER detected
        # (Wait... if it's no longer detected, maybe it was picked up? That's fine.)
        keys_to_remove = [k for k in self.active_events if k not in active_new_waste_ids]
        for k in keys_to_remove:
            # If the person left and THEN the waste disappeared, maybe it was a false positive detection
            # or it was picked up. We don't trigger if it disappeared.
            del self.active_events[k]

        # ---- Car-litter: recent waste near vehicle for CONSECUTIVE frames ----
        cars = tracked_data.get("cars", [])
        all_waste = tracked_data.get("waste", {}).values()
        for waste_obj in all_waste:
            event_key = waste_obj.track_id

            if self._is_in_safe_zone(waste_obj.box):
                if event_key in self.car_litter_events:
                    del self.car_litter_events[event_key]
                continue

            # Restrict to plausible disposable objects for vehicle litter incidents.
            if waste_obj.class_name not in CAR_LITTER_WASTE_CLASS_NAMES:
                if event_key in self.car_litter_events:
                    del self.car_litter_events[event_key]
                continue

            # Ignore old/pre-existing tracks; car-litter should be tied to new appearance.
            if waste_obj.frames_seen > CAR_LITTER_MAX_WASTE_AGE_FRAMES:
                if event_key in self.car_litter_events:
                    del self.car_litter_events[event_key]
                continue

            waste_center = waste_obj.center()
            near_any_car = False
            for car_idx, car in enumerate(cars):
                car_box = car["box"]
                car_center = ((car_box[0] + car_box[2]) // 2, (car_box[1] + car_box[3]) // 2)
                dist = self._distance(waste_center, car_center)
                if dist <= CAR_PROXIMITY_THRESHOLD:
                    near_any_car = True
                    if event_key not in self.car_litter_events:
                        self.car_litter_events[event_key] = CarLitterEvent(
                            waste_obj.track_id, car_idx, waste_center
                        )
                    ev = self.car_litter_events[event_key]
                    ev.update(waste_obj.center(), waste_obj.confidence)
                    if ev.is_confirmed() and not ev.confirmed:
                        region_key = self._region_key(waste_obj.box)
                        if not self._is_in_cooldown(region_key):
                            ev.confirmed = True
                            self.cooldown_log[region_key] = time.time()
                            confirmed_events.append(ev)
                    break  # one car per waste
            if not near_any_car and event_key in self.car_litter_events:
                # Must be consecutive near-car frames.
                self.car_litter_events[event_key].reset()

        # Clean up car_litter_events for waste no longer detected
        for wid in list(self.car_litter_events.keys()):
            if wid not in active_all_waste_ids:
                del self.car_litter_events[wid]

        # Clean up stale/old events
        self._cleanup_stale_events()

        return confirmed_events

    def _temporal_score(self, waste_obj):
        """
        Score based on how "new" the waste object is and how long it has persisted.
        Returns 0.0 to 1.0.
        """
        if not waste_obj.is_new:
            return 0.0

        # Higher score if waste has persisted for more frames
        persistence_ratio = min(waste_obj.frames_seen / WASTE_PERSISTENCE_FRAMES, 1.0)
        return persistence_ratio
 
    def _is_in_safe_zone(self, box):
        """Check if a bounding box center falls within any safe zone."""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2

        for zone in self.safe_zones:
            if (zone["x1"] <= cx <= zone["x2"] and
                    zone["y1"] <= cy <= zone["y2"]):
                return True
        return False

    def _is_in_cooldown(self, region_key):
        """Check if a region is still in cooldown after a recent incident (same as video)."""
        if region_key not in self.cooldown_log:
            return False
        elapsed = time.time() - self.cooldown_log[region_key]
        return elapsed < VIDEO_COOLDOWN_SECONDS

    def _region_key(self, box, grid_size=VIDEO_COOLDOWN_REGION_GRID_SIZE):
        """Map a bounding box to a region key for cooldown tracking."""
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        return (cx // grid_size, cy // grid_size)

    def _cleanup_stale_events(self, max_age_seconds=60):
        """Remove events that are too old and never confirmed."""
        now = time.time()
        keys_to_remove = []
        for key, event in self.active_events.items():
            if event.confirmed:
                keys_to_remove.append(key)
            elif now - event.first_detected > max_age_seconds:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.active_events[key]
        # Stale car-litter events
        keys_to_remove = []
        for key, event in self.car_litter_events.items():
            if event.confirmed or (now - event.first_detected > max_age_seconds):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.car_litter_events[key]

    @staticmethod
    def _distance(p1, p2):
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def reset(self):
        """Reset all analyzer state."""
        self.active_events.clear()
        self.car_litter_events.clear()
        self.cooldown_log.clear()

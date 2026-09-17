"""
Video Processor — Analyzes uploaded video files for garbage dumping events.

Processes frames at a configurable sample rate. Uses multi-frame confirmation:
- Incident only when a person was near an object, the object stayed stationary,
  and the person moved away (object left behind). If the object moves with the
  person until they leave frame, no incident is marked.
- No incident if no waste object is detected (person alone does not trigger).
- Cooldown and region-based dedup ensure one event produces one incident.
"""

import os
import math
import cv2
from datetime import datetime

from config import (
    EVIDENCE_DIR, VIDEO_FRAME_SAMPLE_RATE,
    PROXIMITY_THRESHOLD, CAR_PROXIMITY_THRESHOLD,
    VIDEO_COOLDOWN_SECONDS, VIDEO_COOLDOWN_REGION_GRID_SIZE,
    VIDEO_STATIONARY_THRESHOLD,
    VIDEO_PERSON_NEAR_FRAMES, VIDEO_PERSON_FAR_CONSECUTIVE_FRAMES,
    VIDEO_PERSON_LEFT_FRAMES, VIDEO_WASTE_STATIONARY_NEAR_PERSON_FRAMES,
    VIDEO_MATCH_DISTANCE, VIDEO_CAR_LITTER_ACCUMULATION_FRAMES,
    CAR_LITTER_WASTE_CLASS_NAMES, VIDEO_CAR_LITTER_MAX_TRACK_AGE,
)
from detector import Detector
from database import update_video_upload, insert_video_detection, insert_incident, get_video_upload_camera_id


def _center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _region_key(box, grid_size=VIDEO_COOLDOWN_REGION_GRID_SIZE):
    cx, cy = _center(box)
    return (cx // grid_size, cy // grid_size)


def _draw_video_detections(frame, persons, waste, cars, incidents_found, frame_number, timestamp_in_video):
    """Draw lightweight detection overlays for UI playback preview."""
    out = frame.copy()

    for p in persons:
        x1, y1, x2, y2 = p["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"Person {p['confidence']:.2f}", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for w in waste:
        x1, y1, x2, y2 = w["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"{w['class_name']} {w['confidence']:.2f}", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for c in cars:
        x1, y1, x2, y2 = c["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(out, c["class_name"], (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    status = (
        f"Frame: {frame_number} | Time: {timestamp_in_video:.1f}s | "
        f"Persons: {len(persons)} | Waste: {len(waste)} | Vehicles: {len(cars)} | "
        f"Incidents: {incidents_found}"
    )
    cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


class VideoWasteTrack:
    """
    Tracks one waste object across sampled frames. Confirms incident only when:
    - Person was near for enough frames,
    - Waste was stationary (not moving with person),
    - Person moved away and stayed away for enough frames.
    """

    def __init__(self, track_id, box, class_name, confidence, initial_center):
        self.track_id = track_id
        self.box = box
        self.class_name = class_name
        self.confidence = confidence
        self.initial_center = initial_center
        self.last_center = initial_center
        self.seen_frames = 1
        self.person_near_count = 0
        self.stationary_near_person_count = 0
        self.person_far_consecutive = 0
        self.person_has_left = False
        self.frames_since_left = 0
        self.is_stationary = True
        self.confirmed = False

    def update_state(self, person_near, current_center, detection_conf):
        dist_moved = _distance(current_center, self.initial_center)
        if dist_moved > VIDEO_STATIONARY_THRESHOLD:
            self.is_stationary = False

        if person_near:
            self.person_far_consecutive = 0
            self.person_has_left = False
            self.frames_since_left = 0
            self.person_near_count += 1
            if dist_moved <= VIDEO_STATIONARY_THRESHOLD:
                self.stationary_near_person_count += 1
            else:
                self.stationary_near_person_count = 0
        else:
            self.person_far_consecutive += 1
            if self.person_near_count >= VIDEO_PERSON_NEAR_FRAMES:
                if self.person_far_consecutive >= VIDEO_PERSON_FAR_CONSECUTIVE_FRAMES:
                    if not self.person_has_left:
                        self.person_has_left = True
                        self.frames_since_left = 0
                    self.frames_since_left += 1

    def is_confirmed(self):
        return (
            self.person_has_left
            and self.frames_since_left >= VIDEO_PERSON_LEFT_FRAMES
            and self.is_stationary
            and self.stationary_near_person_count >= VIDEO_WASTE_STATIONARY_NEAR_PERSON_FRAMES
        )

    def reset_motion(self, new_center):
        """Reset accumulation when waste moves too much (e.g. carried away)."""
        self.person_near_count = 0
        self.stationary_near_person_count = 0
        self.person_far_consecutive = 0
        self.person_has_left = False
        self.frames_since_left = 0
        self.is_stationary = True
        self.initial_center = new_center


def process_video(upload_id, video_path, progress_callback=None, frame_callback=None):
    """
    Process an uploaded video file for garbage dump detection.

    Logic (mobile / short-range footage):
    - Only report incident when: person was near an object, object stayed
      stationary in frame, and person moved away (object left behind).
    - If object moves with person (e.g. carried), no incident.
    - No incident if no waste object is detected.
    - Cooldown ensures one event produces one incident per region.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        update_video_upload(upload_id, status="failed")
        return {"total_frames": 0, "processed_frames": 0,
                "incidents_found": 0, "detections": []}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    update_video_upload(upload_id, status="processing", processed_frames=0)

    detector = Detector()

    processed_count = 0
    incidents_found = 0
    all_detections = []

    # Waste tracks: track_id -> VideoWasteTrack (matched across frames by position)
    waste_tracks = {}
    next_waste_id = 0
    # Cooldown: region_key -> last incident timestamp (in video seconds)
    cooldown_log = {}
    # Car-litter: simple accumulation per "waste track" + cooldown (one incident per event)
    car_litter_frames = {}  # waste_track_id -> frames near car (we'll key by matched track)
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % VIDEO_FRAME_SAMPLE_RATE != 0:
                continue

            processed_count += 1
            timestamp_in_video = frame_number / fps

            detections = detector.detect_for_video(frame)
            persons = detections["persons"]
            waste = detections["waste"]
            cars = detections.get("cars", [])

            # ---- Person-dump: only if waste is present; confirm when object stationary + person left ----
            person_centers = [_center(p["box"]) for p in persons]
            current_waste_centers = {}  # track_id -> center for cleanup

            # Match each waste detection to nearest track by last position (greedy)
            used_track_ids = set()
            for w in waste:
                w_center = _center(w["box"])
                w_box = w["box"]
                w_class = w["class_name"]
                w_conf = w["confidence"]

                best_id = None
                best_dist = VIDEO_MATCH_DISTANCE + 1
                for tid, tr in waste_tracks.items():
                    if tid in used_track_ids:
                        continue
                    d = _distance(tr.last_center, w_center)
                    if d <= VIDEO_MATCH_DISTANCE and d < best_dist:
                        best_dist = d
                        best_id = tid

                track_existed = best_id is not None
                if best_id is None:
                    best_id = next_waste_id
                    next_waste_id += 1
                    waste_tracks[best_id] = VideoWasteTrack(
                        best_id, w_box, w_class, w_conf, w_center
                    )
                used_track_ids.add(best_id)

                tr = waste_tracks[best_id]
                if track_existed:
                    tr.seen_frames += 1
                tr.box = w_box
                tr.class_name = w_class
                tr.confidence = w_conf
                tr.last_center = w_center
                current_waste_centers[best_id] = w_center

                # Is any person near this waste?
                person_near = False
                if person_centers:
                    min_dist = min(_distance(w_center, pc) for pc in person_centers)
                    person_near = min_dist <= PROXIMITY_THRESHOLD

                tr.update_state(person_near, w_center, w_conf)

                if tr.is_confirmed() and not tr.confirmed:
                    region_key = _region_key(w_box)
                    last_time = cooldown_log.get(region_key, -1e9)
                    if timestamp_in_video - last_time >= VIDEO_COOLDOWN_SECONDS:
                        tr.confirmed = True
                        cooldown_log[region_key] = timestamp_in_video
                        snapshot_path = _save_video_snapshot(frame, upload_id, frame_number)
                        insert_video_detection(
                            upload_id=upload_id,
                            frame_number=frame_number,
                            timestamp_in_video=round(timestamp_in_video, 2),
                            confidence=round(tr.confidence, 3),
                            snapshot_path=snapshot_path,
                            objects_detected=[tr.class_name],
                            incident_type="person_dump"
                        )
                        camera_id = get_video_upload_camera_id()
                        insert_incident(
                            camera_id=camera_id,
                            confidence=round(tr.confidence, 3),
                            snapshot_path=snapshot_path,
                            description=f"Video upload — frame {frame_number}",
                            objects_detected=[tr.class_name],
                            incident_type="person_dump"
                        )
                        incidents_found += 1
                        all_detections.append({
                            "frame_number": frame_number,
                            "timestamp": round(timestamp_in_video, 2),
                            "confidence": round(tr.confidence, 3),
                            "objects": [tr.class_name],
                            "snapshot_path": snapshot_path,
                            "incident_type": "person_dump"
                        })

                if not tr.is_stationary:
                    tr.reset_motion(w_center)

            # Remove tracks not seen this frame (waste left or occluded)
            for tid in list(waste_tracks.keys()):
                if tid not in current_waste_centers:
                    del waste_tracks[tid]
                    if tid in car_litter_frames:
                        del car_litter_frames[tid]

            # ---- Car-litter: recent waste near vehicle for CONSECUTIVE frames + cooldown ----
            if cars and waste:
                car_centers = [_center(c["box"]) for c in cars]
                for w in waste:
                    w_center = _center(w["box"])
                    if w["class_name"] not in CAR_LITTER_WASTE_CLASS_NAMES:
                        continue

                    near_car = any(_distance(w_center, cc) <= CAR_PROXIMITY_THRESHOLD for cc in car_centers)

                    # Same track as person-dump (match by last_center)
                    match_id = None
                    best_d = VIDEO_MATCH_DISTANCE + 1
                    for tid, tr in waste_tracks.items():
                        d = _distance(tr.last_center, w_center)
                        if d <= VIDEO_MATCH_DISTANCE and d < best_d:
                            best_d = d
                            match_id = tid
                    if match_id is None:
                        continue

                    tr = waste_tracks.get(match_id)
                    if tr is None:
                        continue

                    # Treat only recent tracks as "new throw" candidates.
                    if tr.seen_frames > VIDEO_CAR_LITTER_MAX_TRACK_AGE:
                        car_litter_frames.pop(match_id, None)
                        continue

                    if not near_car:
                        # Must be consecutive near-car frames.
                        car_litter_frames.pop(match_id, None)
                        continue

                    car_litter_frames[match_id] = car_litter_frames.get(match_id, 0) + 1
                    if car_litter_frames[match_id] >= VIDEO_CAR_LITTER_ACCUMULATION_FRAMES:
                        region_key = _region_key(w["box"])
                        last_time = cooldown_log.get(region_key, -1e9)
                        if timestamp_in_video - last_time >= VIDEO_COOLDOWN_SECONDS:
                            cooldown_log[region_key] = timestamp_in_video
                            conf = (w["confidence"] + 0.5) / 2
                            snapshot_path = _save_video_snapshot(frame, upload_id, frame_number)
                            insert_video_detection(
                                upload_id=upload_id,
                                frame_number=frame_number,
                                timestamp_in_video=round(timestamp_in_video, 2),
                                confidence=round(conf, 3),
                                snapshot_path=snapshot_path,
                                objects_detected=[w["class_name"], "vehicle"],
                                incident_type="car_litter"
                            )
                            camera_id = get_video_upload_camera_id()
                            insert_incident(
                                camera_id=camera_id,
                                confidence=round(conf, 3),
                                snapshot_path=snapshot_path,
                                description=f"Video upload — frame {frame_number}",
                                objects_detected=[w["class_name"], "vehicle"],
                                incident_type="car_litter"
                            )
                            incidents_found += 1
                            all_detections.append({
                                "frame_number": frame_number,
                                "timestamp": round(timestamp_in_video, 2),
                                "confidence": round(conf, 3),
                                "objects": [w["class_name"], "vehicle"],
                                "snapshot_path": snapshot_path,
                                "incident_type": "car_litter"
                            })
                            car_litter_frames[match_id] = 0
            elif waste and not cars:
                # No vehicle in this sampled frame means near-car streak is broken.
                for tid in list(car_litter_frames.keys()):
                    if tid in current_waste_centers:
                        del car_litter_frames[tid]

            update_video_upload(
                upload_id,
                processed_frames=processed_count,
                incidents_found=incidents_found
            )
            if progress_callback:
                progress_callback(processed_count, max(1, total_frames // VIDEO_FRAME_SAMPLE_RATE))

            if frame_callback:
                preview = _draw_video_detections(
                    frame=frame,
                    persons=persons,
                    waste=waste,
                    cars=cars,
                    incidents_found=incidents_found,
                    frame_number=frame_number,
                    timestamp_in_video=timestamp_in_video,
                )
                frame_callback(preview, frame_number, timestamp_in_video, incidents_found)

    except Exception as e:
        print(f"[VideoProcessor] Error processing video: {e}")
        update_video_upload(upload_id, status="failed")
        raise
    finally:
        cap.release()

    update_video_upload(upload_id, status="completed",
                      processed_frames=processed_count,
                      incidents_found=incidents_found)

    return {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "incidents_found": incidents_found,
        "detections": all_detections
    }


def _save_video_snapshot(frame, upload_id, frame_number):
    """Save a detection snapshot from video processing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{upload_id}_frame_{frame_number}_{timestamp}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath

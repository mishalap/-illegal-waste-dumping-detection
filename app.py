"""
AI-Based Illegal Garbage Dumping Detection System — Streamlit Dashboard

Run with: streamlit run app.py
"""

import os
import sys
import logging
import warnings

# Suppress Streamlit "missing ScriptRunContext" (safe to ignore; Streamlit says so)
warnings.filterwarnings(
    "ignore",
    message=".*missing ScriptRunContext.*",
    category=UserWarning,
    module="streamlit",
)


class _ScriptRunContextFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage() or ""
        return "missing ScriptRunContext" not in msg


# Apply to root logger so Streamlit's log messages are filtered
logging.getLogger().addFilter(_ScriptRunContextFilter())

import streamlit as st
import json
import subprocess
import platform
import time
import uuid
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import DEVICE, DETECTION_MODEL, UPLOADS_DIR, EVIDENCE_DIR
from database import (
    init_db, get_cameras, get_camera, get_incident_count, get_incident_count_by_camera,
    get_incidents, get_incident, get_video_uploads, get_video_upload, get_video_detections,
    VIDEO_UPLOAD_SOURCE_TYPE,
)
from camera_manager import add_camera, remove_camera, list_cameras, get_camera_source, test_camera_source

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Garbage Dump Detection System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
init_db()

# Session state for monitor processes
if "monitor_processes" not in st.session_state:
    st.session_state.monitor_processes = {}


def _safe_incident_display(incident):
    """Normalize incident dict for display; sqlite3.Row can sometimes return bytes."""
    out = dict(incident)
    for key in ("id", "camera_id", "description", "snapshot_path", "objects_detected", "incident_type", "timestamp"):
        if key in out and out[key] is not None and isinstance(out[key], bytes):
            out[key] = out[key].decode("utf-8", errors="replace")
    if out.get("confidence") is not None and isinstance(out["confidence"], bytes):
        try:
            out["confidence"] = float(out["confidence"].decode("utf-8"))
        except (ValueError, TypeError):
            out["confidence"] = None
    return out


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("🗑️ Garbage Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "📹 Live Monitor", "📤 Video Upload", "📷 Camera Management", "🔍 Incident Viewer"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Device:** `{DEVICE.upper()}`")
st.sidebar.markdown(f"**Model:** `{DETECTION_MODEL}`")


# ============================================================
# Dashboard Page
# ============================================================
def page_dashboard():
    st.title("📊 Dashboard")
    st.markdown("Real-time overview of garbage dumping detection system.")

    # Today's total incidents (primary metric)
    today_incidents = get_incident_count(today_only=True)
    total_incidents = get_incident_count(today_only=False)
    cameras = get_cameras()

    st.subheader("📅 Today's overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Incidents reported today", today_incidents)
    with col2:
        st.metric("Total incidents (all time)", total_incidents)
    with col3:
        st.metric("Registered locations (cameras)", len(cameras))

    st.markdown("---")

    # Incidents by location (each camera = one location)
    st.subheader("📍 Incidents by location (camera)")
    by_camera_today = get_incident_count_by_camera(today_only=True)
    by_camera_all = get_incident_count_by_camera(today_only=False)

    if by_camera_today:
        # Cards: one per location (camera_id), 4 per row
        n = len(by_camera_today)
        for start in range(0, n, 4):
            row_slice = by_camera_today[start : start + 4]
            cols = st.columns(len(row_slice))
            for col, row in zip(cols, row_slice):
                with col:
                    loc_name = row["camera_name"] or f"Camera {row['camera_id']}"
                    label = f"**{loc_name}**" + (f" (ID: {row['camera_id']})" if row["camera_id"] is not None else "")
                    st.metric(label=label, value=row["count"], delta="today")
    else:
        st.info("No locations with incidents today. Registered cameras will appear here when incidents are reported.")

    # Analytics: which location has highest incidents
    st.markdown("---")
    st.subheader("📈 Location analytics")
    if by_camera_all:
        top = by_camera_all[0]
        top_name = top["camera_name"] or f"Camera {top['camera_id']}"
        st.markdown(f"**Highest incidents (all time):** **{top_name}** with **{top['count']}** incidents.")
        # Bar chart: incidents per location
        df_loc = pd.DataFrame(by_camera_all)
        df_loc["location"] = df_loc.apply(
            lambda r: (r["camera_name"] or f"Camera {r['camera_id']}") + (f" (ID:{r['camera_id']})" if r["camera_id"] is not None else ""),
            axis=1
        )
        fig = px.bar(
            df_loc, x="location", y="count",
            title="Incidents per location (all time)",
            labels={"location": "Location (camera)", "count": "Incidents"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No incident data by location yet. Analytics will appear once incidents are reported.")

    st.markdown("---")

    # Recent incidents
    st.subheader("Recent Incidents")
    incidents = get_incidents(limit=20)

    if incidents:
        for incident in incidents:
            inc = _safe_incident_display(incident)
            itype = inc.get("incident_type") or "person_dump"
            type_label = "🚗 Car litter" if itype == "car_litter" else "🚶 Person dump"
            conf_str = f"{inc['confidence']:.1%}" if inc.get("confidence") is not None else "N/A"
            cam = get_camera(inc["camera_id"]) if inc.get("camera_id") else None
            location_label = f"{cam['name']} (ID: {inc['camera_id']})" if cam else (f"ID: {inc.get('camera_id')}" if inc.get('camera_id') else "Unknown / Video")
            with st.expander(
                f"🚨 Incident #{inc['id']} — {type_label} — {inc['timestamp']} — {location_label} — Confidence: {conf_str}"
            ):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    if inc.get("snapshot_path") and os.path.exists(inc["snapshot_path"]):
                        img = Image.open(inc["snapshot_path"])
                        st.image(img, caption="Evidence Snapshot", use_container_width=True, width='stretch')
                    else:
                        st.info("No snapshot available")

                with col_info:
                    st.markdown(f"**Type:** {type_label}")
                    st.markdown(f"**Description:** {inc.get('description', '')}")
                    st.markdown(f"**Location (camera):** {location_label}")
                    st.markdown(f"**Confidence:** {conf_str}")
                    if inc.get("objects_detected"):
                        try:
                            objects = json.loads(inc["objects_detected"])
                            st.markdown(f"**Objects Detected:** {', '.join(objects)}")
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(f"**Objects Detected:** {inc['objects_detected']}")
    else:
        st.info("No incidents detected yet. Start a live monitor or upload a video to begin detection.")

    # Incident trend chart
    if incidents:
        st.markdown("---")
        st.subheader("Incident Trend")

        incidents_safe = [_safe_incident_display(i) for i in incidents]
        df = pd.DataFrame(incidents_safe)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily_counts = df.groupby("date").size().reset_index(name="incidents")
        fig = px.bar(daily_counts, x="date", y="incidents",
                     title="Incidents Per Day",
                     labels={"date": "Date", "incidents": "Number of Incidents"})
        fig.update_layout(xaxis_title="Date", yaxis_title="Incidents")
        st.plotly_chart(fig, use_container_width=True, width='stretch')


# ============================================================
# Live Monitor Page
# ============================================================
def page_live_monitor():
    st.title("📹 Live Monitor")

    all_cameras = list_cameras()
    cameras = [c for c in all_cameras if c.get("source_type") != VIDEO_UPLOAD_SOURCE_TYPE]

    if not cameras:
        st.warning("No cameras registered. Go to **Camera Management** to add one first.")
        return

    # Camera selection
    camera_options = {f"{c['name']} ({c['source_type']})": c['id'] for c in cameras}
    selected_label = st.selectbox("Select Camera", list(camera_options.keys()))
    selected_camera_id = camera_options[selected_label]
    selected_camera = get_camera(selected_camera_id)

    # Choose mode based on camera type
    is_webcam = selected_camera["source_type"] == "webcam"

    if is_webcam:
        st.info(
            "**Webcam detected** — The live feed will open in a **separate OpenCV window** "
            "(outside the browser). Press **Q** in that window to stop."
        )
    else:
        st.info(
            "**IP / Mobile camera detected** — The live feed will stream "
            "**directly in the dashboard** below with real-time detection overlays."
        )

    st.markdown("---")

    # ---- Mode A: OpenCV window (webcam) ----
    if is_webcam:
        _render_opencv_monitor(selected_camera_id, selected_camera)
    # ---- Mode B: In-dashboard stream (IP / mobile / RTSP) ----
    else:
        _render_dashboard_stream(selected_camera_id, selected_camera)


def _render_opencv_monitor(selected_camera_id, selected_camera):
    """Launch the live_monitor.py subprocess for webcam feeds."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Start Live Monitor", type="primary", use_container_width=True, width='stretch'):
            source = get_camera_source(selected_camera_id)
            if source is None:
                st.error("Could not determine camera source.")
                return

            cmd = [
                sys.executable, "live_monitor.py",
                "--source", str(source),
                "--camera-id", str(selected_camera_id),
                "--camera-name", selected_camera["name"]
            ]

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if platform.system() == "Windows" else {})
                )
                st.session_state.monitor_processes[selected_camera_id] = {
                    "process": process,
                    "pid": process.pid,
                    "camera_name": selected_camera["name"],
                    "started_at": datetime.now().strftime("%H:%M:%S")
                }
                st.success(
                    f"✅ Live Monitor launched for **{selected_camera['name']}**! "
                    f"Look for the OpenCV window. Press **Q** in that window to stop."
                )
            except Exception as e:
                st.error(f"Failed to launch monitor: {e}")

    with col2:
        if st.button("🔴 Stop All Monitors", use_container_width=True, width='stretch'):
            stopped = 0
            for cam_id, info in list(st.session_state.monitor_processes.items()):
                proc = info["process"]
                if proc.poll() is None:
                    proc.terminate()
                    stopped += 1
                del st.session_state.monitor_processes[cam_id]
            if stopped:
                st.info(f"Stopped {stopped} monitor(s).")
            else:
                st.info("No active monitors to stop.")

    # Show active monitors
    st.markdown("---")
    st.subheader("Active Monitors")

    active = False
    stopped_cam_ids = []
    for cam_id, info in list(st.session_state.monitor_processes.items()):
        proc = info["process"]
        is_running = proc.poll() is None

        if is_running:
            active = True
            st.markdown(
                f"🟢 **{info['camera_name']}** — PID: {info['pid']} — "
                f"Started: {info['started_at']}"
            )
        else:
            st.markdown(
                f"⚫ **{info['camera_name']}** — Stopped"
            )
            stopped_cam_ids.append(cam_id)

    # Clean up stopped monitors outside the display loop
    for cam_id in stopped_cam_ids:
        del st.session_state.monitor_processes[cam_id]

    if not active:
        st.info("No active monitors. Select a camera and click Start.")


def _render_dashboard_stream(selected_camera_id, selected_camera):
    """Stream IP/mobile camera feed directly in the Streamlit dashboard."""
    import cv2
    import numpy as np

    col1, col2 = st.columns(2)
    with col1:
        start_stream = st.button("🟢 Start Live Stream", type="primary", use_container_width=True, width='stretch')
    with col2:
        stop_stream = st.button("🔴 Stop Stream", use_container_width=True, width='stretch')

    if stop_stream:
        st.session_state.pop("dashboard_streaming", None)
        st.info("Stream stopped.")
        return

    if start_stream:
        st.session_state["dashboard_streaming"] = True

    if not st.session_state.get("dashboard_streaming", False):
        st.markdown("---")
        st.info("Click **Start Live Stream** to begin viewing the camera feed with detection.")
        st.subheader("Camera Info")
        st.json({
            "ID": selected_camera["id"],
            "Name": selected_camera["name"],
            "Type": selected_camera["source_type"],
            "Source": selected_camera.get("source_url") or "N/A",
        })
        return

    # --- Streaming active ---
    st.markdown("---")

    source = get_camera_source(selected_camera_id)
    if source is None:
        st.error("Could not determine camera source.")
        return

    # Status indicators
    status_col1, status_col2, status_col3 = st.columns(3)
    status_text = status_col1.empty()
    fps_text = status_col2.empty()
    detection_text = status_col3.empty()

    status_text.markdown("🟡 **Connecting...**")

    # Frame display area
    frame_placeholder = st.empty()

    # Load models (cached in session state to avoid reloading on every rerun)
    if "detector_instance" not in st.session_state:
        with st.spinner("Loading AI models (first time only)..."):
            from detector import Detector
            st.session_state["detector_instance"] = Detector()
    
    if "tracker_instance" not in st.session_state:
        from tracker import ObjectTracker
        st.session_state["tracker_instance"] = ObjectTracker()
        
    if "analyzer_instance" not in st.session_state:
        from dump_analyzer import DumpAnalyzer
        st.session_state["analyzer_instance"] = DumpAnalyzer()

    detector = st.session_state["detector_instance"]
    tracker = st.session_state["tracker_instance"]
    analyzer = st.session_state["analyzer_instance"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"❌ Could not connect to camera: {source}")
        st.session_state.pop("dashboard_streaming", None)
        return

    # Warm-up
    for _ in range(5):
        cap.read()

    status_text.markdown("🟢 **Live**")
    prev_time = time.time()

    # Performance settings
    INFER_EVERY_N = 3          # Run AI every Nth frame; show raw feed in between
    INFER_WIDTH = 640          # Resize frame to this width before inference

    frame_count = 0
    last_persons = []
    last_waste = []
    last_cars = []
    last_alert = False
    alert_message = ""
    scale_x, scale_y = 1.0, 1.0

    scale_x, scale_y = 1.0, 1.0



    # Process frames in batches to allow Streamlit to handle UI interactions
    # (e.g., Stop button). After each batch, st.rerun() gives the event loop a chance.
    MAX_FRAMES_PER_BATCH = 100
    batch_frame_count = 0

    try:
        while st.session_state.get("dashboard_streaming", False):
            ret, frame = cap.read()
            if not ret:
                status_text.markdown("🟡 **Reconnecting...**")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(source)
                continue

            frame_count += 1
            batch_frame_count += 1

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.001)
            prev_time = now
            fps_text.markdown(f"**FPS:** {fps:.1f}")

            # Run AI inference only every Nth frame
            if frame_count % INFER_EVERY_N == 0:
                orig_h, orig_w = frame.shape[:2]
                ratio = INFER_WIDTH / orig_w
                small_h = int(orig_h * ratio)
                small = cv2.resize(frame, (INFER_WIDTH, small_h))

                scale_x = orig_w / INFER_WIDTH
                scale_y = orig_h / small_h

                detections = detector.detect(small)

                # Update tracker with detections
                tracked_data = tracker.update(detections)

                # Run analyzer BEFORE scaling display boxes; tracker/analyzer operate in inference coordinates.
                confirmed_events = analyzer.analyze(tracked_data)

                # Cache detections for drawing/logging and scale only these copies.
                last_persons = [{**d, "box": d["box"][:]} for d in detections["persons"]]
                last_waste = [{**d, "box": d["box"][:]} for d in detections["waste"]]
                last_cars = [{**d, "box": d["box"][:]} for d in detections.get("cars", [])]

                # Scale boxes back to original resolution
                for det in last_persons + last_waste + last_cars:
                    det["box"] = [
                        int(det["box"][0] * scale_x),
                        int(det["box"][1] * scale_y),
                        int(det["box"][2] * scale_x),
                        int(det["box"][3] * scale_y),
                    ]

                last_alert = len(confirmed_events) > 0
                from dump_analyzer import INCIDENT_TYPE_CAR_LITTER
                car_litter_alert = any(
                    getattr(e, "incident_type", None) == INCIDENT_TYPE_CAR_LITTER
                    for e in confirmed_events
                )
                if last_alert:
                    alert_message = "!! CAR LITTER DETECTED !!" if car_litter_alert else "!! ILLEGAL DUMPING DETECTED !!"
                    
                    # Log to database
                    from database import insert_incident
                    for event in confirmed_events:
                        incident_type = getattr(event, "incident_type", "person_dump")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snapshot_name = f"dash_{selected_camera_id}_{timestamp}.jpg"
                        snapshot_path = os.path.join(EVIDENCE_DIR, snapshot_name)
                        cv2.imwrite(snapshot_path, frame)
                        
                        objects = []
                        for w in last_waste:
                            if w["track_id"] == event.waste_track_id:
                                objects.append(w["class_name"])
                                break
                        
                        if incident_type == INCIDENT_TYPE_CAR_LITTER:
                            desc = "Automated dashboard: Garbage thrown from vehicle detected."
                        else:
                            desc = f"Automated dashboard detection: Person #{event.person_track_id} dumped waste."
                        
                        insert_incident(
                            camera_id=selected_camera_id,
                            confidence=event.final_confidence(),
                            snapshot_path=snapshot_path,
                            description=desc,
                            objects_detected=objects,
                            incident_type=incident_type
                        )

                detection_text.markdown(
                    f"**Persons:** {len(last_persons)} | **Waste:** {len(last_waste)} | **Vehicles:** {len(last_cars)}"
                )

            # Draw cached detections on current frame
            display = frame.copy()

            for det in last_persons:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"Person ({det['confidence']:.2f})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for det in last_waste:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display, f"{det['class_name']} ({det['confidence']:.2f})",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for det in last_cars:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(display, det["class_name"], (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Alert overlay
            if last_alert:
                h, ww = display.shape[:2]
                cv2.rectangle(display, (0, 0), (ww - 1, h - 1), (0, 0, 255), 6)
                cv2.putText(display, alert_message,
                            (ww // 2 - 250, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # OSD
            mode = "AI" if frame_count % INFER_EVERY_N == 0 else "PASS"
            cv2.putText(display, f"FPS: {fps:.1f} | {DEVICE.upper()} | {mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert BGR to RGB for Streamlit
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_rgb, channels="RGB", use_container_width=True, width='stretch')

            # Yield control back to Streamlit after a batch so UI interactions can be processed
            if batch_frame_count >= MAX_FRAMES_PER_BATCH:
                break

    except Exception as e:
        st.error(f"Stream error: {e}")
    finally:
        cap.release()
        # If still streaming, rerun to process next batch (allows Stop button to work)
        if st.session_state.get("dashboard_streaming", False):
            st.rerun()
        else:
            status_text.markdown("⚫ **Stopped**")


# ============================================================
# Video Upload Page
# ============================================================
def page_video_upload():
    st.title("📤 Video Upload & Analysis")
    st.markdown("Upload a video file to analyze it for garbage dumping events.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mkv", "mov", "wmv"],
        help="Supported formats: MP4, AVI, MKV, MOV, WMV"
    )

    if uploaded_file is not None:
        # Cache the saved file to avoid re-saving on every Streamlit rerun
        upload_cache_key = f"upload_cache_{uploaded_file.file_id}"
        if upload_cache_key not in st.session_state:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            unique_name = f"{uuid.uuid4().hex}{file_ext}"
            save_path = os.path.join(UPLOADS_DIR, unique_name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state[upload_cache_key] = {
                "save_path": save_path,
                "unique_name": unique_name
            }
        else:
            save_path = st.session_state[upload_cache_key]["save_path"]
            unique_name = st.session_state[upload_cache_key]["unique_name"]

        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.1f} MB)")

        # Process button
        if st.button("🔍 Analyze Video", type="primary"):
            from database import insert_video_upload
            import cv2

            # Get total frames
            cap = cv2.VideoCapture(save_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            upload_id = insert_video_upload(unique_name, uploaded_file.name, total_frames)

            st.markdown("---")
            st.subheader("Processing...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            preview_text = st.empty()
            preview_frame = st.empty()

            def progress_callback(processed, total):
                if total > 0:
                    pct = min(processed / total, 1.0)
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {processed}/{total} sampled frames...")

            def frame_callback(frame_bgr, frame_number, timestamp_in_video, incidents_found):
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                preview_text.markdown(
                    f"**Live Detection Preview**  |  Frame: `{frame_number}`  |  "
                    f"Time: `{timestamp_in_video:.1f}s`  |  Incidents: `{incidents_found}`"
                )
                preview_frame.image(frame_rgb, channels="RGB", use_container_width=True, width='stretch')

            from video_processor import process_video
            results = process_video(
                upload_id,
                save_path,
                progress_callback=progress_callback,
                frame_callback=frame_callback
            )

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            # Show results
            st.markdown("---")
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", results["total_frames"])
            with col2:
                st.metric("Frames Analyzed", results["processed_frames"])
            with col3:
                st.metric("Incidents Found", results["incidents_found"])

            if results["detections"]:
                st.markdown("### Detected Incidents")
                for det in results["detections"]:
                    dtype = det.get("incident_type", "person_dump")
                    type_label = "Car litter" if dtype == "car_litter" else "Person dump"
                    with st.expander(
                        f"Frame {det['frame_number']} — {type_label} — "
                        f"Time: {det['timestamp']:.1f}s — "
                        f"Confidence: {det['confidence']:.1%}"
                    ):
                        if det["snapshot_path"] and os.path.exists(det["snapshot_path"]):
                            img = Image.open(det["snapshot_path"])
                            st.image(img, caption=f"Frame {det['frame_number']}", use_container_width=True, width='stretch')
                        st.markdown(f"**Objects:** {', '.join(det['objects'])}")
                        st.markdown(f"**Confidence:** {det['confidence']:.1%}")
            else:
                st.info("No dumping incidents detected in this video.")

    # Previously uploaded videos
    st.markdown("---")
    st.subheader("Previous Uploads")

    uploads = get_video_uploads()
    if uploads:
        for upload in uploads:
            status_icon = {
                "queued": "⏳",
                "processing": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(upload["status"], "❓")

            with st.expander(
                f"{status_icon} {upload['original_name']} — "
                f"Status: {upload['status']} — "
                f"Incidents: {upload['incidents_found']}"
            ):
                st.markdown(f"**Upload ID:** {upload['id']}")
                st.markdown(f"**Status:** {upload['status']}")
                st.markdown(f"**Total Frames:** {upload['total_frames']}")
                st.markdown(f"**Processed Frames:** {upload['processed_frames']}")
                st.markdown(f"**Incidents Found:** {upload['incidents_found']}")
                st.markdown(f"**Uploaded:** {upload['created_at']}")

                # Show detections for completed uploads
                if upload["status"] == "completed":
                    detections = get_video_detections(upload["id"])
                    if detections:
                        st.markdown("**Detections:**")
                        for det in detections:
                            dtype = det.get("incident_type") or "person_dump"
                            type_label = "Car litter" if dtype == "car_litter" else "Person dump"
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                if det["snapshot_path"] and os.path.exists(det["snapshot_path"]):
                                    img = Image.open(det["snapshot_path"])
                                    st.image(img, use_container_width=True, width='stretch')
                            with col_info:
                                st.markdown(f"**Type:** {type_label}")
                                st.markdown(f"Frame: {det['frame_number']}")
                                st.markdown(f"Time: {det['timestamp_in_video']:.1f}s")
                                st.markdown(f"Confidence: {det['confidence']:.1%}")
    else:
        st.info("No videos uploaded yet.")


# ============================================================
# Camera Management Page
# ============================================================
def page_camera_management():
    st.title("📷 Camera Management")
    st.markdown("Add, test, and manage camera sources.")

    # Add new camera form
    st.subheader("Add New Camera")

    with st.form("add_camera_form"):
        cam_name = st.text_input("Camera Name", placeholder="e.g., Front Gate Camera")

        cam_type = st.selectbox("Camera Type", ["webcam", "ip_stream", "rtsp"])

        if cam_type == "webcam":
            device_idx = st.number_input("Device Index", min_value=0, max_value=10, value=0,
                                         help="0 = default webcam, 1 = second camera, etc.")
            source_url = None
        else:
            device_idx = 0
            placeholder = (
                "http://192.168.1.5:8080/video" if cam_type == "ip_stream"
                else "rtsp://username:password@ip:port/stream"
            )
            source_url = st.text_input("Camera URL", placeholder=placeholder)

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("➕ Add Camera", type="primary",
                                              use_container_width=True, width='stretch')
        with col2:
            test_btn = st.form_submit_button("🔍 Test Connection", use_container_width=True)

        if submitted:
            if not cam_name:
                st.error("Please enter a camera name.")
            elif cam_type != "webcam" and not source_url:
                st.error("Please enter a camera URL.")
            else:
                camera_id = add_camera(cam_name, cam_type, source_url, device_idx)
                st.success(f"✅ Camera **{cam_name}** added with ID {camera_id}!")
                st.rerun()

        if test_btn:
            test_source = device_idx if cam_type == "webcam" else source_url
            if test_source is not None and (cam_type == "webcam" or test_source):
                with st.spinner("Testing camera connection..."):
                    success, message = test_camera_source(test_source)
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("Please enter a camera URL to test.")

    # Existing cameras
    st.markdown("---")
    st.subheader("Registered Cameras")

    cameras = list_cameras()
    if cameras:
        for cam in cameras:
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                source_display = cam.get("source_url") or f"Device {cam.get('device_index', 0)}"
                st.markdown(
                    f"**{cam['name']}** — `{cam['source_type']}`"
                )

            with col2:
                st.markdown(f"ID: {cam['id']} | Added: {cam['created_at']}")

            with col3:
                if st.button("🗑️ Delete", key=f"del_cam_{cam['id']}"):
                    remove_camera(cam["id"])
                    st.success(f"Deleted camera: {cam['name']}")
                    st.rerun()
    else:
        st.info("No cameras registered yet. Add one above.")

    # Mobile camera instructions
    st.markdown("---")
    st.subheader("📱 Mobile Camera Setup")
    st.markdown("""
    To use your phone as a camera source:

    1. Install **IP Webcam** app on your Android phone (from Play Store)
    2. Open the app and tap **Start Server**
    3. Note the URL shown (e.g., `http://192.168.1.5:8080`)
    4. Add a new camera above with type **ip_stream** and URL: `http://YOUR_IP:8080/video`
    5. Make sure your phone and laptop are on the **same WiFi network**
    """)


# ============================================================
# Incident Viewer Page
# ============================================================
def page_incident_viewer():
    st.title("🔍 Incident Viewer")
    st.markdown("Browse and filter all detected incidents.")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        cameras = get_cameras()
        camera_filter = st.selectbox(
            "Filter by Camera",
            ["All Cameras"] + [f"{c['name']} (ID: {c['id']})" for c in cameras]
        )

    with col2:
        date_from = st.date_input("From Date", value=datetime.now().date() - timedelta(days=30))

    with col3:
        date_to = st.date_input("To Date", value=datetime.now().date())

    # Parse camera filter
    camera_id_filter = None
    if camera_filter != "All Cameras":
        try:
            camera_id_filter = int(camera_filter.split("ID: ")[1].rstrip(")"))
        except (IndexError, ValueError):
            pass

    # Fetch incidents
    incidents = get_incidents(
        limit=100,
        camera_id=camera_id_filter,
        date_from=str(date_from),
        date_to=str(date_to + timedelta(days=1))
    )

    st.markdown("---")
    st.markdown(f"**Showing {len(incidents)} incidents**")

    if incidents:
        for incident in incidents:
            inc = _safe_incident_display(incident)
            confidence_str = f"{inc['confidence']:.1%}" if inc.get("confidence") is not None else "N/A"
            itype = inc.get("incident_type") or "person_dump"
            type_label = "Car litter" if itype == "car_litter" else "Person dump"
            cam = get_camera(inc["camera_id"]) if inc.get("camera_id") else None
            location_label = f"{cam['name']} (ID: {inc['camera_id']})" if cam else (f"ID: {inc.get('camera_id')}" if inc.get('camera_id') else "Unknown / Video")

            with st.expander(
                f"🚨 #{inc['id']} | {type_label} | {inc['timestamp']} | "
                f"Confidence: {confidence_str} | Location: {location_label}"
            ):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    if inc.get("snapshot_path") and os.path.exists(inc["snapshot_path"]):
                        img = Image.open(inc["snapshot_path"])
                        st.image(img, caption="Evidence", use_container_width=True, width='stretch')
                    else:
                        st.info("No snapshot available")

                with col_info:
                    st.markdown(f"**Incident ID:** {inc['id']}")
                    st.markdown(f"**Type:** {type_label}")
                    st.markdown(f"**Timestamp:** {inc['timestamp']}")
                    st.markdown(f"**Location (camera):** {location_label}")
                    st.markdown(f"**Confidence:** {confidence_str}")
                    st.markdown(f"**Description:** {inc.get('description', '')}")

                    if inc.get("objects_detected"):
                        try:
                            objects = json.loads(inc["objects_detected"])
                            st.markdown(f"**Objects:** {', '.join(objects)}")
                        except (json.JSONDecodeError, TypeError):
                            st.markdown(f"**Objects:** {inc['objects_detected']}")
    else:
        st.info("No incidents found for the selected filters.")


# ============================================================
# Page Router
# ============================================================
if page == "📊 Dashboard":
    page_dashboard()
elif page == "📹 Live Monitor":
    page_live_monitor()
elif page == "📤 Video Upload":
    page_video_upload()
elif page == "📷 Camera Management":
    page_camera_management()
elif page == "🔍 Incident Viewer":
    page_incident_viewer()

import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="SoccerTrack - Video Testing & Tracking Suite",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .stApp {
        background-color: #0b1118;
        color: #e2e8f0;
    }
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">⚽ SoccerTrack Video Intelligence Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Object Player & Ball Tracking with Spatial Identity Lock, Anti-Stacking & Stable Trajectories</div>', unsafe_allow_html=True)

# Sample videos discovery
REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEOS = {
    "Left Camera Footage (Match sample)": str(REPO_ROOT / "data" / "raw" / "left_camera.mp4"),
    "Right Camera Footage (Match sample)": str(REPO_ROOT / "data" / "raw" / "right_camera.mp4"),
    "Small Movie Test Clip": str(REPO_ROOT / "tests" / "assets" / "videos" / "small-movie.mp4"),
    "Processed Pitch Simulation": str(REPO_ROOT / "data" / "processed" / "pitch.mp4"),
}
SAMPLE_VIDEOS = {k: v for k, v in SAMPLE_VIDEOS.items() if os.path.exists(v)}

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    video_source_type = st.radio("Select Video Source", ["Sample Video", "Upload Custom Video"])
    
    video_path = None
    if video_source_type == "Sample Video":
        if SAMPLE_VIDEOS:
            selected_sample = st.selectbox("Choose sample clip", list(SAMPLE_VIDEOS.keys()))
            video_path = SAMPLE_VIDEOS[selected_sample]
        else:
            st.warning("No sample videos found in repo.")
    else:
        uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    st.markdown("---")
    st.subheader("🎯 Model & Resolution")
    model_choice = st.selectbox("YOLO Model Architecture", ["yolov8s.pt (Recommended)", "yolov8m.pt (High Accuracy)", "yolov8n.pt (Fastest)"], index=0)
    selected_model_file = model_choice.split(" ")[0]
    
    imgsz_val = st.selectbox("Detection Resolution (imgsz)", [1280, 1920, 640], index=0, help="Higher resolution detects distant pitch players clearly.")
    conf_threshold = st.slider("Detection Confidence", 0.1, 0.8, 0.20, 0.05)

    st.markdown("---")
    st.subheader("🔒 Spatial Identity Lock & Anti-Stacking")
    enable_spatial_lock = st.checkbox("Spatial Identity Lock", value=True, help="Locks the player ID to their physical spatial trajectory so an ID NEVER changes when they turn or move.")
    player_radius_px = st.slider("Player Spatial Radius (px)", 30, 150, 75, 5, help="Radius around a player's last known location within which any new detection retains the original player ID.")
    lock_memory_frames = st.slider("Identity Memory (Frames)", 15, 120, 60, 5, help="How long to lock a player's ID even if they are temporarily occluded.")
    
    min_box_height = st.slider("Min Box Height (px)", 10, 80, 25, 5, help="Filters out micro-slivers and partial body fragments.")
    enable_gap_fill = st.checkbox("Continuous Trajectory Interpolation", value=True, help="Ensures bounding boxes remain visible throughout the video without vanishing.")
    max_gap_frames = st.slider("Max Gap Fill (Frames)", 5, 45, 20, 5)

    st.markdown("---")
    st.subheader("🎨 Overlays & Processing Limits")
    show_boxes = st.checkbox("Bounding Boxes", value=True)
    show_labels = st.checkbox("Player / Ball IDs", value=True)
    show_trails = st.checkbox("Trajectory Trails", value=True)
    trail_len = st.slider("Trail History Length", 5, 50, 20)
    max_frames = st.slider("Max Frames to Process (0 = Entire Video)", 0, 1000, 150, 25)

def get_player_color(track_id: int):
    """Generates a stable, high-contrast color for each player ID."""
    if track_id == 0:
        return (0, 230, 255)
    hue = int((track_id * 137.5) % 180)
    hsv_color = np.uint8([[[hue, 220, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

def process_spatial_identity_lock(
    raw_records: list,
    spatial_radius: float = 75.0,
    memory_frames: int = 60,
    min_h: float = 25.0,
    fill_gaps: bool = True,
    max_gap: int = 20,
) -> pd.DataFrame:
    """
    1. Removes stacked/double detections per frame (keeps only 1 primary box per player).
    2. Enforces Spatial Identity Lock (locks new detections to existing physical player slots).
    3. Fills momentary detection gaps smoothly.
    """
    if not raw_records:
        return pd.DataFrame()

    df = pd.DataFrame(raw_records)

    # 1. Height & Aspect Ratio Sanity Check
    df = df[(df["y_max"] - df["y_min"]) >= min_h].copy()
    if df.empty:
        return pd.DataFrame()

    # 2. Strict Frame-by-Frame Non-Maximum Proximity Suppression (Kills 2nd stacked box)
    cleaned_rows = []
    for f, grp in df.groupby("frame"):
        grp = grp.sort_values("confidence", ascending=False)
        kept_players = []
        kept_balls = []

        for _, row in grp.iterrows():
            cls_name = row["class"]
            cx, cy = row["center_x"], row["center_y"]
            w = row["x_max"] - row["x_min"]
            h = row["y_max"] - row["y_min"]

            if cls_name == "player":
                too_close = False
                for (kcx, kcy, kw, kh) in kept_players:
                    dist = np.hypot(cx - kcx, cy - kcy)
                    # If center is within half-body distance, it's a duplicate torso/head box
                    if dist < spatial_radius * 0.65 or (abs(cx - kcx) < kw * 0.6 and abs(cy - kcy) < kh * 0.6):
                        too_close = True
                        break
                if not too_close:
                    kept_players.append((cx, cy, w, h))
                    cleaned_rows.append(row)
            else:
                # Keep top 1 ball
                if len(kept_balls) == 0:
                    kept_balls.append((cx, cy))
                    cleaned_rows.append(row)

    df_clean = pd.DataFrame(cleaned_rows)
    if df_clean.empty:
        return df_clean

    # 3. Spatial Identity Lock (Sequential Player Slotting)
    # Active player slots: {player_id: {'last_f': int, 'cx': float, 'cy': float, 'hits': int}}
    player_slots = {}
    next_pid = 1
    locked_records = []

    for f, grp in df_clean.groupby("frame"):
        # Match current frame detections to existing active player slots
        assigned_pids = set()
        frame_assignments = []

        for _, row in grp.iterrows():
            cls_name = row["class"]
            cx, cy = row["center_x"], row["center_y"]

            if cls_name == "ball":
                r_dict = row.to_dict()
                r_dict["track_id"] = 0
                locked_records.append(r_dict)
                continue

            # Find closest matching active player slot
            best_pid = None
            best_dist = float("inf")

            for pid, slot in player_slots.items():
                if pid in assigned_pids:
                    continue
                gap = f - slot["last_f"]
                if 0 < gap <= memory_frames:
                    dist = np.hypot(cx - slot["cx"], cy - slot["cy"])
                    # Speed threshold: max pixels per frame allowed during gap
                    max_allowed_dist = spatial_radius + gap * 3.0
                    if dist <= max_allowed_dist:
                        if dist < best_dist:
                            best_dist = dist
                            best_pid = pid

            if best_pid is not None:
                assigned_pids.add(best_pid)
                player_slots[best_pid] = {"last_f": f, "cx": cx, "cy": cy, "hits": player_slots[best_pid]["hits"] + 1}
                r_dict = row.to_dict()
                r_dict["track_id"] = best_pid
                locked_records.append(r_dict)
            else:
                # New player on field
                new_pid = next_pid
                next_pid += 1
                assigned_pids.add(new_pid)
                player_slots[new_pid] = {"last_f": f, "cx": cx, "cy": cy, "hits": 1}
                r_dict = row.to_dict()
                r_dict["track_id"] = new_pid
                locked_records.append(r_dict)

    df_locked = pd.DataFrame(locked_records)
    if df_locked.empty:
        return df_locked

    # Filter out single-frame phantom players (must appear at least 3 times)
    player_hits = df_locked[df_locked["class"] == "player"].groupby("track_id")["frame"].count()
    valid_players = player_hits[player_hits >= 3].index
    df_locked = df_locked[(df_locked["class"] == "ball") | (df_locked["track_id"].isin(valid_players))].copy()

    # 4. Trajectory Gap Filling
    if not fill_gaps or max_gap <= 0:
        return df_locked.sort_values(by=["frame", "track_id"]).reset_index(drop=True)

    processed_groups = []
    for tid, group in df_locked.groupby("track_id"):
        group = group.sort_values("frame").drop_duplicates(subset=["frame"])
        cls_name = group["class"].iloc[0]
        min_f = int(group["frame"].min())
        max_f = int(group["frame"].max())

        full_idx = pd.Index(range(min_f, max_f + 1), name="frame")
        g_reindexed = group.set_index("frame").reindex(full_idx)
        g_reindexed["track_id"] = tid
        g_reindexed["class"] = cls_name

        for col in ["x_min", "y_min", "x_max", "y_max", "center_x", "center_y"]:
            g_reindexed[col] = g_reindexed[col].interpolate(method="linear", limit=max_gap)

        g_reindexed["confidence"] = g_reindexed["confidence"].ffill().fillna(0.5)
        g_clean = g_reindexed.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).reset_index()
        processed_groups.append(g_clean)

    final_df = pd.concat(processed_groups, ignore_index=True)
    return final_df.sort_values(by=["frame", "track_id"]).reset_index(drop=True)

def convert_to_h264(input_path, output_path):
    """Convert raw OpenCV avi output to browser-compatible H.264 MP4 using ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "22",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def draw_tactical_pitch(trajectories, frame_w, frame_h):
    """Draw a 2D top-down mini-pitch radar with player positions."""
    pitch_img = np.zeros((300, 450, 3), dtype=np.uint8)
    pitch_img[:] = (20, 50, 20)

    cv2.rectangle(pitch_img, (15, 15), (435, 285), (255, 255, 255), 2)
    cv2.line(pitch_img, (225, 15), (225, 285), (255, 255, 255), 2)
    cv2.circle(pitch_img, (225, 150), 40, (255, 255, 255), 2)
    cv2.circle(pitch_img, (225, 150), 3, (255, 255, 255), -1)

    cv2.rectangle(pitch_img, (15, 75), (85, 225), (255, 255, 255), 2)
    cv2.rectangle(pitch_img, (365, 75), (435, 225), (255, 255, 255), 2)

    for tid, data in trajectories.items():
        if not data["points"]:
            continue
        last_pt = data["points"][-1]
        cls_name = data.get("class", "player")

        nx = np.clip(last_pt[0] / max(frame_w, 1), 0, 1)
        ny = np.clip(last_pt[1] / max(frame_h, 1), 0, 1)

        px = int(15 + nx * (435 - 15))
        py = int(15 + ny * (285 - 15))

        if cls_name == "ball" or tid == 0:
            cv2.circle(pitch_img, (px, py), 5, (0, 255, 255), -1)
        else:
            color = get_player_color(int(tid))
            cv2.circle(pitch_img, (px, py), 6, color, -1)
            cv2.putText(pitch_img, str(tid), (px - 5, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return pitch_img

if video_path and os.path.exists(video_path):
    col1, col2 = st.columns([1, 1])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    with col1:
        st.subheader("📹 Video Properties")
        st.markdown(f"""
        - **Source:** `{Path(video_path).name}`
        - **Resolution:** `{width} x {height} px`
        - **Total Frames:** `{total_frames}`
        - **Frame Rate:** `{fps:.1f} FPS`
        - **Duration:** `{total_frames / max(fps, 1):.2f} seconds`
        """)

    with col2:
        st.subheader("🎬 Source Video Preview")
        st.video(video_path)

    st.markdown("---")

    if st.button("🚀 Run Player & Ball Tracking"):
        st.info(f"Extracting high-resolution detections with {selected_model_file} (imgsz={imgsz_val})...")

        try:
            model = YOLO(selected_model_file)
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            st.stop()

        frames_limit = total_frames if max_frames == 0 else min(total_frames, max_frames)

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        stat_frame = stats_col1.empty()
        stat_players = stats_col2.empty()
        stat_fps = stats_col3.empty()
        stat_ball = stats_col4.empty()

        progress_bar = st.progress(0)

        raw_records = []
        start_time = time.time()
        frame_idx = 0

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened() and frame_idx < frames_limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            loop_start = time.time()

            preds = model.predict(
                frame,
                conf=conf_threshold,
                imgsz=imgsz_val,
                classes=[0, 32],
                verbose=False,
            )

            active_p = 0
            has_b = False

            if preds and preds[0].boxes is not None:
                boxes = preds[0].boxes.xyxy.cpu().numpy()
                classes = preds[0].boxes.cls.int().cpu().numpy()
                confs = preds[0].boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes, classes, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = int((x1 + x2) / 2), int(y2)
                    cls_name = "player" if cls_id == 0 else "ball"

                    if cls_id == 0:
                        active_p += 1
                    else:
                        has_b = True

                    raw_records.append({
                        "frame": frame_idx,
                        "class": cls_name,
                        "x_min": float(x1),
                        "y_min": float(y1),
                        "x_max": float(x2),
                        "y_max": float(y2),
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "confidence": float(conf),
                    })

            curr_fps = 1.0 / max(time.time() - loop_start, 1e-4)
            progress_bar.progress(frame_idx / frames_limit)

            stat_frame.markdown(f'<div class="stat-card"><div class="stat-val">{frame_idx}/{frames_limit}</div><div class="stat-label">Frames Extracted</div></div>', unsafe_allow_html=True)
            stat_players.markdown(f'<div class="stat-card"><div class="stat-val">{active_p}</div><div class="stat-label">Raw Players</div></div>', unsafe_allow_html=True)
            stat_fps.markdown(f'<div class="stat-card"><div class="stat-val">{curr_fps:.1f}</div><div class="stat-label">Extraction FPS</div></div>', unsafe_allow_html=True)
            stat_ball.markdown(f'<div class="stat-card"><div class="stat-val">{"🟢 Tracked" if has_b else "⚪ Searching"}</div><div class="stat-label">Ball Status</div></div>', unsafe_allow_html=True)

        cap.release()

        # Step 2: Spatial Identity Lock & Anti-Stacking
        st.info("Applying Spatial Identity Lock and Anti-Stacking Trajectory Engine...")
        df_locked = process_spatial_identity_lock(
            raw_records,
            spatial_radius=player_radius_px,
            memory_frames=lock_memory_frames,
            min_h=min_box_height,
            fill_gaps=enable_gap_fill,
            max_gap=max_gap_frames,
        )

        # Step 3: Video Render
        temp_raw_out = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_raw_path = temp_raw_out.name
        temp_raw_out.close()

        out_writer = cv2.VideoWriter(temp_raw_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        cap = cv2.VideoCapture(video_path)

        render_progress = st.progress(0)
        view_col1, view_col2 = st.columns([1.4, 1])
        video_placeholder = view_col1.empty()
        radar_placeholder = view_col2.empty()

        frame_idx = 0
        trajectories = {}
        frame_grouped = {f: g for f, g in df_locked.groupby("frame")} if not df_locked.empty else {}

        while cap.isOpened() and frame_idx < frames_limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            annotated_frame = frame.copy()

            if frame_idx in frame_grouped:
                frame_data = frame_grouped[frame_idx]
                for _, row in frame_data.iterrows():
                    tid = int(row["track_id"])
                    cls_name = row["class"]
                    conf = float(row["confidence"])
                    x1, y1 = int(row["x_min"]), int(row["y_min"])
                    x2, y2 = int(row["x_max"]), int(row["y_max"])
                    cx, cy = int(row["center_x"]), int(row["center_y"])

                    if cls_name == "player":
                        box_color = get_player_color(tid)
                        label = f"Player #{tid}"
                    else:
                        box_color = (0, 230, 255)
                        label = f"Ball"

                    if tid not in trajectories:
                        trajectories[tid] = {"points": [], "class": cls_name}
                    trajectories[tid]["points"].append((cx, cy))
                    if len(trajectories[tid]["points"]) > trail_len:
                        trajectories[tid]["points"].pop(0)

                    if show_boxes:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    if show_labels:
                        label_txt = f"{label} ({conf:.2f})"
                        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(
                            annotated_frame,
                            (x1, max(0, y1 - th - 8)),
                            (x1 + tw + 6, max(th + 8, y1)),
                            box_color,
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label_txt,
                            (x1 + 3, max(th + 2, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 0),
                            1,
                        )
                    if show_trails and len(trajectories[tid]["points"]) > 1:
                        pts = np.array(trajectories[tid]["points"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [pts], False, box_color, 2)

            out_writer.write(annotated_frame)
            pitch_radar = draw_tactical_pitch(trajectories, width, height)

            render_progress.progress(frame_idx / frames_limit)
            if frame_idx % 2 == 0 or frame_idx == frames_limit:
                disp_frame = cv2.resize(annotated_frame, (720, int(720 * height / width)))
                video_placeholder.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB), caption=f"Locked Identity Stream (Frame {frame_idx})", width="stretch")
                radar_placeholder.image(cv2.cvtColor(pitch_radar, cv2.COLOR_BGR2RGB), caption="Live 2D Tactical Radar", width="stretch")

        cap.release()
        out_writer.release()

        # Convert to browser-compatible H.264 MP4
        temp_h264_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_h264_path = temp_h264_out.name
        temp_h264_out.close()

        with st.spinner("⚡ Finalizing high-definition H.264 video..."):
            convert_to_h264(temp_raw_path, temp_h264_path)

        elapsed = time.time() - start_time

        st.session_state["processed_video_path"] = temp_h264_path
        st.session_state["tracking_df"] = df_locked
        st.session_state["elapsed"] = elapsed
        st.session_state["processed_frames"] = frame_idx
        st.session_state["video_w"] = width
        st.session_state["video_h"] = height

        if os.path.exists(temp_raw_path):
            os.remove(temp_raw_path)

    # Render results
    if "processed_video_path" in st.session_state and os.path.exists(st.session_state["processed_video_path"]):
        elapsed = st.session_state.get("elapsed", 0)
        frames_done = st.session_state.get("processed_frames", 0)
        avg_fps = frames_done / max(elapsed, 1e-4)

        st.success(f"🎉 Spatial Identity Lock completed in {elapsed:.2f}s ({avg_fps:.1f} avg FPS)!")

        st.markdown("---")
        st.subheader("🎥 Final Processed Video Playback (Locked Identity & Non-Stacking)")

        with open(st.session_state["processed_video_path"], "rb") as vfile:
            video_bytes = vfile.read()
            st.video(video_bytes)

        st.download_button(
            label="📥 Download Tracked Video (.mp4)",
            data=video_bytes,
            file_name="soccertrack_locked_video.mp4",
            mime="video/mp4",
        )

        st.markdown("---")
        st.subheader("📊 Analytics & Export")

        df_tracks = st.session_state.get("tracking_df", pd.DataFrame())
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown("### 📥 Tracking Dataset (SoccerTrack Format)")
            if not df_tracks.empty:
                st.dataframe(df_tracks.head(15), width="stretch")
                csv_data = df_tracks.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Tracking CSV",
                    data=csv_data,
                    file_name="soccertrack_tracking_results.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No tracking detections recorded.")

        with res_col2:
            st.markdown("### 📈 Tactical Trajectory Heatmap")
            if not df_tracks.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor("#0b1118")
                ax.set_facecolor("#111827")

                vid_w = st.session_state.get("video_w", 1920)
                vid_h = st.session_state.get("video_h", 1080)

                ax.plot([0, vid_w, vid_w, 0, 0], [0, 0, vid_h, vid_h, 0], color="white", lw=1.5)
                ax.axvline(vid_w / 2, color="white", lw=1.5)

                player_df = df_tracks[df_tracks["class"] == "player"]
                for tid, group in player_df.groupby("track_id"):
                    color_rgb = [c / 255.0 for c in get_player_color(int(tid))]
                    color_rgb = [color_rgb[2], color_rgb[1], color_rgb[0]]
                    ax.plot(group["center_x"], group["center_y"], label=f"P#{tid}", color=color_rgb, alpha=0.8, lw=2)

                ax.set_title("Persistent Player Movement Trajectories", color="white", fontsize=12)
                ax.set_xlim(0, vid_w)
                ax.set_ylim(vid_h, 0)
                ax.tick_params(colors="gray")
                for spine in ax.spines.values():
                    spine.set_color("#374151")

                st.pyplot(fig)
            else:
                st.info("No trajectory data to plot.")

else:
    st.info("Please select or upload a video from the sidebar to begin testing.")

"""Batch Podcast Assembly App.

Stitches Intro + Main Episode + Outro for 170+ episodes with precise
timing, loudness normalization, and optional crossfades.
"""

import io
import os
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import static_ffmpeg
static_ffmpeg.add_paths()
_ffmpeg_path, _ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
AudioSegment.converter = _ffmpeg_path
AudioSegment.ffmpeg = _ffmpeg_path
AudioSegment.ffprobe = _ffprobe_path

# ---------------------------------------------------------------------------
# Constants (timing rules)
# ---------------------------------------------------------------------------
MAIN_START_OFFSET_MS = 62_500  # 1:02.5 - where main speech must land
OUTRO_OVERLAP_MS = 27_400      # speech must end 27.4s into outro
TARGET_FRAME_RATE = 44100
TARGET_CHANNELS = 2
OUTPUT_DIR = "output"

st.set_page_config(page_title="Podcast Batch Assembler", layout="wide")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_audio(file_bytes: bytes, name: str) -> AudioSegment:
    ext = os.path.splitext(name)[1].lstrip(".").lower() or "mp3"
    seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=ext)
    return seg.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_FRAME_RATE)


def normalize_loudness(seg: AudioSegment, target_dbfs: float) -> AudioSegment:
    if seg.dBFS == float("-inf"):
        return seg
    gain = target_dbfs - seg.dBFS
    # clamp extreme gains to avoid blowing up near-silent clips
    gain = max(min(gain, 30.0), -30.0)
    return seg.apply_gain(gain)


def detect_speech_bounds(seg: AudioSegment, min_silence_len: int, silence_thresh_offset: int):
    thresh = seg.dBFS + silence_thresh_offset if seg.dBFS != float("-inf") else -50
    nonsilent = detect_nonsilent(seg, min_silence_len=min_silence_len, silence_thresh=thresh)
    if not nonsilent:
        return 0.0, len(seg) / 1000.0
    start_ms = nonsilent[0][0]
    end_ms = nonsilent[-1][1]
    return start_ms / 1000.0, end_ms / 1000.0


def build_episode(intro: AudioSegment, outro: AudioSegment, main: AudioSegment,
                   speech_start: float, speech_end: float,
                   crossfade_ms: int, target_lufs: float):
    intro_n = normalize_loudness(intro, target_lufs)
    outro_n = normalize_loudness(outro, target_lufs)

    start_ms = max(0, int(speech_start * 1000))
    end_ms = max(start_ms + 1, int(speech_end * 1000))
    trimmed_main = main[start_ms:end_ms]
    main_n = normalize_loudness(trimmed_main, target_lufs)

    main_pos = MAIN_START_OFFSET_MS
    main_dur = len(main_n)
    outro_pos = max(0, main_pos + main_dur - OUTRO_OVERLAP_MS)
    total_dur = max(main_pos + main_dur, outro_pos + len(outro_n))

    if crossfade_ms > 0:
        cf = min(crossfade_ms, len(intro_n), len(main_n) // 2, len(outro_n))
        if cf > 0:
            intro_n = intro_n.fade_out(cf)
            main_n = main_n.fade_in(cf).fade_out(cf)
            outro_n = outro_n.fade_in(cf)

    canvas = AudioSegment.silent(duration=total_dur, frame_rate=TARGET_FRAME_RATE)
    canvas = canvas.set_channels(TARGET_CHANNELS)

    canvas = canvas.overlay(intro_n, position=0)
    canvas = canvas.overlay(main_n, position=main_pos)
    canvas = canvas.overlay(outro_n, position=outro_pos)

    return canvas, total_dur / 1000.0


def waveform_dataframe(seg: AudioSegment, max_points: int = 1500) -> pd.DataFrame:
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples = np.abs(samples) / float(1 << (8 * seg.sample_width - 1))

    if len(samples) > max_points:
        chunk = len(samples) // max_points
        samples = samples[: chunk * max_points].reshape(-1, chunk).max(axis=1)

    times = np.linspace(0, len(seg) / 1000.0, num=len(samples))
    return pd.DataFrame({"amplitude": samples}, index=times)


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "episodes" not in st.session_state:
    st.session_state.episodes = {}  # name -> dict
if "export_zip" not in st.session_state:
    st.session_state.export_zip = None
if "intro_bytes" not in st.session_state:
    default_intro = os.path.join("assets", "intro.mp3")
    if os.path.exists(default_intro):
        with open(default_intro, "rb") as f:
            st.session_state.intro_bytes = f.read()
        st.session_state.intro_name = "intro.mp3"
    else:
        st.session_state.intro_bytes = None
        st.session_state.intro_name = None
if "outro_bytes" not in st.session_state:
    default_outro = os.path.join("assets", "outro.mp3")
    if os.path.exists(default_outro):
        with open(default_outro, "rb") as f:
            st.session_state.outro_bytes = f.read()
        st.session_state.outro_name = "outro.mp3"
    else:
        st.session_state.outro_bytes = None
        st.session_state.outro_name = None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header band */
    .pa-header {
        padding: 1.75rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 18px 18px;
        background: linear-gradient(135deg, #0F1B33 0%, #14233F 60%, #0F2F2A 100%);
        border-bottom: 1px solid rgba(16, 185, 129, 0.25);
    }
    .pa-header h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .pa-header p {
        margin: 0.35rem 0 0 0;
        color: #94A3B8;
        font-size: 0.95rem;
    }
    .pa-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        color: #34D399 !important;
        background-color: rgba(16, 185, 129, 0.08);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: 0.01em;
        transition: all 150ms ease-out;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .stButton > button:hover {
        border-color: #34D399;
        color: #34D399;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10B981, #059669);
        border: none;
        color: #06120E;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        color: #06120E;
    }

    /* Cards / containers */
    .pa-card {
        background: #141C2E;
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .pa-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        font-weight: 700;
    }

    /* Info / divider polish */
    .stAlert {
        border-radius: 12px;
    }
    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }

    /* Metric-style badges */
    .pa-stat {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }
    .pa-stat .value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .pa-stat .label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    </style>

    <div class="pa-header">
        <span class="pa-badge">Batch Audio Pipeline</span>
        <h1>Podcast Batch Assembler</h1>
        <p>Stitch intro, episode, and outro audio with frame-accurate timing,
        loudness normalization, and smooth crossfades — across 170+ episodes at once.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

setup_tab, batch_tab = st.tabs(["1. Setup (Intro / Outro)", "2. Batch Process Episodes"])

with setup_tab:
    st.subheader("Intro and Outro (used for every episode)")
    col1, col2 = st.columns(2)
    with col1:
        intro_file = st.file_uploader("Intro file", type=["mp3", "wav", "m4a", "aac"], key="intro_uploader")
        if intro_file is not None:
            st.session_state.intro_bytes = intro_file.read()
            st.session_state.intro_name = intro_file.name
        if st.session_state.intro_bytes:
            st.audio(st.session_state.intro_bytes)
            st.caption(f"Loaded: {st.session_state.intro_name}")
    with col2:
        outro_file = st.file_uploader("Outro file", type=["mp3", "wav", "m4a", "aac"], key="outro_uploader")
        if outro_file is not None:
            st.session_state.outro_bytes = outro_file.read()
            st.session_state.outro_name = outro_file.name
        if st.session_state.outro_bytes:
            st.audio(st.session_state.outro_bytes)
            st.caption(f"Loaded: {st.session_state.outro_name}")

    st.subheader("Processing settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_lufs = st.number_input("Target loudness (dBFS)", value=-16.0, step=0.5)
    with c2:
        crossfade_ms = st.number_input("Crossfade duration (ms)", value=100, min_value=0, max_value=2000, step=50)
    with c3:
        export_format = st.selectbox("Export format", ["mp3", "wav"], index=0)

    st.subheader("Speech detection settings")
    d1, d2 = st.columns(2)
    with d1:
        min_silence_len = st.number_input("Min silence length (ms)", value=500, min_value=50, step=50)
    with d2:
        silence_thresh_offset = st.number_input("Silence threshold offset (dB below avg)", value=-16, step=1)

    st.info(
        f"Timing rules: intro starts at 0:00 • main speech lands at "
        f"{MAIN_START_OFFSET_MS/1000:.1f}s • outro starts {OUTRO_OVERLAP_MS/1000:.1f}s "
        f"before detected speech end."
    )


with batch_tab:
    st.subheader("Upload main episode files")
    main_files = st.file_uploader(
        "Main episode files (select all 170+ at once)",
        type=["mp3", "wav", "m4a", "aac"],
        accept_multiple_files=True,
        key="main_uploader",
    )

    if main_files:
        for f in main_files:
            if f.name not in st.session_state.episodes:
                file_bytes = f.read()
                seg = load_audio(file_bytes, f.name)
                s, e = detect_speech_bounds(seg, min_silence_len, silence_thresh_offset)
                st.session_state.episodes[f.name] = {
                    "bytes": file_bytes,
                    "duration": len(seg) / 1000.0,
                    "speech_start": round(s, 2),
                    "speech_end": round(e, 2),
                    "status": "Pending",
                    "final_duration": None,
                }

    if not st.session_state.episodes:
        st.info("Upload main episode files to begin.")
    else:
        names = list(st.session_state.episodes.keys())
        done_count = sum(1 for n in names if st.session_state.episodes[n]["status"] == "Done")
        error_count = sum(1 for n in names if str(st.session_state.episodes[n]["status"]).startswith("Error"))
        pending_count = len(names) - done_count - error_count

        s1, s2, s3, s4 = st.columns(4)
        for col, value, label in (
            (s1, len(names), "Episodes Loaded"),
            (s2, pending_count, "Pending"),
            (s3, done_count, "Exported"),
            (s4, error_count, "Errors"),
        ):
            with col:
                st.markdown(
                    f"""<div class="pa-card"><div class="pa-stat">
                        <span class="value">{value}</span>
                        <span class="label">{label}</span>
                    </div></div>""",
                    unsafe_allow_html=True,
                )

        st.subheader("Processing table")
        table_rows = []
        for name in names:
            ep = st.session_state.episodes[name]
            est_final = (
                MAIN_START_OFFSET_MS / 1000
                + (ep["speech_end"] - ep["speech_start"])
                + (len(load_audio(st.session_state.outro_bytes, st.session_state.outro_name)) / 1000.0
                   - OUTRO_OVERLAP_MS / 1000)
                if st.session_state.outro_bytes else None
            )
            table_rows.append({
                "Episode": name,
                "Speech Start (s)": ep["speech_start"],
                "Speech End (s)": ep["speech_end"],
                "Source Duration (s)": round(ep["duration"], 2),
                "Est. Final Duration (s)": round(est_final, 2) if est_final else None,
                "Status": ep["final_duration"] and "Done" or ep["status"],
            })

        edited = st.data_editor(
            table_rows,
            column_config={
                "Episode": st.column_config.TextColumn(disabled=True),
                "Source Duration (s)": st.column_config.NumberColumn(disabled=True),
                "Est. Final Duration (s)": st.column_config.NumberColumn(disabled=True),
                "Status": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True,
            key="episode_table",
            width="stretch",
        )

        # write back manual edits to session state
        for row in edited:
            ep = st.session_state.episodes[row["Episode"]]
            ep["speech_start"] = row["Speech Start (s)"]
            ep["speech_end"] = row["Speech End (s)"]

        st.subheader("Preview / manual adjustment")
        selected = st.selectbox("Select episode to preview", names)
        if selected:
            ep = st.session_state.episodes[selected]
            seg = load_audio(ep["bytes"], selected)

            col_a, col_b = st.columns(2)
            with col_a:
                new_start = st.number_input(
                    "Speech start (s)", value=float(ep["speech_start"]),
                    min_value=0.0, max_value=ep["duration"], step=0.05, key=f"start_{selected}"
                )
            with col_b:
                new_end = st.number_input(
                    "Speech end (s)", value=float(ep["speech_end"]),
                    min_value=0.0, max_value=ep["duration"], step=0.05, key=f"end_{selected}"
                )
            ep["speech_start"] = new_start
            ep["speech_end"] = new_end

            st.line_chart(waveform_dataframe(seg), height=180)
            st.caption(f"Speech start: {new_start:.2f}s · Speech end: {new_end:.2f}s")

            if not st.session_state.intro_bytes or not st.session_state.outro_bytes:
                st.error("Please upload an Intro and Outro file in the Setup tab first.")
                st.caption("Raw main episode file:")
                st.audio(ep["bytes"])
            else:
                intro_seg = load_audio(st.session_state.intro_bytes, st.session_state.intro_name)
                outro_seg = load_audio(st.session_state.outro_bytes, st.session_state.outro_name)
                preview_seg, preview_dur = build_episode(
                    intro_seg, outro_seg, seg,
                    new_start, new_end,
                    crossfade_ms, target_lufs,
                )
                buf = io.BytesIO()
                preview_seg.export(buf, format="mp3", bitrate="192k")
                st.caption(f"Full assembled preview ({preview_dur:.1f}s):")
                st.audio(buf.getvalue(), format="audio/mp3")

        st.divider()
        st.subheader("Export")

        if st.button("Process & Export All Episodes", type="primary"):
            if not st.session_state.intro_bytes or not st.session_state.outro_bytes:
                st.error("Please upload an Intro and Outro file in the Setup tab first.")
            else:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                intro_seg = load_audio(st.session_state.intro_bytes, st.session_state.intro_name)
                outro_seg = load_audio(st.session_state.outro_bytes, st.session_state.outro_name)

                progress = st.progress(0)
                status_area = st.empty()
                total = len(names)
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, name in enumerate(names):
                        ep = st.session_state.episodes[name]
                        status_area.text(f"Processing {name} ({i+1}/{total})...")
                        try:
                            main_seg = load_audio(ep["bytes"], name)
                            result, final_dur = build_episode(
                                intro_seg, outro_seg, main_seg,
                                ep["speech_start"], ep["speech_end"],
                                crossfade_ms, target_lufs,
                            )
                            base = os.path.splitext(name)[0]
                            out_name = f"{base}.{export_format}"
                            out_path = os.path.join(OUTPUT_DIR, out_name)
                            export_kwargs = {"format": "mp3", "bitrate": "192k"} if export_format == "mp3" else {"format": "wav"}
                            result.export(out_path, **export_kwargs)

                            file_buf = io.BytesIO()
                            result.export(file_buf, **export_kwargs)
                            zf.writestr(out_name, file_buf.getvalue())

                            ep["status"] = "Done"
                            ep["final_duration"] = round(final_dur, 2)
                        except Exception as exc:
                            ep["status"] = f"Error: {exc}"
                        progress.progress((i + 1) / total)

                status_area.text("Batch processing complete.")
                st.session_state.export_zip = zip_buffer.getvalue()
                st.session_state.export_zip_format = export_format
                st.success(f"Exported {total} episodes.")
                st.rerun()

        if st.session_state.get("export_zip"):
            st.download_button(
                "Download all episodes (.zip)",
                data=st.session_state.export_zip,
                file_name=f"episodes_{st.session_state.export_zip_format}.zip",
                mime="application/zip",
            )

"""Batch Podcast Assembly App.

Stitches Intro + Main Episode + Outro for 170+ episodes with precise
timing, loudness normalization, and optional crossfades.
"""

import io
import os

import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import streamlit as st
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

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


def measure_lufs(seg: AudioSegment) -> float:
    data = np.array(seg.get_array_of_samples()).astype(np.float64)
    data = data.reshape((-1, seg.channels))
    data /= float(1 << (8 * seg.sample_width - 1))
    meter = pyln.Meter(seg.frame_rate)
    return meter.integrated_loudness(data)


def normalize_loudness(seg: AudioSegment, target_lufs: float) -> AudioSegment:
    loudness = measure_lufs(seg)
    if loudness == float("-inf"):
        return seg
    gain = target_lufs - loudness
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


def plot_waveform(seg: AudioSegment, speech_start: float, speech_end: float):
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples /= float(1 << (8 * seg.sample_width - 1))
    times = np.linspace(0, len(seg) / 1000.0, num=len(samples))

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(times, samples, linewidth=0.5, color="steelblue")
    ax.axvline(speech_start, color="green", linestyle="--", label="Speech start")
    ax.axvline(speech_end, color="red", linestyle="--", label="Speech end")
    ax.set_xlim(0, len(seg) / 1000.0)
    ax.set_xlabel("Seconds")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "episodes" not in st.session_state:
    st.session_state.episodes = {}  # name -> dict
if "intro_bytes" not in st.session_state:
    default_intro = os.path.join("assets", "intro.wav")
    if os.path.exists(default_intro):
        with open(default_intro, "rb") as f:
            st.session_state.intro_bytes = f.read()
        st.session_state.intro_name = "intro.wav"
    else:
        st.session_state.intro_bytes = None
        st.session_state.intro_name = None
if "outro_bytes" not in st.session_state:
    default_outro = os.path.join("assets", "outro.wav")
    if os.path.exists(default_outro):
        with open(default_outro, "rb") as f:
            st.session_state.outro_bytes = f.read()
        st.session_state.outro_name = "outro.wav"
    else:
        st.session_state.outro_bytes = None
        st.session_state.outro_name = None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Batch Podcast Assembler")

setup_tab, batch_tab = st.tabs(["1. Setup (Intro/Outro)", "2. Batch Process Episodes"])

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
        target_lufs = st.number_input("Target loudness (LUFS)", value=-16.0, step=0.5)
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
        st.subheader("Processing table")
        names = list(st.session_state.episodes.keys())
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
            use_container_width=True,
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

            st.pyplot(plot_waveform(seg, new_start, new_end))
            st.audio(ep["bytes"])

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
                        out_path = os.path.join(OUTPUT_DIR, f"{base}.{export_format}")
                        if export_format == "mp3":
                            result.export(out_path, format="mp3", bitrate="192k")
                        else:
                            result.export(out_path, format="wav")
                        ep["status"] = "Done"
                        ep["final_duration"] = round(final_dur, 2)
                    except Exception as exc:
                        ep["status"] = f"Error: {exc}"
                    progress.progress((i + 1) / total)

                status_area.text("Batch processing complete.")
                st.success(f"Exported {total} episodes to ./{OUTPUT_DIR}/")
                st.rerun()

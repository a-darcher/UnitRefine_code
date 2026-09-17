from pathlib import Path
import logging
import struct
import numpy as np
import pandas as pd
from datetime import datetime

import spikeinterface.extractors as se
from spikeinterface.core import NumpySorting, create_sorting_analyzer


# ============================================================
# USER-CONTROLLED PATHS
# ============================================================

root_dir = Path("/media/Neuralynx")

csv_path = Path(
    "/media/Projects/alana/UnitRefine/sessionCherryCounts.csv"
)

# EVERYTHING PRODUCED BY THIS SCRIPT GOES HERE
output_root = Path(
    "/media/Projects/alana/UnitRefine/analyzers"
)

output_root.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================
time_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_file = output_root / f"unitrefine_{time_string}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("UnitRefine")

logger.info("=" * 70)
logger.info("Starting UnitRefine analysis")
logger.info(f"Input CSV: {csv_path}")
logger.info(f"Neuralynx root: {root_dir}")
logger.info(f"Output root: {output_root}")
logger.info("=" * 70)


# ============================================================
# METRICS
# ============================================================

metric_names = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "synchrony",
    "firing_range",
    "sd_ratio",
    "amplitude_cutoff",
    "amplitude_median",
    "amplitude_cv",
    "drift",
]

template_metric_names = [
    "peak_to_trough_duration",
    "waveform_ratios",
    "half_width",
    "repolarization_slope",
    "recovery_slope",
    "waveform_baseline_flatness",
    "number_of_peaks",
]


# ============================================================
# READ SESSION LIST
# ============================================================

csv = pd.read_csv(csv_path)

logger.info(f"Found {len(csv)} sessions in CSV")

# ============================================================
# PROCESS SESSIONS
# ============================================================

for session_idx, session_name in enumerate(csv["sessionname"], start=1):

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        f"SESSION {session_idx}/{len(csv)}: {session_name}"
    )
    logger.info("=" * 70)

    try:

        session_path = root_dir / session_name

        logger.info(f"Session path: {session_path}")

        if not session_path.exists():
            logger.error(
                f"Session directory does not exist: {session_path}"
            )
            continue

        if len(list(session_path.glob('do_sort*'))) == 0:
            logger.info(f"Sorter: NOT combinato.")

            sorter = "wave_clus"

        else:
            logger.info(f"Sorter: combinato")
            sorter = "combinato"

        # ----------------------------------------------------
        # CHANNEL NAMES
        # ----------------------------------------------------

        channel_names_file = session_path / "ChannelNames.txt"

        logger.info(
            f"Reading channel names: {channel_names_file}"
        )

        df_channel_names = pd.read_csv(
            channel_names_file,
            header=None
        )

        df_channel_names["ch_name"] = (
            df_channel_names[0]
            .str.removesuffix(".ncs")
        )

        logger.info(
            f"Found {len(df_channel_names)} channels"
        )


        # ----------------------------------------------------
        # READ NEURALYNX RECORDING
        # ----------------------------------------------------

        logger.info("Reading Neuralynx recording...")

        recording = se.read_neuralynx(session_path)

        logger.info(
            f"Recording loaded: {len(recording.channel_ids)} channels"
        )


        # ----------------------------------------------------
        # DETERMINE OFFSET
        # ----------------------------------------------------

        ncs_path = session_path / "CSC1.ncs"

        logger.info(
            f"Reading Neuralynx timestamp from: {ncs_path}"
        )

        with open(ncs_path, "rb") as f:
            f.seek(16384)
            first_record = f.read(1044)

        timestamp_us = struct.unpack(
            "<Q",
            first_record[:8]
        )[0]

        sampling_frequency = 32768

        offset_samples = round(
            timestamp_us
            * sampling_frequency
            / 1e6
        )

        logger.info(
            f"First timestamp: {timestamp_us} us"
        )

        logger.info(
            f"Offset: {offset_samples} samples"
        )


        # ----------------------------------------------------
        # CHANNEL LOOKUP
        # ----------------------------------------------------

        channel_ids = (
            np.arange(df_channel_names.shape[0]) + 1
        )

        ch_lookup = pd.DataFrame({
            "obj_name": recording.channel_ids,
            "ch_location": recording._properties["channel_name"]
        })

        ch_lookup["csc_nr"] = [
            np.int32(
                df_channel_names[
                    df_channel_names["ch_name"]
                    == row.ch_location
                ].index
            )[0] + 1
            for _, row in ch_lookup.iterrows()
        ]

        logger.info("Channel lookup created")


        # ====================================================
        # PROCESS EACH CSC
        # ====================================================

        for ch_nr in channel_ids:

            logger.info("")
            logger.info("-" * 70)
            logger.info(
                f"Processing {session_name} - CSC{ch_nr}"
            )
            logger.info("-" * 70)

            try:    
                

                if sorter == "combinato":
                    
                    # ------------------------------------------------
                    # COMBINATO
                    # ------------------------------------------------
                    
                    path_file = session_path / f"CSC{ch_nr}"

                    logger.info(
                        f"Reading Combinato sorting: {path_file}"
                    )

                    sorting = se.read_combinato(
                        folder_path=path_file,
                        sampling_frequency=32768,
                        user="tho",
                        det_sign="pos",
                        keep_good_only=False,
                    )

                    logger.info(
                        f"Loaded {len(sorting.unit_ids)} units"
                    )

                elif sorter == "wave_clus":
                    # ------------------------------------------------
                    # WAVE CLUS
                    # ------------------------------------------------

                    path_file = session_path / f"times_CSC{ch_nr}.mat"

                    logger.info(
                        f"Reading wave_clus sorting: {path_file}"
                    )

                    sorting = se.read_waveclus(
                        file_path=path_file,
                        keep_good_only=False,
                    )

                    logger.info(
                        f"Loaded {len(sorting.unit_ids)} units"
                    )

                # ------------------------------------------------
                # SHIFT SPIKE TIMES
                # ------------------------------------------------

                logger.info(
                    "Shifting spike trains..."
                )

                shifted_spike_trains = {
                    unit_id:
                        sorting.get_unit_spike_train(unit_id)
                        - offset_samples
                    for unit_id in sorting.unit_ids
                }

                sorting_shifted = (
                    NumpySorting.from_unit_dict(
                        shifted_spike_trains,
                        sampling_frequency=
                            sorting.sampling_frequency
                    )
                )

                spike_counts = (
                    sorting_shifted
                    .count_num_spikes_per_unit()
                )

                logger.info(
                    f"Spike counts: {spike_counts}"
                )


                # ------------------------------------------------
                # GET RECORDING CHANNEL
                # ------------------------------------------------

                matches = ch_lookup[
                    ch_lookup["csc_nr"] == ch_nr
                ]["obj_name"]

                if len(matches) == 0:
                    logger.error(
                        f"No recording channel found for CSC{ch_nr}"
                    )
                    continue

                ch_id = matches.iloc[0]

                logger.info(
                    f"Recording channel: {ch_id}"
                )

                recording_csc = recording.select_channels(
                    [ch_id]
                )

                recording_csc.set_dummy_probe_from_locations(
                    locations=np.array([[0, 0]])
                )


                # ------------------------------------------------
                # OUTPUT DIRECTORY
                # ------------------------------------------------

                analyzer_folder = (
                    output_root
                    / session_name
                    / f"CSC{ch_nr}"
                )

                analyzer_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                logger.info(
                    f"Analyzer output: {analyzer_folder}"
                )


                # ------------------------------------------------
                # CREATE ANALYZER
                # ------------------------------------------------

                logger.info(
                    "Creating sorting analyzer..."
                )

                analyzer = create_sorting_analyzer(
                    sorting=sorting_shifted,
                    recording=recording_csc,
                    format="binary_folder",
                    folder=analyzer_folder,
                    sparse=False,
                    overwrite=True,
                )


                # =================================================
                # COMPUTE EXTENSIONS
                # =================================================

                logger.info("Computing random_spikes...")
                analyzer.compute("random_spikes")

                logger.info("Computing waveforms...")
                analyzer.compute("waveforms")

                logger.info("Computing templates...")
                analyzer.compute("templates")

                logger.info("Computing noise_levels...")
                analyzer.compute("noise_levels")

                logger.info("Computing correlograms...")
                analyzer.compute("correlograms")

                logger.info("Computing spike_amplitudes...")
                analyzer.compute("spike_amplitudes")

                logger.info("Computing template_similarity...")
                analyzer.compute("template_similarity")


                # ------------------------------------------------
                # TEMPLATE METRICS
                # ------------------------------------------------

                logger.info(
                    f"Computing template metrics: "
                    f"{template_metric_names}"
                )

                analyzer.compute(
                    "template_metrics",
                    metric_names=template_metric_names
                )


                # ------------------------------------------------
                # QUALITY METRICS
                # ------------------------------------------------

                logger.info(
                    f"Computing quality metrics: "
                    f"{metric_names}"
                )

                analyzer.compute(
                    "quality_metrics",
                    metric_names=metric_names
                )


                # ------------------------------------------------
                # OPTIONAL LOCATION EXTENSIONS
                # ------------------------------------------------

                # These can produce warnings with a one-channel
                # recording, so they are intentionally omitted.
                #
                # analyzer.compute("unit_locations")
                # analyzer.compute("spike_locations")


                # ------------------------------------------------
                # FINISHED CSC
                # ------------------------------------------------

                logger.info(
                    f"SUCCESS: {session_name} / CSC{ch_nr}"
                )

                logger.info(
                    f"Results saved to: {analyzer_folder}"
                )


            except Exception:

                logger.exception(
                    f"FAILED: {session_name} / CSC{ch_nr}"
                )

                # Continue with the next CSC rather than
                # terminating the entire analysis.
                continue


    except Exception:

        logger.exception(
            f"FAILED SESSION: {session_name}"
        )

        # Continue with the next session.
        continue


# ============================================================
# FINISHED
# ============================================================

logger.info("")
logger.info("=" * 70)
logger.info("UnitRefine analysis finished")
logger.info(f"Log file: {log_file}")
logger.info(f"Results: {output_root}")
logger.info("=" * 70)
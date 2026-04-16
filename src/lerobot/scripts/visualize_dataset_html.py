#!/usr/bin/env python

# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
#
# This file is modified from the lerobot project:
# https://github.com/huggingface/lerobot
# Original copyright: Copyright 2024 The Hugging Face team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python -m lerobot.scripts.visualize_dataset_html \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import json
import logging
import shutil
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for

from lerobot import available_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


class IterableNamespace(SimpleNamespace):
    """A namespace that supports dict-like iteration and dot/bracket access."""

    def __init__(self, dictionary: dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

    def keys(self):
        return vars(self).keys()


def run_server(
    dataset: LeRobotDataset | IterableNamespace | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def hommepage(dataset=dataset):
        if dataset:
            dataset_namespace, dataset_name = dataset.repo_id.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param = None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])

        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        try:
            if dataset is None:
                dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. See how to convert your dataset at https://github.com/huggingface/lerobot/pull/461",
                400,
            )

        # v3.0: _version returns a packaging.Version object, str() gives "3.0" (no "v" prefix)
        if isinstance(dataset, LeRobotDataset):
            major_version = dataset.meta._version.major
        else:
            # IterableNamespace: codebase_version is a plain string like "v2.1" or "v3.0"
            codebase_version = dataset.codebase_version
            major_version = int(codebase_version.lstrip("v").split(".")[0])

        if major_version < 2:
            return "Make sure to convert your LeRobotDataset to v2 & above."

        episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
        dataset_info = {
            "repo_id": f"{dataset_namespace}/{dataset_name}",
            "num_samples": dataset.num_frames
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_frames,
            "num_episodes": dataset.num_episodes
            if isinstance(dataset, LeRobotDataset)
            else dataset.total_episodes,
            "fps": dataset.fps,
        }
        if isinstance(dataset, LeRobotDataset):
            video_paths = [
                dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
            ]
            videos_info = [
                {
                    "url": url_for("static", filename=str(video_path).replace("\\", "/")),
                    "filename": video_path.parent.name,
                }
                for video_path in video_paths
            ]
            # v3.0: meta.episodes is a datasets.Dataset; episodes[ep_id]["tasks"] returns a list
            tasks = dataset.meta.episodes[episode_id]["tasks"]
        else:
            video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
            # v3.0 remote: video_path format uses {chunk_index} and {file_index}
            # Fall back to episode_index-based naming for v2 remote datasets
            videos_info = []
            for video_key in video_keys:
                try:
                    url = (
                        f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                        + dataset.video_path.format(
                            chunk_index=int(episode_id) // dataset.chunks_size,
                            file_index=episode_id % dataset.chunks_size,
                            video_key=video_key,
                            # v2 fallback keys (ignored if not in format string)
                            episode_chunk=int(episode_id) // dataset.chunks_size,
                            episode_index=episode_id,
                        )
                    )
                except KeyError:
                    url = ""
                videos_info.append({"url": url, "filename": video_key})

            response = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl", timeout=5
            )
            response.raise_for_status()
            tasks_jsonl = [json.loads(line) for line in response.text.splitlines() if line.strip()]
            filtered_tasks_jsonl = [row for row in tasks_jsonl if row["episode_index"] == episode_id]
            tasks = filtered_tasks_jsonl[0]["tasks"]

        videos_info[0]["language_instruction"] = tasks

        if episodes is None:
            episodes = list(
                range(dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes)
            )

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            episode_data_csv_str=episode_data_csv_str,
            columns=columns,
            ignored_columns=ignored_columns,
        )

    app.run(host=host, port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def get_episode_data(dataset: LeRobotDataset | IterableNamespace, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    selected_columns.remove("timestamp")

    ignored_columns = []
    for column_name in selected_columns:
        shape = dataset.features[column_name]["shape"]
        shape_dim = len(shape)
        if shape_dim > 1:
            selected_columns.remove(column_name)
            ignored_columns.append(column_name)

    # init header of csv with state and action names
    header = ["timestamp"]

    for column_name in selected_columns:
        dim_state = (
            dataset.meta.shapes[column_name][0]
            if isinstance(dataset, LeRobotDataset)
            else dataset.features[column_name].shape[0]
        )

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    if isinstance(dataset, LeRobotDataset):
        # v3.0: use meta.episodes["dataset_from_index/to_index"] instead of episode_data_index
        ep = dataset.meta.episodes[episode_index]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        data = (
            dataset.hf_dataset.select(range(from_idx, to_idx))
            .select_columns(selected_columns)
            .with_format("pandas")
        )
    else:
        repo_id = dataset.repo_id
        # v3.0 remote: data_path uses {chunk_index} and {file_index}
        try:
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
                chunk_index=int(episode_index) // dataset.chunks_size,
                file_index=episode_index % dataset.chunks_size,
                # v2 fallback
                episode_chunk=int(episode_index) // dataset.chunks_size,
                episode_index=episode_index,
            )
        except KeyError:
            url = ""
        df = pd.read_parquet(url)
        data = df[selected_columns]

    timestamps = np.array(data["timestamp"])
    timestamps = timestamps - timestamps[0]  # normalize to start from 0 for each episode

    rows = np.hstack(
        (
            np.expand_dims(timestamps, axis=1),
            *[np.vstack(data[col]) for col in selected_columns[1:]],
        )
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(header)
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # v3.0: use meta.episodes instead of episode_data_index
    first_frame_idx = dataset.meta.episodes[ep_index]["dataset_from_index"]
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.features:
        return None

    # v3.0: use meta.episodes instead of episode_data_index
    first_frame_idx = dataset.meta.episodes[ep_index]["dataset_from_index"]

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


def get_dataset_info(repo_id: str) -> IterableNamespace:
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    logo_src = template_dir / "dexteleop.png"
    logo_dst = static_dir / "dexteleop.png"
    if logo_src.exists() and not logo_dst.exists():
        shutil.copy2(logo_src, logo_dst)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )
    else:
        # Create a simlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    dataset = None
    if repo_id:
        dataset = (
            LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
            if not load_from_hf_hub
            else get_dataset_info(repo_id)
        )

    visualize_dataset_html(dataset, **vars(args))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import textwrap
from pathlib import Path


NB_PATH = Path(__file__).resolve().parent / "piano_ablation_kaggle.ipynb"


def _to_lines(src: str) -> list[str]:
    return (textwrap.dedent(src).strip("\n") + "\n").splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Fix 6: project root search roots.
    env_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "def find_project_root()" in src and "search_roots" in src:
            env_idx = i
            break
    if env_idx is not None:
        src = "".join(cells[env_idx]["source"])
        old = 'search_roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]'
        new = (
            "search_roots = [\n"
            '        Path("/kaggle/input/datasets/chickaboomcmurtrie/magic-midi"),\n'
            '        Path("/kaggle/input/datasets"),\n'
            '        Path("/kaggle/input"),\n'
            '        Path("/kaggle/working"),\n'
            "        Path.cwd(),\n"
            "    ]"
        )
        if old in src:
            src = src.replace(old, new)
        cells[env_idx]["source"] = src.splitlines(keepends=True)

    # Fix 1: update config path and candidates.
    cfg_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            "USE_PRETOKENIZED_KAGGLE_DATASET" in src
            and "PRETOKENIZED_DATASET_PATH" in src
        ):
            cfg_idx = i
            break
    if cfg_idx is None:
        raise RuntimeError("Could not locate CONFIG cell.")

    cfg_src = "".join(cells[cfg_idx]["source"])
    cfg_src = cfg_src.replace(
        'PRETOKENIZED_DATASET_PATH = "/kaggle/input/godzilla-piano-tokenized-15k/tokenized"\n',
        'PRETOKENIZED_DATASET_PATH = "/kaggle/input/datasets/chickaboomcmurtrie/godzilla-tokenized-15k/tokenized"\n',
    )
    if "PRETOKENIZED_DATASET_CANDIDATES" not in cfg_src:
        anchor = 'PRETOKENIZED_DATASET_PATH = "/kaggle/input/datasets/chickaboomcmurtrie/godzilla-tokenized-15k/tokenized"\n'
        candidates = (
            "PRETOKENIZED_DATASET_CANDIDATES = [\n"
            '    "/kaggle/input/datasets/chickaboomcmurtrie/godzilla-tokenized-15k/tokenized",\n'
            '    "/kaggle/input/datasets/chickaboomcmurtrie/godzilla-tokenized-15k",\n'
            '    "/kaggle/input/godzilla-tokenized-15k/tokenized",\n'
            '    "/kaggle/input/godzilla-tokenized-15k",\n'
            "]\n"
        )
        if anchor in cfg_src:
            cfg_src = cfg_src.replace(anchor, anchor + candidates)
    cfg_src = cfg_src.replace("SKIP_DATASET_SETUP = False\n", "")
    cells[cfg_idx]["source"] = cfg_src.splitlines(keepends=True)

    # Fix 2: add auto-detect cell right after config wall-clock cell.
    auto_detect_exists = any(
        c.get("cell_type") == "code"
        and "def find_tokenized_dataset()" in "".join(c.get("source", []))
        for c in cells
    )
    if not auto_detect_exists:
        auto_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _to_lines(
                """
                from pathlib import Path

                def find_tokenized_dataset():
                    for candidate in PRETOKENIZED_DATASET_CANDIDATES:
                        p = Path(candidate)
                        if p.exists():
                            npz_files = list(p.glob("*.npz"))
                            if len(npz_files) > 0:
                                print(f"Found tokenized dataset: {p}")
                                print(f"NPZ files: {len(npz_files)}")
                                return str(p)
                    raise RuntimeError(
                        "Could not find tokenized dataset in any candidate path.\n"
                        "Paths checked:\n"
                        + "\n".join(f"  {c}" for c in PRETOKENIZED_DATASET_CANDIDATES)
                        + "\nMake sure godzilla-tokenized-15k is added as input dataset."
                    )

                PRETOKENIZED_DATASET_PATH = find_tokenized_dataset()
                DATA_DIR = PRETOKENIZED_DATASET_PATH
                print(f"DATA_DIR set to: {DATA_DIR}")
                """
            ),
        }
        # find config wall-clock cell index
        insert_at = None
        for i, c in enumerate(cells):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            if "CONFIG wall-clock" in src:
                insert_at = i + 1
                break
        if insert_at is None:
            insert_at = 9
        cells.insert(insert_at, auto_cell)

    # Recompute references after potential insertion.
    # Fix 5 + part of Fix 3: dataset setup pretokenized path always rebuilds manifest.
    ds_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            "if USE_PRETOKENIZED_KAGGLE_DATASET:" in src
            and "Pre-tokenized dataset not found" in src
        ):
            ds_idx = i
            break
    if ds_idx is None:
        raise RuntimeError("Could not locate DATASET SETUP cell.")

    old_ds = "".join(cells[ds_idx]["source"])
    else_marker = "\nelse:\n"
    if else_marker in old_ds:
        old_else_body = old_ds.split(else_marker, 1)[1]
    else:
        old_else_body = ""

    pretokenized_prefix = _to_lines(
        """
        from pathlib import Path
        import json
        import numpy as np

        if USE_PRETOKENIZED_KAGGLE_DATASET:
            pretok_path = Path(PRETOKENIZED_DATASET_PATH)
            if not pretok_path.exists():
                raise RuntimeError(
                    f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_PATH}. "
                    "Make sure you have added godzilla-piano-tokenized-15k as an "
                    "input dataset to this Kaggle notebook. "
                    "Go to notebook settings -> Add data -> Your datasets -> "
                    "godzilla-piano-tokenized-15k"
                )
            files = list(pretok_path.glob("*.npz"))
            if len(files) == 0:
                raise RuntimeError(
                    f"No .npz files found in {PRETOKENIZED_DATASET_PATH}. "
                    "Dataset may be empty or path is wrong."
                )

            # Always rebuild manifest in /kaggle/working for this session.
            manifest = []
            for npz_path in sorted(pretok_path.glob("*.npz")):
                try:
                    data = np.load(npz_path)
                    manifest.append(
                        {
                            "md5": npz_path.stem,
                            "npz_path": str(npz_path.absolute()),
                            "length": int(len(data["tokens"])),
                            "source_path": "",
                        }
                    )
                except Exception as e:
                    print(f"Skipping {npz_path.name}: {e}")

            manifest_out = Path("/kaggle/working/manifest.json")
            with open(manifest_out, "w") as f:
                json.dump(manifest, f)

            DATA_DIR = str(pretok_path)
            MANIFEST_PATH = str(manifest_out)
            print(f"Using pre-tokenized dataset: {DATA_DIR}")
            print(f"Files found: {len(files)}")
            print(f"Manifest rebuilt: {len(manifest)} files -> {MANIFEST_PATH}")
            # Skip all download/tokenization
        else:
        """
    )
    cells[ds_idx]["source"] = pretokenized_prefix + textwrap.indent(
        old_else_body,
        "    ",
    ).splitlines(keepends=True)

    # Fix 3: manifest rebuild cell content update (absolute npz path + working output)
    rebuild_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            "pretok_path = Path(PRETOKENIZED_DATASET_PATH)" in src
            and "Manifest rebuilt:" in src
        ):
            rebuild_idx = i
            break
    if rebuild_idx is not None:
        cells[rebuild_idx]["source"] = _to_lines(
            """
            from pathlib import Path
            import json
            import numpy as np

            pretok_path = Path(PRETOKENIZED_DATASET_PATH)
            manifest = []
            for npz_path in sorted(pretok_path.glob("*.npz")):
                try:
                    data = np.load(npz_path)
                    manifest.append(
                        {
                            "md5": npz_path.stem,
                            "npz_path": str(npz_path.absolute()),
                            "length": int(len(data["tokens"])),
                            "source_path": "",
                        }
                    )
                except Exception as e:
                    print(f"Skipping {npz_path.name}: {e}")

            manifest_out = pretok_path / "manifest.json"
            # Note: Kaggle input datasets are read-only so save to working dir
            manifest_out = Path("/kaggle/working/manifest.json")
            with open(manifest_out, "w") as f:
                json.dump(manifest, f)
            MANIFEST_PATH = str(manifest_out)
            print(f"Manifest rebuilt: {len(manifest)} files -> {manifest_out}")
            """
        )

    # Fix 4: data loading cell with robust npz resolution helper.
    load_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "manifest_path = Path(MANIFEST_PATH)" in src and "resolved_manifest" in src:
            load_idx = i
            break
    if load_idx is None:
        raise RuntimeError("Could not locate data loading cell.")

    cells[load_idx]["source"] = _to_lines(
        """
        import json
        from pathlib import Path
        from typing import Any, Dict, List

        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        from data.dataset import PianoDataset


        def resolve_npz_path(entry, data_dir):
            p = Path(entry["npz_path"])
            if p.is_absolute() and p.exists():
                return p
            direct = Path(data_dir) / p.name
            if direct.exists():
                return direct
            return None


        manifest_path = Path(MANIFEST_PATH)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            raw_manifest = json.load(f)

        if not isinstance(raw_manifest, list) or not raw_manifest:
            raise RuntimeError("Manifest is empty or invalid")

        resolved_manifest: List[Dict[str, Any]] = []
        skipped_unresolved = 0

        for row in raw_manifest:
            if not isinstance(row, dict):
                skipped_unresolved += 1
                continue

            npz_path = row.get("npz_path")
            if isinstance(npz_path, str) and npz_path:
                p = resolve_npz_path({"npz_path": npz_path}, DATA_DIR)
            else:
                md5 = str(row.get("md5", "")).strip()
                if not md5:
                    skipped_unresolved += 1
                    continue
                p = resolve_npz_path({"npz_path": f"{md5}.npz"}, DATA_DIR)

            if p is None or not p.exists():
                skipped_unresolved += 1
                print(f"Warning: could not resolve npz for entry md5={row.get('md5', '')}")
                continue

            length = int(row.get("length", row.get("tokens", -1)))
            if length <= 0:
                with np.load(p, allow_pickle=False) as pack:
                    length = int(pack["tokens"].shape[0])

            resolved_manifest.append(
                {
                    "piece_id": str(row.get("md5", p.stem)),
                    "tokens_path": str(p.resolve()),
                    "onset_times_path": "",
                    "durations_path": "",
                    "length": int(length),
                    "tokens": int(length),
                    "source": "godzilla_piano",
                }
            )

        print(
            f"Manifest entries resolved successfully: {len(resolved_manifest):,} | "
            f"skipped: {skipped_unresolved:,}"
        )
        if not resolved_manifest:
            raise RuntimeError("No valid .npz files resolved from manifest")

        lengths = np.asarray([int(m["length"]) for m in resolved_manifest], dtype=np.int64)
        print(f"Total files: {len(resolved_manifest):,}")
        print(
            f"Length mean/min/max: "
            f"{lengths.mean():.2f}/{lengths.min()}/{lengths.max()}"
        )
        print(
            "Length percentiles p50/p90/p99:",
            np.percentile(lengths, [50, 90, 99]).tolist(),
        )

        idx = np.arange(len(resolved_manifest))
        rng = np.random.default_rng(GLOBAL_SEED)
        rng.shuffle(idx)

        n_val = max(1, int(0.1 * len(idx)))
        val_idx = set(idx[:n_val].tolist())
        train_manifest = [
            resolved_manifest[i] for i in range(len(resolved_manifest)) if i not in val_idx
        ]
        val_manifest = [
            resolved_manifest[i] for i in range(len(resolved_manifest)) if i in val_idx
        ]

        print(f"Train files: {len(train_manifest):,}")
        print(f"Val files: {len(val_manifest):,}")

        tokenizer = CustomDeltaTokenizer(include_special_tokens=False)


        class NpzWindowDataset(PianoDataset):
            # Adapt existing PianoDataset windowing to .npz tokenized files.

            def __getitem__(self, idx: int):
                item = self.manifest[idx]
                npz_path = Path(str(item["tokens_path"]))
                with np.load(npz_path, allow_pickle=False) as pack:
                    token_seq = np.asarray(pack["tokens"], dtype=np.int64)
                    onset_seq = np.asarray(pack["onsets"], dtype=np.float32)
                    duration_seq = np.asarray(pack["durations"], dtype=np.float32)

                total_needed = (
                    self.data_config.seed_length + self.data_config.continuation_length
                )
                if token_seq.shape[0] < total_needed:
                    raise RuntimeError(
                        f"Piece {npz_path} shorter than required window {total_needed}"
                    )

                max_start = int(token_seq.shape[0] - total_needed)
                raw_start = self.rng.randint(0, max_start) if max_start > 0 else 0
                start = self._snap_to_triplet_boundary(raw_start, max_start)
                if start % 3 != 0:
                    raise AssertionError(f"Triplet boundary violation: {start}")

                seed = token_seq[start : start + self.data_config.seed_length]
                cont = token_seq[
                    start + self.data_config.seed_length : start + total_needed
                ]
                onset = onset_seq[start : start + total_needed]
                duration = duration_seq[start : start + total_needed]

                seed_t = torch.from_numpy(seed.astype(np.int64, copy=False))
                cont_t = torch.from_numpy(cont.astype(np.int64, copy=False))
                onset_t = torch.from_numpy(onset.astype(np.float32, copy=False))
                duration_t = torch.from_numpy(duration.astype(np.float32, copy=False))

                return {
                    "seed": seed_t,
                    "continuation": cont_t,
                    "token_ids": torch.cat([seed_t, cont_t], dim=0),
                    "onset_times": onset_t,
                    "durations": duration_t,
                    "new_piece": torch.tensor(True),
                }


        data_config = DataConfig(
            tokenizer_path="",
            processed_path=str(Path(DATA_DIR).resolve()),
            vocab_size=155,
            tokenization_strategy="custom_delta",
            seed_length=SEED_LENGTH,
            continuation_length=CONTINUATION_LENGTH,
            max_sequence_length=SEED_LENGTH + CONTINUATION_LENGTH,
            use_continuous_time=True,
            time_feature_fallback_step_seconds=0.1,
        )

        train_dataset = NpzWindowDataset(train_manifest, data_config, seed=GLOBAL_SEED)
        val_dataset = NpzWindowDataset(val_manifest, data_config, seed=GLOBAL_SEED + 1)

        num_workers = 2
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            collate_fn=PianoDataset.collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            collate_fn=PianoDataset.collate_fn,
            drop_last=False,
        )

        sample_batch = next(iter(train_loader))
        print("Sample batch shape check:")
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {tuple(v.shape)} | {v.dtype}")
        """
    )

    # Keep requested label text.
    for c in cells:
        if c.get("cell_type") == "markdown" and "MANIFEST REBUILD" in "".join(
            c.get("source", [])
        ):
            c["source"] = [
                "## MANIFEST REBUILD — only run if manifest.json is missing\n"
            ]

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Notebook updated: {NB_PATH}")


if __name__ == "__main__":
    main()

"""
Main entry point for data processing pipeline.
Usage: python main.py [config_file]
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path

# Ensure we can import from src regardless of how we are run
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # Try importing as if we are inside the package (relative or direct)
    from src.chunking import ChunkingConfig
    from src.pipeline import parse_markdown_and_chunk, run_docling_pipeline
except ImportError:
    # Fallback for when running from root as `python data_processing/main.py`
    # and sys.path setup above might behave differently depending on env
    try:
        from data_processing.src.chunking import ChunkingConfig
        from data_processing.src.pipeline import (
            parse_markdown_and_chunk,
            run_docling_pipeline,
        )
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
_log = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from a JSON or YAML file."""
    if not config_path.exists():
        _log.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Data Processing Pipeline")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=Path,
        help="Path to configuration file (JSON/YAML).",
    )
    # Allow overriding/specifying inputs via CLI if no config or in addition
    parser.add_argument("--input", type=Path, help="Input file path.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output directory."
    )

    args = parser.parse_args()

    # Default configuration
    config_data = {
        "chunk_token_num": 512,
        "overlapped_percent": 0.0,
        "table_context_size": 50,
        "image_context_size": 30,
        "with_children": False,
        "mode": "auto",
    }

    # Load from file if provided
    if args.config_file:
        _log.info("Loading config from %s", args.config_file)
        file_config = load_config(args.config_file)
        config_data.update(file_config)

    # CLI args override config file
    if args.input:
        config_data["input"] = str(args.input)
    if args.output_dir:
        config_data["output_dir"] = str(args.output_dir)

    if "input" not in config_data:
        parser.print_help()
        _log.error("\nNo input file specified in config or arguments.")
        sys.exit(1)

    input_path = Path(config_data["input"])
    output_dir = Path(config_data.get("output_dir", "output"))

    # Create ChunkingConfig object
    chunk_config = ChunkingConfig(
        chunk_token_num=config_data.get("chunk_token_num", 512),
        overlapped_percent=config_data.get("overlapped_percent", 0.0),
        table_context_size=config_data.get("table_context_size", 50),
        image_context_size=config_data.get("image_context_size", 30),
        enable_child_chunks=config_data.get("with_children", False),
        include_children_in_output=config_data.get("with_children", False),
        attach_media_context=True,
        # Allow passing other advanced config keys dynamically
        **{
            k: v
            for k, v in config_data.items()
            if k
            in [
                "delimiter",
                "custom_delimiters",
                "min_chunk_tokens",
                "hierarchical_merge",
                "hierarchical_depth",
                "attach_images_to_section",
            ]
        },
    )

    mode = config_data.get("mode", "auto")
    if mode == "auto":
        mode = "markdown" if input_path.suffix.lower() == ".md" else "docling"

    if mode == "docling":
        _log.info("Running Docling pipeline on %s...", input_path)
        out_file = run_docling_pipeline(input_path, output_dir, config=chunk_config)
        _log.info("Pipeline finished. Chunks saved to: %s", out_file)
    else:
        _log.info("Running Markdown chunking on %s...", input_path)
        if not input_path.exists():
            _log.error("Input file not found: %s", input_path)
            sys.exit(1)

        chunks = parse_markdown_and_chunk(input_path, config=chunk_config)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / "chunks.json"

        payload = {
            "source": str(input_path),
            "config": chunk_config.__dict__,
            "chunks": chunks,
        }
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _log.info("Chunking finished. %d chunks saved to: %s", len(chunks), out_json)


if __name__ == "__main__":
    main()

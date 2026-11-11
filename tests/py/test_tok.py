"""
Pytest integration tests for Qwen2 BPE tokenizer.
Validates parity between HuggingFace tokenizer, fixtures, and C implementation.
"""

import os
import sys
import json
import struct
import subprocess
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures" / "tok"
BUILD_DIR = ROOT_DIR / "build"


@pytest.fixture(scope="module")
def tok_fixtures():
    """Ensure fixtures exist."""
    vocab_file = FIXTURES_DIR / "vocab.json"
    merges_file = FIXTURES_DIR / "merges.txt"
    corpus_file = FIXTURES_DIR / "corpus.txt"
    ids_json_file = FIXTURES_DIR / "corpus_ids.json"
    ids_bin_file = FIXTURES_DIR / "corpus_ids.bin"

    if not (vocab_file.exists() and merges_file.exists() and corpus_file.exists() and ids_json_file.exists()):
        from tools.tok_parity import setup_fixtures
        setup_fixtures()

    return {
        "vocab": vocab_file,
        "merges": merges_file,
        "corpus": corpus_file,
        "ids_json": ids_json_file,
        "ids_bin": ids_bin_file,
    }


def test_fixtures_presence_and_structure(tok_fixtures):
    """Verify fixtures are present, non-empty, and structurally valid."""
    assert tok_fixtures["vocab"].exists()
    assert tok_fixtures["vocab"].stat().st_size > 2_000_000
    assert tok_fixtures["merges"].exists()
    assert tok_fixtures["merges"].stat().st_size > 1_000_000
    assert tok_fixtures["corpus"].exists()
    assert tok_fixtures["ids_json"].exists()
    assert tok_fixtures["ids_bin"].exists()

    with open(tok_fixtures["ids_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 30, f"Expected at least 30 lines, got {len(data)}"

    # Check ~600-token prompt
    prompt_entry = data[-1]
    assert 500 <= prompt_entry["n"] <= 700, f"Prompt tokens {prompt_entry['n']} not in range [500, 700]"


def test_hf_tokenizer_parity_with_corpus_ids(tok_fixtures):
    """Verify that HuggingFace AutoTokenizer matches golden corpus_ids bit-identically."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = Path(r"C:\Users\sbaih\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987")
    if not cache_dir.exists():
        pytest.skip("Local HuggingFace model cache not present")

    with open(tok_fixtures["ids_json"], "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    tok = AutoTokenizer.from_pretrained(str(cache_dir))

    for item in golden_data:
        text = item["text"]
        expected_ids = item["ids"]
        actual_ids = tok.encode(text, add_special_tokens=False)
        assert actual_ids == expected_ids, f"Mismatch on line {item['line_idx']}: {text!r}"


def test_corpus_ids_bin_format(tok_fixtures):
    """Verify binary fixture integrity and match with corpus_ids.json."""
    with open(tok_fixtures["ids_json"], "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(tok_fixtures["ids_bin"], "rb") as f:
        n_lines = struct.unpack("<I", f.read(4))[0]
        assert n_lines == len(json_data)

        for i in range(n_lines):
            text_len = struct.unpack("<I", f.read(4))[0]
            text = f.read(text_len).decode("utf-8")
            n_ids = struct.unpack("<I", f.read(4))[0]
            ids = list(struct.unpack(f"<{n_ids}i", f.read(n_ids * 4)))

            assert text == json_data[i]["text"]
            assert ids == json_data[i]["ids"]


def test_windows_crlf_semantics(tok_fixtures):
    """Verify CRLF behavior matches specification: CRLF token differs from LF unless normalized."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    cache_dir = Path(r"C:\Users\sbaih\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987")
    if not cache_dir.exists():
        pytest.skip("Local HuggingFace model cache not present")

    tok = AutoTokenizer.from_pretrained(str(cache_dir))

    text_crlf = "Hello\r\nWorld"
    text_lf = "Hello\nWorld"

    ids_crlf = tok.encode(text_crlf, add_special_tokens=False)
    ids_lf = tok.encode(text_lf, add_special_tokens=False)

    # Token for \r\n (319) vs \n (198)
    assert ids_crlf != ids_lf
    assert 319 in ids_crlf
    assert 198 in ids_lf

    # Normalized text must produce identical tokens to LF
    text_crlf_normalized = text_crlf.replace("\r\n", "\n")
    assert tok.encode(text_crlf_normalized, add_special_tokens=False) == ids_lf


def test_c_tokenizer_binary():
    """Verify that the C test binary executes and passes all test assertions."""
    exe_name = "test_tok.exe" if sys.platform == "win32" else "test_tok"
    exe_path = BUILD_DIR / exe_name
    if not exe_path.exists():
        pytest.skip(f"Binary {exe_path} not built yet")

    result = subprocess.run([str(exe_path)], capture_output=True, text=True, cwd=str(ROOT_DIR))
    assert result.returncode == 0, f"{exe_name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "ALL TOKENIZER UNIT TESTS PASSED!" in result.stdout

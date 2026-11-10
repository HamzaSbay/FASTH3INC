"""
Tokenizer parity tool: sets up vocab/merges fixtures, generates corpus.txt,
and computes golden token IDs from HuggingFace Qwen2.5-0.5B tokenizer.
"""

import os
import sys
import json
import struct
import shutil
from pathlib import Path

HF_CACHE_DIR = Path(r"C:\Users\sbaih\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987")
FIXTURES_DIR = Path("tests/fixtures/tok")

PROMPT_600_TOKENS = (
    "A breathtaking cinematic wide-angle establishing shot of an awe-inspiring alpine sanctuary at the golden hour of dawn. "
    "Towering snow-draped granite peaks catch the very first amber rays of morning sunlight, casting a magnificent warm illumination across the rugged ridges, sheer precipices, and ancient glacial moraines. "
    "Below the colossal peaks, a pristine turquoise glacial lake spreads out like an immaculate crystal mirror, perfectly undisturbed by even the faintest ripple, reflecting the inverted majesty of the sky and mountain ramparts in crystalline clarity. "
    "Delicate translucent ribbons of luminous morning mist drift gently across the tranquil water surface and weave gracefully through dense old-growth stands of deep emerald spruce, majestic Douglas firs, and fragrant alpine larch trees. "
    "In the vibrant foreground, a rich carpet of alpine flora—azure forget-me-nots, golden buttercups, delicate edelweiss, and fiery crimson alpine rhododendrons—sways softly in the crisp, refreshing mountain breeze, each petal glistening with trembling droplets of morning dew that prismatically refract the rising sun. "
    "A pristine crystal-clear mountain brook cascades joyfully over centuries-old moss-covered river stones, tumbling down from a glacier-fed waterfall in a shimmering series of bubbling rapids before merging serenely into the mirror-like lake. "
    "Perched gracefully on a gentle emerald knoll at the forest edge stands an ancient timber cabin built from hand-hewn cedar logs, featuring hand-split shake shingles and a slender plume of aromatic pale woodsmoke curling lazily from a handcrafted fieldstone chimney into the pale lavender morning sky. "
    "The camera executes an ultra-smooth cinematic crane and tracking movement, gliding gracefully inches above the dew-drenched meadow wildflowers, ascending gracefully over the sparkling stream, and rotating gently in three dimensions to reveal the staggering panoramic depth of the sunlit valley. "
    "Every minute texture is rendered with hyper-realistic 8k visual clarity: the coarse fissures of the granite monoliths, the delicate crystalline lattice of fresh snowdrifts, individual droplets of dew trembling on grass blades, and dynamic volumetric god rays streaming through the canopy with visible atmospheric particles. "
    "The lighting transitions seamlessly from cool lavender pre-dawn shadows in the valley floor to brilliant, blazing amber highlights across the summit ridge, captured with full dynamic range and zero clipping. "
    "Accompanied by an immersive, multi-layered acoustic soundscape recorded with spatial fidelity: the distant resonant rumble of falling meltwater echoing through the canyon, the crisp chirp of high-altitude songbirds welcoming the day, the gentle rhythmic rustle of alpine pine needles stirred by a soft wind, and the delicate acoustic lap of tiny water ripples against the pebble shoreline, creating a deep sense of peaceful solitude and timeless wilderness."
)

CORPUS_LINES = [
    # English with contractions
    "don't you know it's true they'll find out we've seen it all",
    "I'm sure you'd agree that wouldn't happen if she'd known",
    "They're going to tell us who's responsible and what's next",
    "DON'T SHOUT IT'S LOUD AND THEY'LL HEAR US",
    # Chinese (CJK)
    "一只红色小狐狸在雪地奔跑",
    "快速三维生成与高保真视频合成引擎。",
    "天行健，君子以自强不息；地势坤，君子以厚德载物。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    # Japanese
    "日本語のテストです。カタカナと漢字。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "春はあけぼの。やうやう白くなりゆく山ぎは、少しあかりて、紫だちたる雲の細くたなびきたる。",
    # Korean
    "한국어 테스트입니다.",
    "인공지능 언어 모델 텍스트 토크나이저 고속 C 언어 구현.",
    "나랏말싸미 듕귁에 달아 문자와로 서르 사맛디 아니할쎄.",
    # Emojis & complex symbols
    "🦊 ❄️ 🚀 🌟 ✨ 🎉 🔥 💡 🤖 🎬",
    "Family emoji: 👨‍👩‍👧‍👦 Rainbow flag: 🏳️‍🌈 Hand wave: 👋🏽",
    "Symbols: ☯️ ⚛️ ⚡ ☕ ☕️ © ® ™ ✓ ✗ → ⇐ ⇄",
    # Accents & Latin extensions
    "café Über naïve señor résumé façade voilà smörgåsbord",
    "El niño comió jalapeños en el zócalo de la ciudad",
    "Über den Wolken muss die Freiheit wohl grenzenlos sein",
    # Numbers, currencies, mathematics
    "12345 67890 3.141592653589793 -42 +100 1e-5 2.71828",
    "$1,000,000.00 €49.99 £100 ¥10,000 ₹500 50% discount",
    "x <= y && a != b || (c & d) ^ ~e >> 2",
    "E = mc^2 and ∫ e^(-x^2) dx = √π",
    # Whitespace patterns
    "   leading three spaces",
    "trailing three spaces   ",
    "multiple   spaces   between   words",
    "\t\ttwo leading tabs and one\ttab inside",
    "   \t   mixed indentation   \t   ",
    # Code snippet
    "def forward(self, x: torch.Tensor, timesteps: torch.LongTensor) -> torch.Tensor:",
    "int fh3_tok_encode(const Fh3Tok *t, const char *utf8, int **ids, int *n);",
    # Punctuation torture test
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "mix: 1+1=2, a_b_c, foo-bar, test@example.com, https://qwenlm.ai",
    # ~600 token prompt paragraph
    PROMPT_600_TOKENS,
]


def setup_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    vocab_dst = FIXTURES_DIR / "vocab.json"
    merges_dst = FIXTURES_DIR / "merges.txt"

    vocab_src = HF_CACHE_DIR / "vocab.json"
    merges_src = HF_CACHE_DIR / "merges.txt"

    if not vocab_dst.exists():
        print(f"Copying {vocab_src} -> {vocab_dst}")
        shutil.copyfile(vocab_src.resolve(), vocab_dst)
    if not merges_dst.exists():
        print(f"Copying {merges_src} -> {merges_dst}")
        shutil.copyfile(merges_src.resolve(), merges_dst)

    # Write corpus.txt
    corpus_file = FIXTURES_DIR / "corpus.txt"
    print(f"Writing {corpus_file} ({len(CORPUS_LINES)} lines)...")
    with open(corpus_file, "w", encoding="utf-8", newline="\n") as f:
        for line in CORPUS_LINES:
            f.write(line + "\n")

    # Generate golden token IDs
    from transformers import AutoTokenizer
    print("Loading HuggingFace AutoTokenizer for Qwen/Qwen2.5-0.5B...")
    tok = AutoTokenizer.from_pretrained(str(HF_CACHE_DIR))

    golden_data = []
    total_tokens = 0
    for idx, line in enumerate(CORPUS_LINES):
        ids = tok.encode(line, add_special_tokens=False)
        golden_data.append({"line_idx": idx, "text": line, "ids": ids, "n": len(ids)})
        total_tokens += len(ids)

    print(f"Total lines: {len(golden_data)}, total tokens: {total_tokens}")
    prompt_tokens = len(golden_data[-1]["ids"])
    print(f"Prompt paragraph tokens: {prompt_tokens}")

    # Write corpus_ids.json
    ids_json_file = FIXTURES_DIR / "corpus_ids.json"
    with open(ids_json_file, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {ids_json_file}")

    # Write corpus_ids.bin
    ids_bin_file = FIXTURES_DIR / "corpus_ids.bin"
    with open(ids_bin_file, "wb") as f:
        f.write(struct.pack("<I", len(golden_data)))
        for item in golden_data:
            text_bytes = item["text"].encode("utf-8")
            f.write(struct.pack("<I", len(text_bytes)))
            f.write(text_bytes)
            ids = item["ids"]
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
    print(f"Wrote {ids_bin_file} ({ids_bin_file.stat().st_size} bytes)")


if __name__ == "__main__":
    setup_fixtures()

from typing import List, Literal, Optional, Tuple
from langchain_core.documents import Document

import cv2
import numpy as np
import pytesseract
from pytesseract import Output, TesseractNotFoundError
import os
import shutil
import json
from pathlib import Path

Mode = Literal["auto", "elements", "unstructured"]

# --------- NEW: robust tesseract configuration ----------
def _configure_tesseract(tesseract_cmd: Optional[str] = None) -> str:
    """
    Ensure pytesseract knows where the tesseract binary is.
    Returns the resolved path to tesseract.exe (or 'tesseract' if on PATH).
    Raises a helpful RuntimeError if not found.
    """
    # 1) explicit argument
    candidates: List[str] = []
    if tesseract_cmd:
        candidates.append(tesseract_cmd)

    # 2) environment variables (allow both common names)
    env_cmd = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    if env_cmd:
        candidates.append(env_cmd)

    # 3) PATH discovery
    which = shutil.which("tesseract")
    if which:
        candidates.append(which)

    # 4) common Windows locations (UB Mannheim defaults)
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    candidates.extend(common_paths)

    # normalize and test
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.exists():
            # set and verify
            pytesseract.pytesseract.tesseract_cmd = str(p)
            try:
                _ = pytesseract.get_tesseract_version()
                return str(p)
            except Exception:
                # try next candidate
                pass

    # Final attempt: maybe PATH but not yet verified
    try:
        _ = pytesseract.get_tesseract_version()
        return "tesseract"
    except Exception:
        raise RuntimeError(
            "Tesseract OCR not found from code. "
            "Set the absolute path via:\n"
            "  - pass tesseract_cmd= r'C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe'\n"
            "  - or set env var TESSERACT_CMD / TESSERACT_PATH to that path\n"
            "Also ensure the service/venv has permission to read it."
        )

# --------------------------------------------------------

def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None. Check the path.")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    white_ratio = np.mean(thr == 255)
    if white_ratio < 0.5:
        thr = cv2.bitwise_not(thr)
    kernel = np.ones((1, 1), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return thr

def _deskew(image: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(image, 50, 150)
        coords = np.column_stack(np.where(edges > 0))
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return image

def _bbox_to_rel(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x / width, y / height, (x + w) / width, (y + h) / height)

def _extract_blocks(image: np.ndarray, lang: str = "eng", psm: int = 3, oem: int = 3) -> List[dict]:
    h, w = image.shape[:2]
    config = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT, config=config)

    n = len(data["text"])
    blocks = {}

    for i in range(n):
        txt = data["text"][i] or ""
        if txt.strip() == "":
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        bnum = data.get("block_num", [0])[i]
        pnum = data.get("par_num", [0])[i]
        lnum = data.get("line_num", [0])[i]

        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        key = (bnum, pnum)

        if key not in blocks:
            blocks[key] = {
                "text_parts": [],
                "x0": x, "y0": y, "x1": x + bw, "y1": y + bh,
                "conf": [],
                "line_nums": set(),
                "block_num": bnum,
                "par_num": pnum,
            }
        else:
            blocks[key]["x0"] = min(blocks[key]["x0"], x)
            blocks[key]["y0"] = min(blocks[key]["y0"], y)
            blocks[key]["x1"] = max(blocks[key]["x1"], x + bw)
            blocks[key]["y1"] = max(blocks[key]["y1"], y + bh)

        blocks[key]["text_parts"].append(txt)
        blocks[key]["conf"].append(conf)
        blocks[key]["line_nums"].add(lnum)

    results = []
    for _, b in blocks.items():
        text = " ".join(b["text_parts"]).strip()
        if not text:
            continue
        avg_conf = float(np.mean([c for c in b["conf"] if c >= 0])) if any(c >= 0 for c in b["conf"]) else -1.0
        bbox_abs = (b["x0"], b["y0"], b["x1"] - b["x0"], b["y1"] - b["y0"])
        bbox_rel = _bbox_to_rel(bbox_abs, w, h)
        results.append({
            "text": text,
            "bbox_abs": bbox_abs,
            "bbox": bbox_rel,
            "avg_conf": avg_conf,
            "line_count": len(b["line_nums"]),
            "block_num": b["block_num"],
            "par_num": b["par_num"],
            "image_size": (w, h),
        })

    results.sort(key=lambda r: (r["bbox_abs"][1], r["bbox_abs"][0]))
    return results

def _auto_pick_mode(blocks: List[dict]) -> Literal["elements", "unstructured"]:
    """
    Heuristics:
      - If there are multiple sizable blocks spread vertically (e.g., forms, multi-paragraph docs),
        choose 'elements'.
      - If text density is high but mostly one contiguous block, choose 'unstructured'.
    """
    if not blocks:
        return "unstructured"

    num_blocks = len(blocks)

    heights = [b["bbox_abs"][3] for b in blocks]
    ys = [b["bbox_abs"][1] for b in blocks]
    xs = [b["bbox_abs"][0] for b in blocks]
    line_counts = [b["line_count"] for b in blocks]

    vertical_spread = np.ptp(ys) if len(ys) > 1 else 0
    horizontal_spread = np.ptp(xs) if len(xs) > 1 else 0
    avg_lines = float(np.mean(line_counts)) if line_counts else 0

    sizable_blocks = sum(1 for lc in line_counts if lc >= 2)

    if num_blocks >= 3 and sizable_blocks >= 2:
        return "elements"
    if sizable_blocks >= 2 and (vertical_spread > 60 or horizontal_spread > 60):
        return "elements"
    if num_blocks == 1 and avg_lines >= 5:
        return "unstructured"

    return "elements" if num_blocks > 1 else "unstructured"

class ImageOCRLoader:
    """
    LangChain-compatible image loader (OpenCV + Tesseract)
    mode:
      - "auto" (default): decide between "unstructured" and "elements" from layout
      - "unstructured": single Document with full-page text
      - "elements": multiple Documents with paragraph/block-level metadata
    """

    def __init__(
        self,
        path: str,
        mode: Mode = "auto",
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
        metadata: Optional[dict] = None,
        tesseract_cmd: Optional[str] = None,  # <-- NEW: allow explicit path injection
    ):
        self.path = path
        self.mode = mode
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.metadata = metadata or {}
        self._tesseract_cmd = tesseract_cmd

    def load(self) -> List[Document]:
        # Ensure tesseract is discoverable in THIS process (important for Windows services)
        resolved = _configure_tesseract(self._tesseract_cmd)

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Image not found at {self.path}")

        img = cv2.imread(self.path)
        if img is None:
            raise RuntimeError(f"Failed to read image at {self.path}")

        pre = _preprocess_for_ocr(img)
        pre = _deskew(pre)

        base_meta = {
            "source": self.path,
            "filetype": "image",
            "tesseract_cmd": resolved,  # helpful for debugging
            **self.metadata,
        }

        # Always compute blocks once (used by auto and elements)
        try:
            blocks = _extract_blocks(pre, lang=self.lang, psm=self.psm, oem=self.oem)
        except TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract not found while extracting blocks. "
                f"Resolved path tried: {resolved}. "
                "Ensure the executing user/service can access tesseract.exe."
            ) from e

        chosen_mode = self.mode
        if self.mode == "auto":
            chosen_mode = _auto_pick_mode(blocks)

        if chosen_mode == "unstructured":
            config = f"--oem {self.oem} --psm {self.psm}"
            try:
                text = pytesseract.image_to_string(pre, lang=self.lang, config=config).strip()
            except TesseractNotFoundError as e:
                raise RuntimeError(
                    "Tesseract not found while extracting text (unstructured). "
                    f"Resolved path tried: {resolved}."
                ) from e
            return [Document(page_content=text, metadata={**base_meta, "mode": "unstructured"})]

        # elements path
        docs: List[Document] = []
        for b in blocks:
            # Convert bbox tuples to JSON strings for ChromaDB compatibility
            # ChromaDB only supports: str, int, float, bool, SparseVector, None
            meta = {
                **base_meta,
                "bbox": json.dumps(list(b["bbox"])),  # Convert tuple to JSON string
                "bbox_abs": json.dumps(list(b["bbox_abs"])),  # Convert tuple to JSON string
                "avg_conf": b["avg_conf"],
                "line_count": b["line_count"],
                "block_num": b["block_num"],
                "par_num": b["par_num"],
                "image_width": b["image_size"][0],
                "image_height": b["image_size"][1],
                "ocr_engine": "tesseract",
                "mode": "elements",
            }
            docs.append(Document(page_content=b["text"], metadata=meta))
        return docs

def load_image_ocr(
    path: str,
    mode: Mode = "auto",
    lang: str = "eng",
    psm: int = 3,
    oem: int = 3,
    tesseract_cmd: Optional[str] = None,  # <-- NEW: bubble up
) -> List[Document]:
    """
    Auto-selects mode by default (user only provides image).
    Always returns List[Document].
    """
    try:
        loader = ImageOCRLoader(
            path,
            mode=mode,
            lang=lang,
            psm=psm,
            oem=oem,
            tesseract_cmd=tesseract_cmd,
        )
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("filetype", "image")
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {path} via OCR: {str(e)}") from e

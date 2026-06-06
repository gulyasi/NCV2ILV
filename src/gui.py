from __future__ import annotations

import base64
import json
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .composer import compose, format_report
from .ocr_pipeline import image_to_pdf, load_metadata, write_text_pdf


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Personal Handwriting Converter</title>
<style>
:root{font-family:Inter,system-ui,sans-serif;color:#19202a;background:#eef2f7}*{box-sizing:border-box}body{margin:0}.shell{max-width:1180px;margin:28px auto;padding:0 18px}h1{margin:0 0 18px}.modes{display:flex;gap:10px;margin-bottom:18px}button{border:0;border-radius:9px;padding:11px 17px;background:#dbe4ef;font-weight:700;cursor:pointer}button.primary,.active{background:#2458a6;color:white}.grid{display:grid;grid-template-columns:3fr 2fr;gap:18px}.card{background:white;border-radius:14px;padding:20px;box-shadow:0 5px 20px #1b2b4214}.form{display:none}.form.active{display:block}label{display:block;font-weight:650;margin:12px 0 5px}textarea,input,select{width:100%;border:1px solid #c7d0dc;border-radius:8px;padding:10px;background:white}textarea{min-height:210px;resize:vertical}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.check{display:flex;gap:8px;align-items:center;font-weight:500}.check input{width:auto}.action{width:100%;margin-top:18px}.preview{height:350px;display:flex;align-items:center;justify-content:center;background:#f6f8fb;border-radius:10px;overflow:hidden;text-align:center;color:#687588}.preview img{max-width:100%;max-height:100%}pre{white-space:pre-wrap;min-height:115px;background:#f6f8fb;border-radius:10px;padding:12px}.status{margin-top:16px;padding:10px 14px;background:#dbe4ef;border-radius:9px}.open{display:none;margin-top:10px;text-decoration:none;text-align:center}.open.show{display:block}@media(max-width:850px){.grid{grid-template-columns:1fr}.row{grid-template-columns:1fr}}
</style></head><body><main class="shell"><h1>Personal Handwriting Converter</h1>
<div class="modes"><button id="textBtn" class="active">Text to Handwriting</button><button id="ocrBtn">Handwriting to Text</button></div>
<div class="grid"><section class="card">
<form id="textForm" class="form active"><h2>Text to Handwriting</h2><label>Enter text</label><textarea id="sourceText">The quick brown fox jumps over the lazy dog.</textarea><div class="row"><div><label>Engine</label><select id="engine"><option value="hybrid">hybrid (font + learned writing)</option><option>script</option><option>font</option><option>glyph</option></select></div><div><label>Random seed</label><input id="seed" value="7"></div></div><label class="check"><input id="jitter" type="checkbox" checked> Character jitter</label><label>Glyph library</label><input id="library" value="data/glyph_library.json"><label>Output PNG or PDF</label><input id="renderOutput" value="outputs/gui/handwriting.png"><button id="renderButton" type="submit" class="primary action">Generate Handwriting</button></form>
<form id="ocrForm" class="form"><h2>Handwriting to Text</h2><label>Handwriting image</label><input id="image" type="file" accept="image/*" required><div class="row"><div><label>OCR method</label><select id="method"><option>auto</option><option>metadata</option><option>tesseract</option><option>qwen</option></select></div><div><label>Preprocessing</label><select id="preprocess"><option>none</option><option>grayscale</option><option>otsu</option><option>adaptive</option><option>denoise-deskew</option></select></div></div><label class="check"><input id="ensemble" type="checkbox"> Try preprocessing ensemble (Tesseract)</label><label>Metadata CSV</label><input id="metadata" value="data/metadata.csv"><label>Tesseract language</label><input id="language" value="eng"><label>Manual transcription (optional fallback)</label><textarea id="manualText" placeholder="Enter the text here when no OCR engine is installed."></textarea><label>Output PDF</label><input id="ocrOutput" value="outputs/gui/transcription.pdf"><button id="ocrButton" type="submit" class="primary action">Recognize Handwriting</button></form>
</section><aside class="card"><h2>Preview and Result</h2><div id="preview" class="preview">Your image preview will appear here.</div><h3>Recognized text / render report</h3><pre id="result"></pre><a id="open" class="open primary" target="_blank">Open Output</a></aside></div><div id="status" class="status">Ready</div></main>
<script>
const byId = (id) => document.getElementById(id);

function setMode(which) {
  byId("textForm").classList.toggle("active", which === "text");
  byId("ocrForm").classList.toggle("active", which === "ocr");
  byId("textBtn").classList.toggle("active", which === "text");
  byId("ocrBtn").classList.toggle("active", which === "ocr");
}

function setBusy(button, label) {
  button.disabled = true;
  button.dataset.label = button.textContent;
  button.textContent = label;
  byId("status").textContent = label;
  byId("result").textContent = "Working...";
  byId("open").classList.remove("show");
}

function setReady(button) {
  button.disabled = false;
  button.textContent = button.dataset.label;
}

function showResult(data) {
  if (!data.ok) throw new Error(data.error || "Conversion failed");
  byId("status").textContent = data.status;
  byId("result").textContent = data.text || "";
  if (data.preview) byId("preview").innerHTML = '<img src="' + data.preview + '&t=' + Date.now() + '">';
  if (data.file) {
    byId("open").href = data.file;
    byId("open").classList.add("show");
  }
}

async function postJSON(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed (" + response.status + ")");
  return data;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the selected image"));
    reader.readAsDataURL(file);
  });
}

fetch("/api/status").then((response) => response.json()).then((data) => {
  if (!data.tesseract) byId("status").textContent = "Ready. New images use GPU-backed Qwen OCR; the first run may take longer while the model loads.";
});

byId("textBtn").addEventListener("click", () => setMode("text"));
byId("ocrBtn").addEventListener("click", () => setMode("ocr"));
byId("image").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) byId("preview").innerHTML = '<img src="' + URL.createObjectURL(file) + '">';
});

byId("textForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("renderButton");
  setBusy(button, "Generating handwriting...");
  try {
    const data = await postJSON("/api/render", {
      text: byId("sourceText").value,
      engine: byId("engine").value,
      seed: byId("seed").value,
      jitter: byId("jitter").checked,
      library: byId("library").value,
      output: byId("renderOutput").value,
    });
    showResult(data);
  } catch (error) {
    byId("status").textContent = "Failed: " + error.message;
    byId("result").textContent = error.message;
  } finally {
    setReady(button);
  }
});

byId("ocrForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("ocrButton");
  const file = byId("image").files[0];
  if (!file) {
    byId("status").textContent = "Choose a handwriting image first.";
    return;
  }
  setBusy(button, "Recognizing handwriting...");
  try {
    const data = await postJSON("/api/ocr", {
      name: file.name,
      image: await readFile(file),
      method: byId("method").value,
      preprocess: byId("preprocess").value,
      ensemble: byId("ensemble").checked,
      metadata: byId("metadata").value,
      language: byId("language").value,
      output: byId("ocrOutput").value,
      manual_text: byId("manualText").value,
    });
    showResult(data);
  } catch (error) {
    byId("status").textContent = "Failed: " + error.message;
    byId("result").textContent = error.message;
  } finally {
    setReady(button);
  }
});
</script></body></html>'''


class GUIHandler(BaseHTTPRequestHandler):
    project_root = Path.cwd().resolve()

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    @classmethod
    def _path(cls, value: str) -> Path:
        path = (cls.project_root / value).resolve()
        if path != cls.project_root and cls.project_root not in path.parents:
            raise ValueError("Paths must stay inside the project directory")
        return path

    @staticmethod
    def _url(path: Path) -> str:
        return "/file?path=" + quote(str(path.resolve().relative_to(GUIHandler.project_root)))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            metadata = load_metadata()
            self._json({"ok": True, "tesseract": shutil.which("tesseract") is not None, "metadata_images": len(metadata)})
            return
        if parsed.path == "/file":
            try:
                path = self._path(parse_qs(parsed.query)["path"][0])
                types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf"}
                self._send(200, path.read_bytes(), types.get(path.suffix.lower(), "application/octet-stream"))
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 404)
            return
        self._json({"ok": False, "error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            if self.path == "/api/render":
                self._render(data)
            elif self.path == "/api/ocr":
                self._ocr(data)
            else:
                self._json({"ok": False, "error": "Not found"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)

    def _render(self, data: dict) -> None:
        output = self._path(data.get("output") or "outputs/gui/handwriting.png")
        seed_text = str(data.get("seed", "")).strip()
        report = compose(
            str(data.get("text", "")), output_name=str(output), library_path=str(self._path(data.get("library") or "data/synthetic_glyph_library.json")),
            seed=int(seed_text) if seed_text else None, jitter=bool(data.get("jitter", True)), engine=data.get("engine", "script"),
        )
        is_image = output.suffix.lower() in {".png", ".jpg", ".jpeg"}
        self._json({"ok": True, "status": f"Handwriting created: {output.relative_to(self.project_root)}", "text": format_report(report), "file": self._url(output), "preview": self._url(output) if is_image else None})

    def _ocr(self, data: dict) -> None:
        name = Path(data.get("name") or "upload.png").name
        upload = self._path(f"outputs/gui/uploads/{name}")
        upload.parent.mkdir(parents=True, exist_ok=True)
        encoded = str(data["image"]).split(",", 1)[-1]
        upload.write_bytes(base64.b64decode(encoded))
        output = self._path(data.get("output") or "outputs/gui/transcription.pdf")
        manual_text = str(data.get("manual_text", "")).strip()
        method = data.get("method", "auto")
        metadata_path = str(self._path(data.get("metadata") or "data/metadata.csv"))
        if manual_text:
            write_text_pdf(manual_text, str(output), source_image=str(upload), method="manual transcription")
            self._json({"ok": True, "status": f"Manual transcription saved: {output.relative_to(self.project_root)}", "text": manual_text, "file": self._url(output), "preview": self._url(upload)})
            return
        if method == "metadata" and name not in load_metadata(metadata_path):
            raise RuntimeError("No metadata label exists for this image. Choose Auto or Qwen to transcribe a new image.")
        result = image_to_pdf(str(upload), output_path=str(output), method=method, metadata_path=metadata_path, tesseract_lang=data.get("language") or "eng", preprocess=data.get("preprocess", "none"), ensemble_preprocess=bool(data.get("ensemble", False)))
        self._json({"ok": True, "status": f"Recognized with {result.method}: {output.relative_to(self.project_root)}", "text": result.text, "file": self._url(output), "preview": self._url(upload)})


def run_gui(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    try:
        server = ThreadingHTTPServer((host, port), GUIHandler)
    except OSError as exc:
        if exc.errno != 98:
            raise
        server = ThreadingHTTPServer((host, 0), GUIHandler)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"
    print(f"Handwriting GUI running at {url}")
    print("Press Ctrl+C to stop it.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

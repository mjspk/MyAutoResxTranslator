"""
AutoResxTranslator — protected words + RTL-safe restoration (full app)

Key fixes:
 - Protected words replaced with self-closing XML tags <kw id="i"/> prior to translation
 - Strings containing placeholders or protected words use DeepL tag_handling="xml"
 - On restore, protected words are inserted back and wrapped with LRM (U+200E) if target is RTL
 - Keeps glossary, placeholders, preview dialog, thread-safety, chunking, retries, settings persistence
"""

import os
import re
import json
import csv
import time
import threading
from pathlib import Path
from xml.etree import ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText

try:
    import deepl
    from deepl import DeepLException
except Exception as e:
    raise RuntimeError("Install deepl package: pip install deepl") from e

# ---------- settings ----------
SETTINGS_PATH = Path.home() / ".auto_resx_translator_settings.json"
DEFAULT_SETTINGS = {
    "deepl_api_key": "",
    "chunk_size": 50,
    "max_chars_per_call": 30000,
    "filename_style": 0,
    "last_glossary": "",
    "use_tag_handling": False,
    "persist_api_key": False,
    "protected_words": ""
}

def load_settings():
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s = json.load(f)
            for k, v in DEFAULT_SETTINGS.items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except Exception:
        pass

# ---------- placeholder patterns ----------
PLACEHOLDER_PATTERNS = [
    re.compile(r"\{[^}]+\}"),    # {0}, {Name}, {0:N2}, ICU fragments
    re.compile(r"%\w"),          # %s, %d
]

TOKEN_PREFIX = "__PH__"
KW_TAG = "kw"   # xml tag name for protected words
GLOSS_TAG = "gl" # xml tag name for glossary placeholders

# RTL language code prefixes (approx)
RTL_PREFIXES = ("ar", "he", "fa", "ur")

# ---------- resx parsing/writing ----------
def parse_resx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    items = []
    for data in root.findall("data"):
        val = data.find("value")
        if val is not None:
            name = data.get("name")
            items.append((data, val, name))
    return tree, root, items

def write_resx(tree, out_path):
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

# ---------- glossary loaders ----------
def load_glossary_from_csv(path):
    mapping = {}
    with open(path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row: continue
            if len(row) < 2: continue
            src = row[0].strip()
            tgt = row[1].strip()
            if src:
                mapping[src] = tgt
    return mapping

def load_glossary(path):
    p = Path(path)
    if not p.exists():
        return {}
    if p.suffix.lower() == ".csv":
        return load_glossary_from_csv(p)
    elif p.suffix.lower() == ".json":
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    return {}

# ---------- DeepL helpers ----------
DEFAULT_CHUNK_SIZE = 50
DEFAULT_MAX_CHARS = 30000
MAX_RETRIES = 5
BASE_BACKOFF = 1.0

def make_translator(api_key):
    return deepl.Translator(api_key)

def get_target_langs(translator):
    langs = translator.get_target_languages()
    return [{"name": getattr(l,"name",str(l)), "code": getattr(l,"code",str(l))} for l in langs]

def translate_with_retry(translator, texts, target_lang, use_tag_handling=False, max_chars=DEFAULT_MAX_CHARS, chunk_size=DEFAULT_CHUNK_SIZE):
    if not texts:
        return []
    # chunk by count and char budget
    chunks = []
    cur = []
    cur_chars = 0
    for i, t in enumerate(texts):
        L = len(t or "")
        if cur and (len(cur) >= chunk_size or (cur_chars + L) > max_chars):
            chunks.append(cur)
            cur = []
            cur_chars = 0
        cur.append(i)
        cur_chars += L
    if cur:
        chunks.append(cur)
    out = [None]*len(texts)
    for chunk in chunks:
        inputs = [texts[i] for i in chunk]
        attempt = 0
        while True:
            attempt += 1
            try:
                if use_tag_handling:
                    res = translator.translate_text(inputs, target_lang=target_lang, tag_handling="xml")
                else:
                    res = translator.translate_text(inputs, target_lang=target_lang)
                if not isinstance(res, (list, tuple)):
                    res_list = [res]
                else:
                    res_list = list(res)
                for local_i, trans_obj in enumerate(res_list):
                    out[chunk[local_i]] = trans_obj.text
                break
            except DeepLException:
                if attempt > MAX_RETRIES:
                    raise
                time.sleep(BASE_BACKOFF * (2 ** (attempt-1)))
            except Exception:
                if attempt > MAX_RETRIES:
                    raise
                time.sleep(BASE_BACKOFF * (2 ** (attempt-1)))
    return out

# ---------- Preview dialog ----------
class PreviewDialog(simpledialog.Dialog):
    def __init__(self, parent, keys, originals, translations, title="Preview translations"):
        self.keys = keys
        self.originals = originals
        self.translations = translations
        super().__init__(parent, title)

    def body(self, master):
        master.columnconfigure(1, weight=1)
        master.rowconfigure(0, weight=1)
        ttk.Label(master, text="Preview translations (edit any cell then press Save)").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,6))
        self.container = ttk.Frame(master)
        self.container.grid(row=1, column=0, sticky="nsew")
        canvas = tk.Canvas(self.container, height=360)
        scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")
        h = ttk.Frame(self.inner)
        h.pack(fill="x", pady=(0,2))
        ttk.Label(h, text="Key", width=30).grid(row=0, column=0, sticky="w")
        ttk.Label(h, text="Original", width=50).grid(row=0, column=1, sticky="w")
        ttk.Label(h, text="Translation", width=50).grid(row=0, column=2, sticky="w")
        self.edit_entries = []
        for i, key in enumerate(self.keys):
            row = ttk.Frame(self.inner)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=str(key)[:40], width=30).grid(row=0, column=0, sticky="w")
            orig = self.originals[i]
            trans = self.translations[i] or ""
            ttk.Label(row, text=str(orig)[:80], width=50).grid(row=0, column=1, sticky="w")
            ent = ttk.Entry(row, width=60)
            ent.insert(0, trans)
            ent.grid(row=0, column=2, sticky="w")
            self.edit_entries.append(ent)

    def apply(self):
        self.result = [e.get() for e in self.edit_entries]

# ---------- Locale detection utility ----------
def detect_existing_culture_codes(base_path):
    p = Path(base_path)
    directory = p.parent
    base_name = p.name
    if base_name.lower().endswith(".resx"):
        stem = base_name[:-5]
    else:
        stem = base_name
    found = set()
    for f in directory.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if not name.lower().endswith(".resx"):
            continue
        if name == base_name:
            continue
        if name.startswith(stem + ".") and len(name) > len(stem) + 5:
            code_part = name[len(stem)+1:-5]
            if code_part:
                found.add(code_part.lower())
    return found

# ---------- Main App ----------
class AutoResxTranslatorApp:
    def __init__(self, root):
        self.root = root
        root.title("AutoResxTranslator — protected words & RTL fix")
        root.geometry("980x760")
        self.settings = load_settings()

        # internal
        self.loaded_langs = []   # list of tuples (display, code)
        self.detected_codes = set()
        self.glossary = {}
        self._stop_requested = False

        # Top: file + key + protected words
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Base .resx file:").grid(row=0, column=0, sticky="w")
        self.file_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.file_var, width=80).grid(row=0, column=1, padx=6)
        ttk.Button(top, text="Browse...", command=self.browse_file).grid(row=0, column=2)

        ttk.Label(top, text="DeepL API Key:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.key_var = tk.StringVar(value=self.settings.get("deepl_api_key",""))
        self.key_entry = ttk.Entry(top, textvariable=self.key_var, width=80, show="*" if not self.settings.get("persist_api_key") else "")
        self.key_entry.grid(row=1, column=1, padx=6, pady=(6,0))
        ttk.Button(top, text="Show/Hide", command=self.toggle_key).grid(row=1, column=2)
        self.persist_key_var = tk.BooleanVar(value=self.settings.get("persist_api_key", False))
        ttk.Checkbutton(top, text="Persist API key (local)", variable=self.persist_key_var, command=self.toggle_persist_api_key).grid(row=2, column=1, sticky="w", pady=(6,0))

        ttk.Label(top, text="Protected words (comma-separated):").grid(row=3, column=0, sticky="w", pady=(6,0))
        self.protected_var = tk.StringVar(value=self.settings.get("protected_words",""))
        ttk.Entry(top, textvariable=self.protected_var, width=80).grid(row=3, column=1, padx=6, pady=(6,0))
        ttk.Label(top, text="(Words here will stay identical in translations)").grid(row=4, column=1, sticky="w")

        # Middle: languages + controls
        mid = ttk.Frame(root, padding=8)
        mid.pack(fill="both", expand=False)
        left = ttk.Frame(mid)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Target languages (DeepL):").pack(anchor="w")
        self.lang_listbox = tk.Listbox(left, selectmode="multiple", height=14, width=44, exportselection=False)
        self.lang_listbox.pack(side="left", fill="y")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.lang_listbox.yview)
        scrollbar.pack(side="left", fill="y")
        self.lang_listbox.config(yscrollcommand=scrollbar.set)

        right = ttk.Frame(mid, padding=(12,0))
        right.pack(side="left", fill="y")
        ttk.Button(right, text="Load languages", command=self.load_languages).pack(pady=4)
        ttk.Button(right, text="Select all", command=lambda: self.lang_listbox.select_set(0, tk.END)).pack(pady=4)
        ttk.Button(right, text="Clear", command=lambda: self.lang_listbox.select_clear(0, tk.END)).pack(pady=4)

        # Options
        opts = ttk.LabelFrame(root, text="Options", padding=8)
        opts.pack(fill="x", padx=8, pady=6)
        ttk.Label(opts, text="Chunk size (#strings per call):").grid(row=0, column=0, sticky="w")
        self.chunk_var = tk.IntVar(value=self.settings.get("chunk_size", DEFAULT_CHUNK_SIZE))
        ttk.Entry(opts, textvariable=self.chunk_var, width=6).grid(row=0, column=1, sticky="w", padx=(6,12))
        ttk.Label(opts, text="Max chars per call:").grid(row=0, column=2, sticky="w")
        self.chars_var = tk.IntVar(value=self.settings.get("max_chars_per_call", DEFAULT_MAX_CHARS))
        ttk.Entry(opts, textvariable=self.chars_var, width=8).grid(row=0, column=3, sticky="w", padx=6)
        self.filename_style_var = tk.IntVar(value=self.settings.get("filename_style", 0))
        ttk.Radiobutton(opts, text="short (Strings.es.resx)", variable=self.filename_style_var, value=0).grid(row=0, column=4, padx=12, sticky="w")
        ttk.Radiobutton(opts, text="full (Strings.es-ES.resx)", variable=self.filename_style_var, value=1).grid(row=0, column=5, sticky="w")
        self.use_tag_handling_manual = tk.BooleanVar(value=self.settings.get("use_tag_handling", False))
        ttk.Checkbutton(opts, text="Enable DeepL XML tag handling (manual)", variable=self.use_tag_handling_manual).grid(row=0, column=6, padx=12, sticky="w")

        # Glossary controls
        glf = ttk.Frame(root, padding=8)
        glf.pack(fill="x")
        ttk.Label(glf, text="Glossary (CSV or JSON):").grid(row=0, column=0, sticky="w")
        self.glossary_var = tk.StringVar(value=self.settings.get("last_glossary",""))
        ttk.Entry(glf, textvariable=self.glossary_var, width=70).grid(row=0, column=1, padx=6)
        ttk.Button(glf, text="Load...", command=self.load_glossary_dialog).grid(row=0, column=2)
        ttk.Button(glf, text="Clear", command=self.clear_glossary).grid(row=0, column=3, padx=6)

        # Actions
        action = ttk.Frame(root, padding=8)
        action.pack(fill="x")
        self.start_btn = ttk.Button(action, text="Start translation (with preview)", command=self.start_translation)
        self.start_btn.pack(side="left")
        ttk.Button(action, text="Stop", command=self.request_stop).pack(side="left", padx=6)
        self.status_label = ttk.Label(action, text="Idle")
        self.status_label.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(root, maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0,6))

        # Log
        log_frame = ttk.Frame(root, padding=8)
        log_frame.pack(fill="both", expand=True)
        ttk.Label(log_frame, text="Log:").pack(anchor="w")
        self.log_text = ScrolledText(log_frame, height=16)
        self.log_text.pack(fill="both", expand=True)

        # populate fallback languages to avoid empty UI
        fallback = {
            "Spanish":"ES","Arabic":"AR","French":"FR","German":"DE","Italian":"IT",
            "Portuguese (PT)":"PT-PT","Portuguese (BR)":"PT-BR","Dutch":"NL","Russian":"RU",
            "Japanese":"JA","Chinese":"ZH","Polish":"PL","Turkish":"TR","Korean":"KO"
        }
        for n,c in fallback.items():
            self.lang_listbox.insert("end", f"{n} ({c})")
            self.loaded_langs.append((f"{n} ({c})", c))

    # ---------- thread-safe GUI helpers ----------
    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        try:
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
        except Exception:
            pass

    def safe_log(self, msg):
        try:
            self.root.after(0, lambda: self._append_log(msg))
        except Exception:
            pass

    def _set_status(self, txt):
        try:
            self.status_label.config(text=txt)
        except Exception:
            pass

    def safe_status(self, txt):
        try:
            self.root.after(0, lambda: self._set_status(txt))
        except Exception:
            pass

    def _set_progress(self, value):
        try:
            self.progress['value'] = value
        except Exception:
            pass

    def safe_progress(self, value):
        try:
            self.root.after(0, lambda: self._set_progress(value))
        except Exception:
            pass

    def safe_show_info(self, title, message):
        try:
            self.root.after(0, lambda: messagebox.showinfo(title, message))
        except Exception:
            pass

    def safe_show_error(self, title, message):
        try:
            self.root.after(0, lambda: messagebox.showerror(title, message))
        except Exception:
            pass

    # ---------- preview modal helper ----------
    def show_preview_modal(self, keys, originals, translations, title="Preview translations"):
        result_container = {}
        done = threading.Event()
        def _show():
            try:
                dlg = PreviewDialog(self.root, keys, originals, translations, title=title)
                result_container['result'] = getattr(dlg, "result", None)
            except Exception as e:
                result_container['exception'] = e
            finally:
                done.set()
        self.root.after(0, _show)
        done.wait()
        if 'exception' in result_container:
            raise result_container['exception']
        return result_container.get('result')

    # ---------- UI helpers ----------
    def browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("ResX files","*.resx")])
        if p:
            self.file_var.set(p)
            detected = detect_existing_culture_codes(p)
            self.detected_codes = detected
            if detected:
                self.safe_log(f"Detected existing localized files: {', '.join(sorted(detected))}")
            else:
                self.safe_log("No existing localized files detected.")
            self._select_detected_codes_in_listbox(detected)

    def _select_detected_codes_in_listbox(self, detected_codes):
        if not detected_codes:
            return
        detected_normal = {c.lower() for c in detected_codes}
        code_to_index = {code.lower(): idx for idx, (_, code) in enumerate(self.loaded_langs)}
        try:
            self.lang_listbox.select_clear(0, tk.END)
        except Exception:
            pass
        for code in detected_normal:
            if code in code_to_index:
                idx = code_to_index[code]
                try:
                    self.lang_listbox.select_set(idx)
                except Exception:
                    pass
            else:
                display = f"Existing ({code})"
                try:
                    self.lang_listbox.insert("end", display)
                    self.loaded_langs.append((display, code))
                    self.lang_listbox.select_set(self.lang_listbox.size()-1)
                except Exception:
                    pass

    def toggle_key(self):
        if self.key_entry.cget("show") == "":
            self.key_entry.config(show="*")
        else:
            self.key_entry.config(show="")

    def toggle_persist_api_key(self):
        self.settings["persist_api_key"] = self.persist_key_var.get()
        if not self.persist_key_var.get():
            self.settings["deepl_api_key"] = ""
            save_settings(self.settings)

    # ---------- glossary ----------
    def load_glossary_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("CSV or JSON","*.csv;*.json")])
        if not p:
            return
        try:
            g = load_glossary(p)
            self.glossary = g
            self.glossary_var.set(p)
            self.settings["last_glossary"] = p
            save_settings(self.settings)
            self.safe_log(f"Loaded glossary ({len(g)} entries) from {p}")
        except Exception as e:
            self.safe_show_error("Glossary error", f"Failed to load glossary: {e}")

    def clear_glossary(self):
        self.glossary = {}
        self.glossary_var.set("")
        self.settings["last_glossary"] = ""
        save_settings(self.settings)
        self.safe_log("Cleared glossary")

    # ---------- languages ----------
    def load_languages(self):
        key = self.key_var.get().strip() or os.getenv("DEEPL_API_KEY","")
        if not key:
            self.safe_show_info("No API key", "Provide DeepL API key to load languages.")
            return
        try:
            trans = make_translator(key)
            langs = get_target_langs(trans)
            prev_detected = set(self.detected_codes) if hasattr(self, "detected_codes") else set()
            self.lang_listbox.delete(0, tk.END)
            self.loaded_langs = []
            for l in sorted(langs, key=lambda x: x["name"]):
                disp = f"{l['name']} ({l['code']})"
                self.lang_listbox.insert("end", disp)
                self.loaded_langs.append((disp, l["code"]))
            self.safe_log(f"Loaded {len(self.loaded_langs)} languages from DeepL")
            if self.persist_key_var.get():
                self.settings["deepl_api_key"] = key
                self.settings["persist_api_key"] = True
                save_settings(self.settings)
            if prev_detected:
                self._select_detected_codes_in_listbox(prev_detected)
        except Exception as e:
            self.safe_log(f"Failed to load languages from DeepL: {e}")
            self.safe_show_info("Load failed", f"Could not fetch languages from DeepL: {e}. Using current list.")

    # ---------- translation control ----------
    def request_stop(self):
        self._stop_requested = True
        self.safe_log("Stop requested")

    def start_translation(self):
        path = self.file_var.get().strip()
        if not path or not Path(path).exists():
            self.safe_show_error("Missing file", "Choose a valid .resx file first.")
            return
        selected = self.lang_listbox.curselection()
        if not selected:
            self.safe_show_error("No targets", "Select at least one target language.")
            return
        key = self.key_var.get().strip() or os.getenv("DEEPL_API_KEY","")
        if not key:
            self.safe_show_error("No API key", "Provide DeepL API key or set DEEPL_API_KEY.")
            return

        codes = [ self.loaded_langs[i][1] for i in selected ] if self.loaded_langs else []
        if not codes:
            codes = []
            for i in selected:
                text = self.lang_listbox.get(i)
                m = re.search(r"\((.*?)\)", text)
                codes.append(m.group(1) if m else text)

        # save settings
        self.settings["chunk_size"] = int(self.chunk_var.get())
        self.settings["max_chars_per_call"] = int(self.chars_var.get())
        self.settings["filename_style"] = int(self.filename_style_var.get())
        self.settings["use_tag_handling"] = bool(self.use_tag_handling_manual.get())
        self.settings["protected_words"] = self.protected_var.get().strip()
        if self.persist_key_var.get():
            self.settings["deepl_api_key"] = key
        self.settings["persist_api_key"] = bool(self.persist_key_var.get())
        save_settings(self.settings)

        self.start_btn.config(state="disabled")
        self._stop_requested = False
        thr = threading.Thread(target=self._run_translation, args=(path, key, codes), daemon=True)
        thr.start()

    def _is_rtl_lang(self, target_code):
        if not target_code:
            return False
        lower = target_code.lower()
        return any(lower.startswith(pref) for pref in RTL_PREFIXES)

    def _run_translation(self, base_path, api_key, target_codes):
        try:
            tree, root, items = parse_resx(base_path)
            total = len(items)
            if total == 0:
                self.safe_show_info("Nothing to translate", "No <data><value> items found.")
                self._finish()
                return

            originals = []
            token_maps = []   # per-string metadata
            names = []

            protected_text = self.protected_var.get().strip()
            protected_words = [w.strip() for w in protected_text.split(",") if w.strip()]

            # create glossary token mapping
            glossary_tokens = {}
            if self.glossary:
                for i, src in enumerate(sorted(self.glossary.keys(), key=lambda x: -len(x))):
                    tag = f"<{GLOSS_TAG} id=\"{i}\"/>"
                    glossary_tokens[tag] = (src, self.glossary[src])

            # Build pre-processed strings:
            for data_el, val_el, name in items:
                orig = val_el.text or ""
                working = orig
                pre_map = {}  # token->original substring mapping (for non-tag tokens, rarely used)

                # Protected words: replace with XML self-closing tag <kw id="i"/>
                kw_map = {}
                if protected_words:
                    for i, w in enumerate(sorted(protected_words, key=lambda x: -len(x))):
                        if not w:
                            continue
                        tag = f"<{KW_TAG} id=\"{i}\"/>"
                        if w in working:
                            working = working.replace(w, tag)
                            kw_map[tag] = w

                # Glossary source replacements -> self-closing tags
                if glossary_tokens:
                    for tag, (src, tgt) in glossary_tokens.items():
                        if src and src in working:
                            working = working.replace(src, tag)
                            # mark in pre_map if needed (we'll map tag->tgt on restore)
                            pre_map[tag] = src

                # placeholders detection
                has_placeholders = any(p.search(working) for p in PLACEHOLDER_PATTERNS)
                has_kw_or_gloss = bool(kw_map) or bool(glossary_tokens and pre_map)
                # decide xml tag handling if placeholders or protected words or forced
                use_tag_handling_for_string = bool(self.use_tag_handling_manual.get()) or has_placeholders or has_kw_or_gloss

                if use_tag_handling_for_string:
                    # convert placeholders to <ph id="n"/> tags
                    ph_map = {}
                    idx = 0
                    def _tag_repl(m):
                        nonlocal idx
                        ph = m.group(0)
                        tag = f"<ph id=\"{idx}\"/>"
                        ph_map[tag] = ph
                        idx += 1
                        return tag
                    new_working = working
                    for pat in PLACEHOLDER_PATTERNS:
                        new_working = pat.sub(_tag_repl, new_working)
                    working = new_working
                    token_maps.append({
                        "type": "xml",
                        "ph_map": dict(ph_map),
                        "kw_map": dict(kw_map),
                        "pre_map": dict(pre_map)
                    })
                else:
                    # fallback tokenization (should be rare for protected words because we prefer XML tags)
                    tok_map = {}
                    idx = 0
                    def _tok_repl(m):
                        nonlocal idx
                        ph = m.group(0)
                        token = f"{TOKEN_PREFIX}{idx}__"
                        tok_map[token] = ph
                        idx += 1
                        return token
                    new_working = working
                    for pat in PLACEHOLDER_PATTERNS:
                        new_working = pat.sub(_tok_repl, new_working)
                    working = new_working
                    token_maps.append({
                        "type": "tokens",
                        "tok_map": dict(tok_map),
                        "kw_map": dict(kw_map),
                        "pre_map": dict(pre_map)
                    })

                originals.append(working)
                names.append(name)

            translator = make_translator(api_key)

            # translate per target
            for idx_lang, target in enumerate(target_codes, start=1):
                if self._stop_requested:
                    break
                self.safe_status(f"Translating {target} ({idx_lang}/{len(target_codes)})")
                self.safe_log(f"Translating to {target}")
                chunk_size = int(self.chunk_var.get())
                max_chars = int(self.chars_var.get())

                # split by xml vs tokens
                indices_xml = [i for i, m in enumerate(token_maps) if m["type"] == "xml"]
                indices_tok = [i for i, m in enumerate(token_maps) if m["type"] == "tokens"]
                translated = [None] * len(originals)

                # xml-tagged texts
                if indices_xml:
                    texts_xml = [originals[i] for i in indices_xml]
                    trans_xml = translate_with_retry(translator, texts_xml, target, use_tag_handling=True, max_chars=max_chars, chunk_size=chunk_size)
                    for local_i, t in enumerate(trans_xml):
                        translated[indices_xml[local_i]] = t

                # tokenized texts
                if indices_tok:
                    texts_tok = [originals[i] for i in indices_tok]
                    trans_tok = translate_with_retry(translator, texts_tok, target, use_tag_handling=False, max_chars=max_chars, chunk_size=chunk_size)
                    for local_i, t in enumerate(trans_tok):
                        translated[indices_tok[local_i]] = t

                # restore per-string
                restored = []
                rtl = self._is_rtl_lang(target)
                LRM = "\u200E"  # left-to-right mark
                for i, trans_text in enumerate(translated):
                    t = trans_text or ""
                    meta = token_maps[i]
                    if meta["type"] == "xml":
                        # restore placeholder tags <ph id="n"/> -> original placeholder text
                        for tag, ph in meta["ph_map"].items():
                            # simple replace; DeepL preserves tags but may insert spaces — replace occurrences of tag
                            t = t.replace(tag, ph)
                        # restore glossary tags: tags in pre_map or glossary_tokens
                        # first restore glossary tags to their TARGET form (glossary token -> target string)
                        if self.glossary:
                            # glossary_tokens was tag->(src,tgt)
                            for tag, (src, tgt) in glossary_tokens.items():
                                if tag in t:
                                    t = t.replace(tag, tgt)
                        # restore protected word tags -> original word, and wrap with LRM for RTL target
                        for tag, orig_word in meta["kw_map"].items():
                            if tag in t:
                                insert = orig_word
                                if rtl:
                                    # wrap with LRM on both sides to keep the enclosed LTR text correct inside Arabic
                                    insert = LRM + orig_word + LRM
                                t = t.replace(tag, insert)
                        # finally restore any other pre_map tokens (rare)
                        for tag, orig_sub in meta["pre_map"].items():
                            if tag in t:
                                t = t.replace(tag, orig_sub)
                    else:
                        # tokens approach
                        for tok, ph in meta["tok_map"].items():
                            if tok in t:
                                t = t.replace(tok, ph)
                        # restore glossary tokens if any (they were preserved as tags in originals only if created)
                        if self.glossary:
                            for tag, (src, tgt) in glossary_tokens.items():
                                if tag in t:
                                    t = t.replace(tag, tgt)
                        # protected words restored via meta["kw_map"]
                        for token, orig_word in meta["kw_map"].items():
                            if token in t:
                                insert = orig_word
                                if rtl:
                                    insert = "\u200E" + orig_word + "\u200E"
                                t = t.replace(token, insert)
                        for token, orig_sub in meta["pre_map"].items():
                            if token in t:
                                t = t.replace(token, orig_sub)

                    restored.append(t)

                # preview and save
                originals_for_preview = [ (items[i][1].text or "") for i in range(len(items)) ]
                try:
                    preview_result = self.show_preview_modal(names, originals_for_preview, restored, title=f"Preview - {target}")
                except Exception as e:
                    self.safe_log(f"Preview error: {e}")
                    preview_result = None

                if preview_result is None:
                    self.safe_log(f"User cancelled preview for {target}, skipping save.")
                    continue
                edited = preview_result

                tree_out, root_out, items_out = parse_resx(base_path)
                for i, (_, val_el, _) in enumerate(items_out):
                    val_el.text = edited[i]

                base_name = Path(base_path).name
                stem = base_name[:-5] if base_name.lower().endswith(".resx") else base_name
                if self.filename_style_var.get() == 0:
                    short = target.split("-")[0].lower()
                    out_name = f"{stem}.{short}.resx"
                else:
                    out_name = f"{stem}.{target.lower()}.resx"
                out_path = Path(base_path).parent / out_name
                try:
                    write_resx(tree_out, out_path)
                    self.safe_log(f"Wrote {out_path}")
                except Exception as e:
                    self.safe_log(f"Failed to write {out_path}: {e}")
                    self.safe_show_error("Write error", f"Failed to write {out_path}: {e}")

                percent = (idx_lang / len(target_codes)) * 100
                self.safe_progress(percent)
                time.sleep(0.2)

            if not self._stop_requested:
                self.safe_status("Done")
                self.safe_show_info("Done", "Translations completed.")
            else:
                self.safe_status("Stopped")
                self.safe_log("Stopped by user.")
        except Exception as e:
            self.safe_log(f"Fatal error: {e}")
            self.safe_show_error("Error", f"Fatal error: {e}")
        finally:
            self._finish()

    def _finish(self):
        try:
            self.start_btn.config(state="normal")
        except Exception:
            pass
        self._stop_requested = False
        self.safe_progress(0)
        self.safe_status("Idle")
        # persist settings
        self.settings["chunk_size"] = int(self.chunk_var.get())
        self.settings["max_chars_per_call"] = int(self.chars_var.get())
        self.settings["filename_style"] = int(self.filename_style_var.get())
        self.settings["use_tag_handling"] = bool(self.use_tag_handling_manual.get())
        self.settings["protected_words"] = self.protected_var.get().strip()
        if self.persist_key_var.get():
            self.settings["deepl_api_key"] = self.key_var.get().strip()
        self.settings["persist_api_key"] = bool(self.persist_key_var.get())
        self.settings["last_glossary"] = self.glossary_var.get().strip()
        save_settings(self.settings)

    # ---------- helpers ----------
    def _is_rtl_lang(self, target_code):
        if not target_code:
            return False
        lower = target_code.lower()
        return any(lower.startswith(pref) for pref in RTL_PREFIXES)

    # UI & thread-safe helpers (same as before)
    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        try:
            self.log_text.insert("end", f"[{ts}] {msg}\n")
            self.log_text.see("end")
        except Exception:
            pass

    def safe_log(self, msg):
        try:
            self.root.after(0, lambda: self._append_log(msg))
        except Exception:
            pass

    def _set_status(self, txt):
        try:
            self.status_label.config(text=txt)
        except Exception:
            pass

    def safe_status(self, txt):
        try:
            self.root.after(0, lambda: self._set_status(txt))
        except Exception:
            pass

    def _set_progress(self, value):
        try:
            self.progress['value'] = value
        except Exception:
            pass

    def safe_progress(self, value):
        try:
            self.root.after(0, lambda: self._set_progress(value))
        except Exception:
            pass

    def safe_show_info(self, title, message):
        try:
            self.root.after(0, lambda: messagebox.showinfo(title, message))
        except Exception:
            pass

    def safe_show_error(self, title, message):
        try:
            self.root.after(0, lambda: messagebox.showerror(title, message))
        except Exception:
            pass

    def show_preview_modal(self, keys, originals, translations, title="Preview translations"):
        result_container = {}
        done = threading.Event()
        def _show():
            try:
                dlg = PreviewDialog(self.root, keys, originals, translations, title=title)
                result_container['result'] = getattr(dlg, "result", None)
            except Exception as e:
                result_container['exception'] = e
            finally:
                done.set()
        self.root.after(0, _show)
        done.wait()
        if 'exception' in result_container:
            raise result_container['exception']
        return result_container.get('result')

    # remaining UI helpers reused from previous code
    def browse_file(self):
        p = filedialog.askopenfilename(filetypes=[("ResX files","*.resx")])
        if p:
            self.file_var.set(p)
            detected = detect_existing_culture_codes(p)
            self.detected_codes = detected
            if detected:
                self.safe_log(f"Detected existing localized files: {', '.join(sorted(detected))}")
            else:
                self.safe_log("No existing localized files detected.")
            self._select_detected_codes_in_listbox(detected)

    def _select_detected_codes_in_listbox(self, detected_codes):
        if not detected_codes:
            return
        detected_normal = {c.lower() for c in detected_codes}
        code_to_index = {code.lower(): idx for idx, (_, code) in enumerate(self.loaded_langs)}
        try:
            self.lang_listbox.select_clear(0, tk.END)
        except Exception:
            pass
        for code in detected_normal:
            if code in code_to_index:
                idx = code_to_index[code]
                try:
                    self.lang_listbox.select_set(idx)
                except Exception:
                    pass
            else:
                display = f"Existing ({code})"
                try:
                    self.lang_listbox.insert("end", display)
                    self.loaded_langs.append((display, code))
                    self.lang_listbox.select_set(self.lang_listbox.size()-1)
                except Exception:
                    pass

    def toggle_key(self):
        if self.key_entry.cget("show") == "":
            self.key_entry.config(show="*")
        else:
            self.key_entry.config(show="")

    def toggle_persist_api_key(self):
        self.settings["persist_api_key"] = self.persist_key_var.get()
        if not self.persist_key_var.get():
            self.settings["deepl_api_key"] = ""
            save_settings(self.settings)

    def load_glossary_dialog(self):
        p = filedialog.askopenfilename(filetypes=[("CSV or JSON","*.csv;*.json")])
        if not p:
            return
        try:
            g = load_glossary(p)
            self.glossary = g
            self.glossary_var.set(p)
            self.settings["last_glossary"] = p
            save_settings(self.settings)
            self.safe_log(f"Loaded glossary ({len(g)} entries) from {p}")
        except Exception as e:
            self.safe_show_error("Glossary error", f"Failed to load glossary: {e}")

    def clear_glossary(self):
        self.glossary = {}
        self.glossary_var.set("")
        self.settings["last_glossary"] = ""
        save_settings(self.settings)
        self.safe_log("Cleared glossary")

    def load_languages(self):
        key = self.key_var.get().strip() or os.getenv("DEEPL_API_KEY","")
        if not key:
            self.safe_show_info("No API key", "Provide DeepL API key to load languages.")
            return
        try:
            trans = make_translator(key)
            langs = get_target_langs(trans)
            prev_detected = set(self.detected_codes) if hasattr(self, "detected_codes") else set()
            self.lang_listbox.delete(0, tk.END)
            self.loaded_langs = []
            for l in sorted(langs, key=lambda x: x["name"]):
                disp = f"{l['name']} ({l['code']})"
                self.lang_listbox.insert("end", disp)
                self.loaded_langs.append((disp, l["code"]))
            self.safe_log(f"Loaded {len(self.loaded_langs)} languages from DeepL")
            if self.persist_key_var.get():
                self.settings["deepl_api_key"] = key
                self.settings["persist_api_key"] = True
                save_settings(self.settings)
            if prev_detected:
                self._select_detected_codes_in_listbox(prev_detected)
        except Exception as e:
            self.safe_log(f"Failed to load languages from DeepL: {e}")
            self.safe_show_info("Load failed", f"Could not fetch languages from DeepL: {e}. Using current list.")

# ---------- run ----------
def main():
    root = tk.Tk()
    app = AutoResxTranslatorApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        try:
            app.safe_log("KeyboardInterrupt received — stopping...")
            app._stop_requested = True
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass

if __name__ == "__main__":
    main()

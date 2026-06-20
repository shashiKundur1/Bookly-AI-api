# Bookly AI API

Backend for Bookly, a personal library where you upload PDF books, organize them on shelves, track reading progress page by page, and listen to natural AI narration with word-level timestamps.

Every user only ever sees their own books.

## Stack

- **FastAPI** on Python 3.12, fully async (SQLAlchemy 2 + asyncpg, Postgres 18)
- **PyMuPDF** for PDF processing: text + layout extraction, table of contents, cover rendering
- **Real-time narration over WebSockets** — sentences are synthesized one by one and streamed as raw PCM (sub-second first audio with the streaming engine), and the text, highlight, and page turns stay in sync with what is actually being spoken
- **Read Out Anything (browser extension)** — the same narration core powers a Chrome extension: highlight text on any web page and it streams to `WS /extension/narrate` through the shared `app/services/synthesis.py` engine. An empty-selection click saves the page as a book via `POST /extension/articles`. The extension authenticates REST with the existing session cookie and the WebSocket with a short-lived single-use ws-ticket — no cookie or web-app changes. See `Bookify-extension/`
- **Emotional narration** — 12 selectable tones (narrator, storyteller, dramatic, cinematic, excited, calm, whisper, …) drive pacing, pitch, and acting cues. A GoEmotions classifier (RoBERTa int8, 28 human emotion labels, runs on CPU via onnxruntime) reads every sentence and steers the delivery: laughter where the text is amused, hushed nerves where it is afraid. Boundary pauses follow speech-breathing research (sentence ≈ 0.6–0.9 s, paragraph ≈ 1.0–1.5 s, scaled by tone) — narrators inhale between lines, so there are no fake exhale sounds, ever
- **Five TTS engines** behind one interface, chosen with `TTS_ENGINE`: `gemini` (Gemini native TTS on the free API tier: cloud-side emotional acting — directed chuckles, gasps, whispers — zero server RAM, automatic edge-tts fallback on quota errors), `orpheus` (Orpheus 3B via llama.cpp + SNAC, truly emotive open-source voice with real laughs/gasps/breaths, streams PCM as it decodes; needs a llama-server host — see `ORPHEUS_URL`), `edge` (edge-tts neural voices: real-time on tiny servers, word timestamps), `kokoro` (torch, offline, word timestamps), and `kokoro-onnx` (lightweight offline). Engines degrade gracefully: if the Orpheus host is down, pieces fall back to edge-tts so narration never stalls
- **Gemini** (optional) to polish narration text for awkward pages like tables — used only when `GEMINI_API_KEY` is set, with automatic fallback. Everything in this project runs on free and open-source resources
- **Docker Compose** for the whole stack, with hot reload in development

## Getting started

```bash
cp .env.example .env        # set JWT_SECRET to something long and random
docker compose up
```

That is all. The API container runs database migrations on boot and serves on `http://localhost:8000` (OpenAPI docs at `/docs`). Code changes under `app/` reload automatically.

The first time narration audio is requested, the Kokoro model (~310 MB) is downloaded into the `bookly-data` volume and reused forever after. Expect that one request to take a couple of minutes; everything after it is fast.

## How a book flows through the system

1. **Upload** — a PDF is streamed to disk (magic-byte validated, size capped) and a book row is created.
2. **Extraction** — in the background, PyMuPDF walks every page: it detects the body font size, strips repeated running headers, footers, page numbers and rotated watermarks, classifies headings (by size, boldness and TOC matching), list items and paragraphs, and stores normalized bounding boxes for highlight overlays.
3. **Narration script** — blocks are packed into speech chunks: headings are announced ("Chapter 1. Welcome to Software Construction."), numbered chapters get a natural prefix, dot leaders become ", page 463", bullets are read with pauses, fragmented table cells are merged into flowing units.
4. **Listening** — the client opens `WS /api/v1/books/{id}/narrate` and the server streams each chunk sentence by sentence: a JSON frame with the sentence text, word timestamps, and acting cues, then raw 24 kHz PCM frames. The emotion planner classifies each sentence (GoEmotions) and chooses pace, pitch, pauses, and emotive tags per the selected tone; the streaming engine pushes PCM the moment frames decode, then sends exact word timings (energy-aligned for generative voices) in a correction frame. Completed chunks are cached to disk per voice + emotion + speech hash (cached chunks replay instantly and polish invalidates stale audio automatically); seek/voice/speed/emotion changes apply live over the same socket.
5. **Polish (optional)** — when a Gemini key is configured, the first listen of a page rewrites its chunks into smoother spoken prose. Failures fall back silently and a circuit breaker stops retrying a dead key.

## API overview

All routes live under `/api/v1`. Authentication uses httpOnly JWT cookies (an `Authorization: Bearer` header works too).

| Area | Routes |
| --- | --- |
| Auth | `POST /auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout` |
| Profile | `GET/PATCH /users/me`, `PUT /users/me/password`, `GET/POST /users/me/avatar`, `GET /users/me/stats` |
| Library | `GET/POST /books`, `GET/PATCH/DELETE /books/{id}`, `PUT /books/reorder`, `POST /books/{id}/reprocess` |
| Files | `GET /books/{id}/file` (range requests supported), `GET/POST/DELETE /books/{id}/cover` |
| Reading | `GET/PUT /books/{id}/progress`, `POST /books/{id}/sessions`, `PATCH /sessions/{id}` |
| Narration | `WS /books/{id}/narrate` (real-time streaming), `GET /voices`, `GET /emotions`, `GET /books/{id}/content`, `GET /books/{id}/pages/{n}` |
| Extension | `POST /extension/ws-ticket` (short-lived WebSocket ticket), `WS /extension/narrate` (read highlighted text), `POST /extension/articles` (save a page as a book) |

Book organization is first-class: status (`to_read`, `reading`, `finished`), priority, a color mark, favorites, manual shelf order, and search across title and author. Progress tracks the current page plus a per-page read map, and status moves forward automatically as you read.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | compose-internal Postgres | SQLAlchemy async URL |
| `JWT_SECRET` | change me | signs access and refresh tokens |
| `ACCESS_TOKEN_MINUTES` / `REFRESH_TOKEN_DAYS` | 30 / 30 | token lifetimes |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | allowed browser origins |
| `MAX_UPLOAD_MB` | 200 | PDF size cap |
| `DEFAULT_VOICE` | `af_heart` | narration voice when none is chosen (auto-mapped per engine) |
| `TTS_ENGINE` | `kokoro` | `gemini` (free-tier emotional acting, recommended for small servers), `orpheus` (emotive, needs llama-server), `edge` (real-time neural voices), `kokoro` (torch), or `kokoro-onnx` (lite offline) |
| `GEMINI_TTS_MODEL` | `gemini-3.1-flash-tts-preview` | Gemini TTS model used when `TTS_ENGINE=gemini` |
| `ORPHEUS_URL` | `http://localhost:8080` | llama-server hosting `orpheus-3b-0.1-ft-q4_k_m.gguf` (run: `llama-server -m data/models/orpheus-3b-0.1-ft-q4_k_m.gguf -c 8192 --port 8080 -ngl 99`); needs `snac_24khz.decoder.onnx` in `data/models` |
| `GEMINI_API_KEY` | empty | enables AI narration polish |
| `COOKIE_SECURE` | false | set true behind HTTPS |

## Testing

A full Postman collection with 60+ assertions lives in `postman/`. Run it locally:

```bash
npx newman run postman/bookly-api.postman_collection.json \
  -e postman/local.postman_environment.json --timeout-request 600000
```

It covers the happy paths, auth failures, validation errors, range streaming, reorder persistence, narration synthesis and cleanup.

## Production

With this repo and [Bookly-AI-client](https://github.com/shashiKundur1/Bookly-AI-client) cloned side by side:

```bash
JWT_SECRET=$(openssl rand -hex 32) docker compose -f docker-compose.prod.yml up -d --build
```

That starts Postgres (internal only), the API (TTS warmed on boot, health-checked, auto-restarting), and the web app on port 3000. Set `POSTGRES_PASSWORD`, `CORS_ORIGINS`, and `COOKIE_SECURE` through the environment or a `.env` file next to the compose file.

The API ships with a caching layer: content endpoints answer conditional requests with ETags and 304s, narration audio and word timings are cached for a week, the voice list is publicly cacheable for a day, and JSON responses are gzip-compressed. Security headers (nosniff, frame deny, referrer policy) are applied to every response.

Worth knowing before exposing it to the internet:

- Terminate TLS in front (Caddy, nginx, Traefik) and keep `COOKIE_SECURE=true`.
- The image is large (~9 GB) because Kokoro runs on PyTorch; swap to `kokoro-onnx` for a ~10x smaller image if you can live without word timestamps.
- PyMuPDF is AGPL-licensed: fine for personal use, review the license before offering this as a hosted service.
- Add a rate limiter at the proxy layer.

## Roadmap

Everything planned stays on free and open-source resources — self-hosted models, free-tier APIs, and infrastructure already in place. No paid services, ever.

### Formats and library

- **EPUB, MOBI, FB2 and CBZ support** — PyMuPDF already opens all of these; the work is relaxing the PDF magic-byte check on upload and teaching extraction about reflowable layouts. Removes the single biggest limitation of a PDF-only library.
- **Highlights, bookmarks and notes** — extraction already stores normalized bounding boxes per block; an annotations table (book, page, block, color, note) plugs straight into the existing overlay system.
- **Full-text search inside books** — every page's text already lives in the content store; indexing it into a Postgres `tsvector` at extraction time gives search across the whole library with jump-to-result in the reader.
- **Open Library metadata autofill** — on upload, look the book up by title and author for a description, genre tags, and a canonical cover (free API).
- **Named shelves and tags** — book organization beyond the current status/priority/color/favorites model.

### Listening

- **Audiobook export** — batch-synthesize a whole book through the existing engine and cache, then mux to `.m4b` with chapter markers taken from the TOC via ffmpeg (already in the image). Download your library as audiobooks.
- **Resume listening position** — progress currently tracks pages; storing the last narrated chunk plus per-book voice, emotion and speed preferences lets Listen resume exactly where it stopped, sounding the way you left it.
- **Multi-voice dialogue narration** — detect quoted dialogue in chunks (quote detection already feeds the prosody planner) and narrate it in a second voice. Kokoro and Orpheus both ship multiple voices.
- **Multilingual narration** — edge-tts exposes hundreds of free voices across dozens of languages; detect the book's language at extraction and offer matching voices.

### AI features (free tier and self-hosted only)

- **Ask the book** — RAG chat over a book's content: pgvector (free Postgres extension) for storage, an int8 ONNX embedding model (all-MiniLM-L6-v2, ~23 MB, same onnxruntime already shipped for GoEmotions) for retrieval, Gemini free tier as the answerer behind the existing circuit-breaker pattern.
- **Chapter summaries and "previously on…"** — reopening a book after days away shows a one-paragraph recap of everything up to the current page, cached per page range in the content store the same way polish results are.
- **Reading goals and streaks** — daily streaks, pages and minutes goals, and a year-in-books recap, all computed from the reading-session rows already being written.

### Hardening

- **Refresh-token rotation and revocation** — tokens already carry a `jti` that nothing checks; logout should actually invalidate the refresh token via a denylist instead of only clearing cookies.
- **Durable extraction queue** — extraction runs as an in-process background task, so a restart mid-job leaves a book stuck at `processing`. A boot-time sweep that re-queues stuck books, or a Postgres-backed queue (procrastinate), fixes both.
- **Auto-close stale reading sessions** — sessions whose client disappeared keep `ended_at` null and silently vanish from stats; cap them at start time plus a few hours.
- **Audio cache eviction** — narration audio accumulates per book × voice × emotion with no bound; a size-capped LRU sweep keeps small servers healthy.
- **Password reset and email verification** — over plain SMTP, no paid mail service.
- **In-app rate limiting** — slowapi on auth and upload endpoints instead of relying on the proxy.
- **Per-user storage quotas** — a sum-of-file-sizes check at upload, needed before opening registration to others.
- **pytest suite** — the Postman collection covers the API surface; the emotion planner, chunk builder and extraction classifier deserve fast unit tests too.
- **Observability and backups** — structured logs, a Prometheus metrics endpoint, and a nightly `pg_dump` plus data-dir backup to a free object-storage tier.

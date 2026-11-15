# Agent Instructions — HoloCube

A standalone Quest 3 / 3S app with an on-device Transmitter Recognition Model (a single-class trained YOLO26n detector) that loads the supplied `holocube_transmitter.pt` checkpoint using embedded Python/PyTorch, plus an eight-class Orientation Detection Model (`holocube_orientation.pt`, YOLO26n-seg, in `Assets/HoloCube/Orientation Model`) that runs after it on each frame. Start in `Assets/HoloCube/`: `Main App`, `Transmitter Inference`, and `Transmitter Model` are the three runtime parts. Meta's original camera samples remain as supporting assets and references; `QuestYOLO.unity` is the app's only build scene.

## Source-of-truth files (read these first, do not duplicate their contents in this file)

For setup, build steps, SDK versions, and project layout, read:

- `README.md` — app overview, research inspiration, file structure, labels, build steps
- `ProjectSettings/ProjectVersion.txt` — Unity editor version
- `Packages/manifest.json` — Unity dependencies; `Assets/HoloCube/Editor/PyTorchAndroidBuild.cs` — Python runtime packaging
- `Assets/HoloCube/` — app scene, runtime code, model, and editor validation tools
- `Docs/DEVELOPMENT.md` and `VALIDATION.md` — model format, limitations, and checks actually performed
- `UPSTREAM_README.md` — original Meta sample documentation
- `LICENSE.txt` and `Assets/PassthroughCameraApiSamples/LICENSE.txt` — license terms

## Quest / Horizon-specific notes

- Requires **Quest 3 / 3S** — older Quest hardware does not expose passthrough camera frames.
- App needs the `horizonos.permission.HEADSET_CAMERA` permission and passthrough enabled in the project; do not strip these from the manifest.
- **XR Simulator does not support the Passthrough Camera API.** Test on a physical device or via Meta Horizon Link 2.1+; agents that try to verify in-editor will silently get nothing.
- The active transmitter model (YOLO26 architecture) carries the included Ultralytics AGPL-3.0 license. The legacy upstream YOLO model has a separate MIT notice; keep both licenses with their respective files.
- Model and SDK assets are stored directly in Git; this repository does not currently use Git LFS.

# Meta Quest tooling

This is a Meta Quest / Horizon OS sample. The bespoke intro above is the source of truth for what this project is and how it's built — use it (and the files it points at) instead of restating facts from memory.

When the user asks anything about Quest device behavior, build / deploy / debug / capture flows, on-device performance, or Horizon OS APIs, reach for these tools instead of generic Unity answers:

- **`hzdb`** — Quest-aware ADB wrapper (device list, install / launch / stop, logs, screenshots, Perfetto traces, on-device docs search). Already wired up as an MCP server via `.mcp.json`, `.vscode/mcp.json`, and `.cursor/mcp.json`. Also runnable directly: `npx -y @meta-quest/hzdb <subcommand>`.
- **Meta Quest Agentic Tools** — the full skill set, including Unity-specific skills: <https://github.com/meta-quest/agentic-tools>. Install per your client (Claude Code: `/plugin install meta-vr@meta-quest`; Gemini CLI: `gemini extensions install https://github.com/meta-quest/agentic-tools`; Cursor / VS Code: install the **Meta Horizon** extension from the Marketplace).

A few behavior expectations:

- **Read this repo's files first.** Before answering anything project-specific, read `README.md` and whichever source-of-truth files the intro above points at. Don't restate their contents in chat — quote or link instead.
- **Use `hzdb` for device-side work.** Anything that touches an attached Quest (install, launch, logs, screenshot, capture, manifest inspection) goes through `hzdb`, not raw `adb`.
- **Check live Horizon OS docs before answering API questions.** `hzdb docs search "..."` queries the live docs; training data on Horizon OS APIs goes stale fast.
- **Don't fabricate SDK / engine versions.** If a version isn't visible in this repo's files, say so rather than guessing.

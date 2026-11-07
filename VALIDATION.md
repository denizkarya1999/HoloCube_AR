# HoloCube validation

## In-Device Local File Inference — November 7, 2026

- HoloCube **0.9 (code 9)**, package `com.developer27.holocube`, installed on the Quest 3S
  over 0.7 (the 0.8 pyramid-model build had never reached the headset). Adds
  `LocalFileInference` (Settings → right trigger): a runtime-built panel that runs the
  Transmitter Detection Model on images stored on the headset (`Pictures/HoloCube Test
  Dataset`, or the app's own files folder) and shows the image, the box in image pixels,
  its confidence, and the inference time; joystick left/right picks the image, A runs it
  again, Y returns to Settings. The Settings controls line and the controller guide
  mention it, the manifest gains `READ_MEDIA_IMAGES` (and `READ_EXTERNAL_STORAGE` for
  Android 12), the scene setup and validation gained matching code, and
  `Tools/push-test-images.sh` copies test images to a device. Same checkpoint as 0.8
  (SHA-256 `1b65b56b…`).
- Before the Unity runs, the runtime scripts were compiled with Unity's bundled Roslyn
  against the editor's `UnityEngine.*Module.dll` assemblies with stubs for the Meta SDK
  and UGUI types; this only caught type and syntax errors.

### Passed on the development computer

- Unity batch mode needed a refreshed Hub sign-in (the licensing client reported "Access
  token is unavailable" until Unity Hub was started once); after that:
- `bash Tools/run-unity.sh Validate`: scene wiring, decoder checks, the new local-file
  picture-fit and box-mapping checks, and the PyTorch 1.8.1 parity on the three fixtures
  (`HOLOCUBE_VALIDATION_PASSED`, `Builds/Validate-local-file.log`).
- `bash Tools/run-unity.sh ValidateGPU` on the RTX 3050 Ti: padding, orientation, and
  sRGB checks passed; its real RGBA output through the checkpoint reproduced the reference
  box (`HOLOCUBE_PT_GPU_INPUT_PASSED`, `Builds/Validate-local-file-gpu-model.log`).
- `bash Tools/run-unity.sh Build`: scene regenerated, validation repeated, packaging guard
  passed (`HOLOCUBE_TRANSMITTER_PACKAGE_PASSED`). The regeneration-only object-ID churn in
  the saved scene was discarded afterwards, as in earlier releases.

APK: `Builds/HoloCube.apk`, 111,090,040 bytes, SHA-256
`cd448d1e24f035a850f41c69f8f21af7c01cb2dce1abce558380a2eb879e2dd0`; `aapt` reports 0.9 /
code 9 with `READ_MEDIA_IMAGES` and `READ_EXTERNAL_STORAGE`; the Python archive holds one
`.pt` with the approved hash and the `frame_01153` startup reference.

### Passed on the connected Quest 3S

- `adb install -r` over 0.7 succeeded (package information confirms 0.9 / code 9; camera
  and microphone permissions preserved). Cold launch: `HOLOCUBE_PT_DEVICE_PARITY_PASSED`,
  `HOLOCUBE_PT_READY`, first live camera frame in **137 ms** (190 ms on a later launch);
  no crash. This is the pyramid-only model's first run on a headset.
- `bash Tools/push-test-images.sh` placed 18 frames of the pyramid dataset's testing
  split (five per clip) plus the three parity-fixture frames in `Pictures/HoloCube Test
  Dataset`, registered with the media store.
- Worn by the user: Settings → right trigger opened the panel, Horizon OS showed the
  photo-access prompt and the user allowed it (`READ_MEDIA_IMAGES` now `granted=true`,
  user-set), the page found the images (`HOLOCUBE_LOCAL_FILES: 18 images found`), ran the
  model, and the joystick/A controls worked. Headset screenshots show the panel with
  `bright_static_frame_01360.jpg (1/18) · 1280 × 720` and, after stepping to image 3,
  `bright_static_frame_01432.jpg (3/18)` with one maize box labelled **Transmitter 81 %**
  around the pyramid (`Docs/Screenshots/local-file-inference-quest.png`); the file path
  logged `HOLOCUBE_TRANSMITTER_SINGLE_BOX`. Device log: `Builds/Quest-local-file.log`.
- Note for later log reading: the Quest's logcat ring buffer is 256 KB and loses the app's
  lines within a minute or two; `adb logcat -G 16M` plus a continuous capture was needed.
  The app process is reclaimed by the system (`has died: cch`) whenever it sits in the
  background, so it must be relaunched before testing.

### Remaining checks

- The reported boxes were not compared numerically with the dataset labels of the same
  frames; the 81 % box on `frame_01432` visibly covers the pyramid and part of the cap.
- The app's own `Android/data/.../files/HoloCube Test Dataset` fallback folder, the
  no-images message, and the permission-denied path were not exercised.
- Live camera detection of the physical transmitter with the pyramid-only model, and
  longer headset sessions.

---

## Pyramid-only Transmitter Recognition Model — November 6, 2026

- HoloCube **0.8 (code 8)**, package `com.developer27.holocube`. Version and model
  integration only: no APK was built and nothing was installed on a Quest in this
  session.
- Replaced the checkpoint with the file supplied on the Desktop on November 6
  (10,384,601 bytes), embedded unchanged. The only class is still **0: Transmitter**,
  but it now means the transmitter's **central pyramid** (the black inverted hologram
  prism inside the cube frame). Checkpoints through 0.7 boxed the whole assembly, so
  boxes from this release are much smaller than before.
- Model SHA-256:
  `1b65b56bb61c0102bb963e34126d1e58cb0c459f0f14dc5a232262ec62d0e260`.
- Trained in the Colab notebook from `HoloCube_Transmitter_Pyramid_YOLO26_1323frames.zip`
  (928 training / 264 validation / 131 testing frames, every 3rd frame of three
  same-room recordings, semi-automatic labels): YOLO26n at 320 px, 59 of 100 epochs
  before early stopping. Ultralytics scores on the dataset's own splits: validation
  mAP50 0.79 / mAP50-95 0.59, testing mAP50 0.95 / mAP50-95 0.71 (full table in
  `provenance.json`). These are same-room, same-rig numbers, not a new-scene benchmark,
  and the validation split contains the steep side-angle frames where the labels are
  least reliable.
- New fixtures from that dataset's validation split (1280 × 720 frames): bright
  `frame_01153`, angled `frame_03905`, and low-light `frame_02444`, with reference boxes
  from independent Ultralytics 8.4.56 / PyTorch 2.12.0 inference on the unfused model.
  Each has exactly one detection above 35%, on the pyramid.
- Desktop PyTorch 1.8.1 (`Tools/validate-pt.py`, the embedded runtime's version)
  matched all three references within 0.001 confidence and 1 original-image pixel.
  Blank input, signed RGBA orientation, startup-reference mismatch, and
  modified-checkpoint rejection checks passed. Log: `Builds/Validate-pyramid-python.log`.
- `bash Tools/run-unity.sh Validate` did **not** run: Unity 6000.0.66f2 exited with
  "No valid Unity Editor license found" before executing the method
  (`Builds/Validate-pyramid-unity-license.log`). The scene, decoder, GPU-preprocessing,
  and APK-guard checks were therefore not repeated. The only Unity-side changes are the
  checkpoint hash constant, which equals the file's SHA-256, and the version bump.
- The Android phone edition received the same checkpoint, hash, fixtures, and
  provenance (HoloCube Android 1.1); its checks are recorded in that repository.

Remaining checks for this release:

- Sign in to Unity Hub, then run `Validate`, `ValidateGPU`, and `Build`.
- Install on a Quest 3 / 3S: startup parity (`HOLOCUBE_PT_DEVICE_PARITY_PASSED`), live
  camera detection of the physical transmitter, and whether the smaller pyramid box
  reads well in the headset at typical distances.
- Any scene other than the lab room, and the side-angle views where the dataset's own
  validation score is lowest.

---

## Single Transmitter box release — October 23, 2026

- HoloCube **0.7 (code 7)**, package `com.developer27.holocube`, installed on Quest 3S.
- The transmitter pipeline now returns at most one detection: the highest-confidence
  valid Transmitter above the selected threshold. If none qualifies, no box is shown.
  The limit is fixed in the pipeline; the old scene field and joystick box-count
  adjustment have been removed. Settings shows **Number of boxes (fixed): 1** and
  still lets the user adjust confidence with the left joystick's left/right axis.
- Unity validation passed with deliberately unsorted candidates: the strongest
  valid transmitter wins, while stronger wrong-class and invalid-box candidates
  are rejected. The no-confident-result case returns no box. Existing coordinates,
  scene wiring, model parity, and negative-input checks also passed.
- Android ARM64/IL2CPP build, recorder packaging, and the single-model packaging
  guard passed. The saved scene was validated again after discarding unrelated
  regeneration-only object-ID changes.
- Independent APK inspection confirmed the checkpoint is unchanged from 0.6 and
  matches the supplied Downloads file byte for byte; startup references also match.
  Model SHA-256: `3da988ae9e3aa5a66b175adb6275463d21f48828faac5ad77ebd0465570e7cf5`.
- Installed with `hzdb app install --replace`, preserving app data. Device package
  information confirms 0.7 / code 7. Cold launch focused the app and passed its
  crash scan. After the user opened HoloCube facing the transmitter, a fresh
  launch confirmed startup model parity and live camera inference. The first
  logged camera worker call took **147 ms**, and the Unity pipeline reported
  exactly one highest-confidence detection with its fixed limit of 1. These
  markers were captured on October 23 at 21:35:44. This is a single worker-call
  timing sample, not sustained FPS or end-to-end display latency.
- A subsequent stereo headset screenshot visually confirms one green box labelled
  **Transmitter 98%** around the transmitter, rendered once in each eye's view.
  The local capture is `Builds/Quest-single-box-overlay.png`.

Headset log markers:

```text
HOLOCUBE_PT_DEVICE_PARITY_PASSED: Transmitter Recognition Model; reference boxes match.
HOLOCUBE_PT_READY: Transmitter Recognition Model loaded on device; class 0: Transmitter
HOLOCUBE_PT_CAMERA_FRAME: 300 detection rows in 147 ms
HOLOCUBE_TRANSMITTER_SINGLE_BOX: 1 detection selected; highest confidence; limit 1.
```

APK: `Builds/HoloCube.apk`, 111,049,244 bytes. SHA-256:
`0b44fc1910b12ed70780dc523318b281f6b29ab820dcd51f3a61eaaaa5494de4`.
Local logs: `Builds/Build-single-box.log`, `Builds/Validate-single-box.log`,
`Builds/Quest-single-box-app.json`, `Builds/Quest-single-box-launch.log`, and
`Builds/Quest-single-box.log`; live verification is recorded in
`Builds/Quest-single-box-live-launch.log` and `Builds/Quest-single-box-live-relaunch.log`.
Artifacts and logs remain excluded from Git.

This cap chooses one candidate per frame; it does not add object tracking or
improve the detector's accuracy. The model limitations documented below still apply.

---

## Updated Transmitter Recognition release — October 23, 2026

- HoloCube **0.6 (code 6)**, package `com.developer27.holocube`, installed on Quest 3S.
- Replaced the checkpoint with the file supplied in Downloads on October 23.
  Its 10,386,814 bytes are embedded unchanged; its only class is **0: Transmitter**.
- Model SHA-256:
  `3da988ae9e3aa5a66b175adb6275463d21f48828faac5ad77ebd0465570e7cf5`.
- Desktop PyTorch 1.8.1 matches independent Ultralytics 8.4.56 / PyTorch 2.12.0
  inference on bright, angled, and low-light validation frames within 0.001
  confidence and 1 original-image pixel. Blank input, signed RGBA orientation,
  startup-reference mismatch, and modified-checkpoint rejection checks passed.
- Unity scene wiring, decoder/filter checks, and Android ARM64/IL2CPP build passed.
  The original saved scene was validated again after discarding regeneration-only
  object-ID changes. Actual desktop Vulkan preprocessing passed padding,
  orientation, and sRGB checks; its RGBA output also passed model comparison.
- The final build guard and independent APK archive inspection confirmed exactly
  one embedded `.pt`, matching the supplied file byte for byte, and matching
  startup reference files. Recorder packaging checks also passed.
- Installed through `hzdb app install --replace`, preserving existing app data.
  Device package information confirms version 0.6 / code 6. The first cold launch
  focused the app and passed the crash scan. After the user woke the headset and
  opened HoloCube, the on-device startup reference passed and live camera inference
  returned 300 candidate rows in **191 ms**. This is one worker-call sample, not
  sustained FPS or end-to-end display latency. These markers were recorded on
  October 23 at 21:26:35, after the version 0.6 installation.

Headset log markers:

```text
HOLOCUBE_PT_DEVICE_PARITY_PASSED: Transmitter Recognition Model; reference boxes match.
HOLOCUBE_PT_READY: Transmitter Recognition Model loaded on device; class 0: Transmitter
HOLOCUBE_PT_CAMERA_FRAME: 300 detection rows in 191 ms
```

APK: `Builds/HoloCube.apk`, 111,053,040 bytes. SHA-256:
`fda6445ab2f68b7f3b3f2e7f05a7f0086e1830fb7ee7ad67e71331de48f307ce`.
Local logs: `Builds/Build.log`, `Builds/Validate.log`, `Builds/ValidateGPU.log`,
`Builds/Validate-300frames-python.log`, `Builds/Validate-300frames-gpu-model.log`,
`Builds/Quest-300frames-app.json`, `Builds/Quest-300frames-launch.json`, and
`Builds/Quest-300frames-live.log`.
Build artifacts and device logs are excluded from Git.

The fixtures come from the validation split of the prepared 300-frame dataset
(210 training / 60 validation / 30 testing images). The supplied model-only
checkpoint contains no training history or accuracy metrics, so this dataset is
recorded as intended context rather than verified training provenance. These are
runtime compatibility checks, not an accuracy benchmark. An additional clipped
close-up validation frame produced no detection above the app's 35% threshold.
Longer headset testing and visual confirmation of the updated detector remain.

---

## Transmitter Recognition release — October 19, 2026

- HoloCube **0.5 (code 5)**, package `com.developer27.holocube`, deployed to Quest 3S.
- Replaced the general COCO detector with the user-supplied `holocube_transmitter.pt`.
  The checkpoint is embedded unchanged. The only model class is **0: Transmitter**.
- Model SHA-256:
  `8c6e981c7a49ff52c7113cf9cd95763ae6495e3baab3e3d68c17a6919e3d8fef`.
- Desktop PyTorch 1.8.1 matches the independent Ultralytics 8.4.56 / PyTorch 2.12.0
  oracle on three transmitter frames within 0.001 confidence and 1 original-image
  pixel. Blank input, RGBA orientation, startup reference mismatch, and modified
  checkpoint rejection checks also passed.
- Unity scene wiring, single-class decoder/filter checks, and Android ARM64/IL2CPP
  build passed. Moved assets retain their original Unity GUIDs. The existing saved
  scene was validated again after discarding regeneration-only object-ID changes.
- Actual Vulkan GPU preprocessing passed padding, orientation, and sRGB checks;
  its RGBA output produced the expected transmitter box through the new model.
- The final APK guard and a separate archive inspection confirmed exactly one
  embedded `.pt`: the checksum-verified transmitter checkpoint. The cached general
  `yolo26n.pt` is removed during incremental builds. Recorder packaging checks passed.
- Installed with `hzdb app install --replace`, preserving app data. Cold launch
  focused the app and passed the crash scan. The on-device startup reference passed,
  and a live camera frame returned 300 candidate rows in **157 ms**. This is one
  worker-call sample, not sustained FPS or end-to-end display latency.
- **The user confirmed a box labelled “Transmitter” appears around the transmitter
  in the Quest 3S headset.**

Headset log markers:

```text
HOLOCUBE_PT_DEVICE_PARITY_PASSED: Transmitter Recognition Model; reference boxes match.
HOLOCUBE_PT_READY: Transmitter Recognition Model loaded on device; class 0: Transmitter
HOLOCUBE_PT_CAMERA_FRAME: 300 detection rows in 157 ms
```

APK: `Builds/HoloCube.apk`, 111,063,016 bytes. SHA-256:
`ee604a527600c11731ddcb3b280d3beac7835f09ae2c23c80d2c1c83c6c2fd72`.
Local logs: `Builds/Build.log`, `Builds/Validate.log`, `Builds/ValidateGPU.log`,
`Builds/Quest-transmitter.log`. Build artifacts and logs are excluded from Git.

The three reference images come from the dataset source recording and may overlap
training. These checks establish runtime compatibility and a successful live use,
not a held-out accuracy benchmark. Generalization to other lighting, viewpoints,
transmitter variants, multiple transmitters, and sustained thermal behavior remains
unmeasured. Detection provides boxes and class labels, not optical data decoding.

---

# Historical general-detector validation

Verified on October 4, 2026.

Current builds use Android app ID `com.developer27.holocube`. The historical
checks below used the earlier package ID and artifacts recorded here.

## Model and app

- App: HoloCube 0.4 (code 4), package `com.holocube.questyolo`.
- Model: original Ultralytics `yolo26n.pt`; no ONNX, TorchScript, or other model export.
- Checkpoint SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- The checkpoint extracted from the APK's embedded Python archive has the exact
  same SHA-256 as the original downloaded file.
- Embedded runtime: Python 3.8, PyTorch 1.8.1, NumPy 1.19.5, Chaquopy 16.1.0.
- Input `[1,3,320,320]` RGB floats in `[0,1]`; output `[1,300,6]` end-to-end rows.
- Export-free loading uses inference-only layer methods adapted from Ultralytics
  8.4.56 for this exact checkpoint. Arbitrary checkpoints are not supported.

## Passed on the development computer

- Unity 6000.0.66f2 compilation and Android ARM64/IL2CPP/Vulkan APK build.
- Scene wiring, original checkpoint hash, 80 labels, and shader references.
- Letterbox coordinate mapping; confidence/class checks; detection limits;
  invalid-value and incompatible-output rejection; no second NMS pass.
- Direct PyTorch 1.8.1 inference from the unchanged checkpoint produces four
  people and one bus, matching the independent PyTorch 2.12.0 fixture within
  0.001 confidence and 1 original-image pixel.
- Actual Vulkan GPU image preparation on the desktop NVIDIA RTX 3050 Ti passes
  padding, orientation, sRGB, and RGBA-readback checks.
- Those real Unity RGBA pixels also produce four people and one bus through
  the new Python preprocessing and direct checkpoint inference.
- APK inspection confirms the original `.pt` and reference fixture are embedded
  in `assets/chaquopy/app.imy`, with no exported model in that Python archive.

## Passed on the connected Quest

- Installed the APK over the existing app while preserving its data.
- Cold launch completed with no crash reported by the Quest launch verifier.
- Embedded Python/PyTorch loaded the original checkpoint on the headset.
- The startup reference check passed on the headset: five detections match the
  independent fixture's class IDs, confidence scores, and box coordinates.
- The live camera path returned 300 detection rows from a captured frame.
- The first logged camera-frame worker call took **155 ms**. This is one sample
  including Python input/output handling, not sustained FPS or total AR latency.

Headset log markers:

```text
HOLOCUBE_PT_DEVICE_PARITY_PASSED: original checkpoint; five reference detections match.
HOLOCUBE_PT_READY: original YOLO26n checkpoint loaded on device
HOLOCUBE_PT_CAMERA_FRAME: 300 detection rows in 155 ms
```

## Remaining checks

Visual stereo-overlay alignment, repeated pause/resume cycles, sustained frame
rate, memory use, and thermal behavior still need longer headset testing.
The runtime uses older Python/PyTorch packages to enable direct `.pt` loading;
desktop and initial headset success do not establish compatibility with every
future Horizon OS version or arbitrary trained YOLO checkpoints.

## Local artifacts

Builds and logs are excluded from Git. Reproduce them using the Unity menu or
`Tools/run-unity.sh` after running `Tools/setup-python.sh`.

- APK: `Builds/HoloCube.apk`
- Size: 110,003,196 bytes
- SHA-256: `07a5d2e7e52f2894a4c4d45a2cd70198d1298da881324fc7397b72f7d52728d3`
- Previous build: `Builds/HoloCube-before-PT.apk`
- Build/CPU checks: `Builds/Build.log`
- GPU preprocessing checks: `Builds/ValidateGPU.log`
- GPU-readback fixture: `Builds/pt-preprocessed.rgba`

Run `.build-tools/pt/bin/python Tools/validate-pt.py --rgba Builds/pt-preprocessed.rgba`
after `bash Tools/run-unity.sh ValidateGPU` to repeat the GPU-to-PyTorch comparison.

## Recording feedback update — October 5, 2026

- Rebuilt from the complete repository. The previously deployed APK lacked the
  native `QuestCameraVideoRecorder` class, even though its source was in Git.
- Android build guards now require the recorder source, independent recording
  and save-feedback canvases, the red circle renderer, and the recorder class
  in the completed APK/AAB's DEX files.
- The final build passed the existing scene/model validation and Android build
  guards. Log: `Builds/Build-final.log`.
- Installed on the connected Quest 3S with app data preserved. Cold launch
  completed, the app was foreground, and the launch verifier found no crash.
- After launch, four new B-button recordings completed on the Quest. Device
  logs reported `HOLOCUBE_VIDEO_STARTED`, `HOLOCUBE_VIDEO_SAVING`, and
  `HOLOCUBE_VIDEO_SAVED` for each, including the 34-frame
  `Movies/HoloCube/HoloCube_20261005_182656_785.mp4`. All four files were present
  in the headset's Movies/HoloCube folder.
- Device model parity also passed, and live camera inference returned 300
  detection rows. Log: `Builds/Quest-final.log`.
- The user confirmed the red recording badge and saved-file message now work
  in the headset. The inference-off timer is set to three seconds; a separate
  visual timing check was not reported.
- Final APK SHA-256:
  `4b12f0f8f26d9af11d023418022abb5e3d59acb2d0e45031b2102b486b910505`.

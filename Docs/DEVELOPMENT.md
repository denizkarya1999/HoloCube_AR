# Build and development notes

HoloCube loads the trained `holocube_transmitter.pt` with Python `torch.load` on the Quest.
Its SHA-256 is checked before loading. The supplied native `.pt` is embedded unchanged; deployment performs no further
model export or conversion.
`checkpoint_runtime.py` supplies inference-only module methods adapted from
Ultralytics 8.4.56; the checkpoint's saved module structure and weights are restored
directly. Training and export dependencies are not included in the app.

| Component | Version |
| --- | --- |
| Unity | 6000.0.66f2 |
| Meta MR Utility Kit / XR Core | 85.0.0 |
| OpenXR | 1.15.1 |
| Embedded Python | 3.8 |
| Embedded PyTorch | 1.8.1, CPU |
| Embedded NumPy | 1.19.5 |
| Chaquopy Android packaging | 16.1.0 |
| Model | Trained single-class YOLO26n transmitter `.pt` checkpoint |
| Android | ARM64, IL2CPP, Vulkan; minimum API 32, target API 35 |

PyTorch 1.8.1 is available as a Python package for Android ARM64 through Chaquopy.
This older runtime is used specifically for direct checkpoint loading. A desktop parity test compares three transmitter frames with the independent
Ultralytics implementation. See `VALIDATION.md` for device checks on each release;
recorded checks from earlier releases do not establish this model’s performance. Unity Inference Engine remains a dependency of Meta's reference
samples, but the HoloCube pipeline does not use it.

## Build

1. Install Unity's Android Build Support, SDK/NDK, and OpenJDK.
2. On Linux, run `bash Tools/setup-python.sh`. It creates isolated Python 3.8
   and PyTorch tools under `.build-tools/`, which is excluded from Git.
3. Open `Assets/HoloCube/QuestYOLO.unity` in Unity.
4. Choose **HoloCube → Build standalone Quest APK**, or run
   `bash Tools/run-unity.sh Build`. Output: `Builds/HoloCube.apk`.

The build tool regenerates the scene from Meta's configured rig. Preserve custom
scene edits before running it. To build a manually edited scene, use Unity's
Android Build UI with only `QuestYOLO.unity` enabled.

`PyTorchAndroidBuild.cs` adds Chaquopy to the generated Android project and embeds
the Python source, Java worker, trained transmitter checkpoint, and Python dependencies.
Dependency downloads happen on the build computer. The installed app works offline.
A small reference-image tensor is bundled for a one-time startup inference check;
its known detections must match before the model accepts camera frames.
Incremental builds remove the former `yolo26n.pt` from generated Python sources.
The final APK/AAB guard checks that exactly one `.pt` is packaged and verifies its
SHA-256 against the supplied transmitter weights, preventing stale generic models.

Set `UNITY_EDITOR_PATH` if Unity is installed elsewhere. `HOLOCUBE_PYTHON` can point
to another Python 3.8 environment containing PyTorch 1.8.1 and NumPy 1.19.5; use it
when building on another host. The default path is `.build-tools/pt/bin/python`.
The shell setup script targets Linux; other hosts need an equivalent environment.

On this Ubuntu workstation, older Unity also requires local libxml2/ICU compatibility
libraries under `.build-tools/compat`. The embedded Meta XR Core package retains
the editor-only Linux simulator-installer fix from the initial implementation.

## Runtime flow

1. `MainApp` reads camera frames and controller buttons.
2. `ImagePreprocessor` fits a frame into 320 × 320 pixels with RGB 114 padding,
   preserves camera sRGB values, and asynchronously reads back RGBA bytes.
3. `PyTorchWorker` runs Python on one Android background worker. PyTorch uses two
   CPU threads by default; **Cpu Threads** on the model asset changes this.
4. `holocube_runtime.py` flips Unity's bottom-first pixels to top-first RGB,
   normalizes them to 0–1, and runs the transmitter checkpoint.
5. `DetectionFilter` reads `[x1,y1,x2,y2,confidence,class_id]` rows, removes padding,
   and maps boxes to original camera pixels. The transmitter pipeline requests
   only the highest-confidence valid detection (`TransmitterRecognitionModel.MaxDetections = 1`).
   `DetectionOverlay` places its AR label; no box appears below the confidence threshold.

Input is `[1,3,320,320]` float32; output is `[1,300,6]`. The one-to-one head already
selects detections. No second non-maximum suppression step is applied. The final
single-box cap is fixed in the pipeline; the headset settings can adjust confidence
but cannot increase the number of boxes.

## Local file inference

`LocalFileInference` (Settings → right trigger) builds its own world-space panel at
runtime, so the saved scene needs no changes. It lists `.jpg`/`.jpeg`/`.png` files in
`/sdcard/Pictures/HoloCube Test Dataset` (shared storage; `READ_MEDIA_IMAGES`, or
`READ_EXTERNAL_STORAGE` on Android 12, is requested the first time the page opens) and in
`Application.persistentDataPath/HoloCube Test Dataset` (no permission needed). The chosen
file is decoded with `Texture2D.LoadImage` and sent through the same
`TransmitterInference.Detect` path as a camera frame (GPU letterbox, Python worker,
`DetectionFilter`), so the result is what the camera path would produce for that image.
While the page is open `MainApp` cancels the camera detection and advances the file run
instead; leaving the page cancels the file run so the worker is free again. The box is
drawn over the picture with `LocalFileInference.ToPanel`, which the Unity validation checks.

Pausing discards pending results. Native inference finishes safely in the background
before another frame is submitted. Camera pose is captured with each image, and
old results expire. Depth is estimated at two metres when scene depth is unavailable;
such labels show **~**. Boxes do not provide persistent object identity or metric
3D measurements.

## Checkpoints

The supplied file’s provenance, hash, and runtime versions are recorded in
`Assets/HoloCube/Transmitter Model/provenance.json`. The supplied file is committed unchanged beside
the model settings. Do not rename an exported model to `.pt`.

The loader supports only the bundled transmitter checkpoint and checks that its
class mapping is exactly `0: Transmitter`. The general COCO checkpoint is no longer
packaged in the app. For custom trained weights or
another YOLO architecture, update the allowed checkpoint hash, supported layer
methods, class names, and real-image validation together. This sample does not
provide a general-purpose arbitrary-checkpoint loader.

## Validation

- `bash Tools/run-unity.sh Validate`: scene wiring, coordinates, decoder, and
  actual direct-checkpoint inference against three transmitter reference fixtures.
- `bash Tools/run-unity.sh ValidateGPU`: real GPU preprocessing checks for
  padding, orientation, and sRGB. Saves `Builds/pt-preprocessed.rgba`.
- `.build-tools/pt/bin/python Tools/validate-pt.py --rgba Builds/pt-preprocessed.rgba`:
  run the trained transmitter checkpoint on those actual Unity pixels.

The full Python/Java bridge runs on Android. The Unity editor does not run that
bridge; use a Quest to verify startup, live camera detection, and latency.
See [recorded validation](../VALIDATION.md) for the actual test status.

## Sources and licenses

- [Meta camera samples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples),
  original commit `f9c382190907e20232a74c5ecf6513df47bc13b8`.
- [Ultralytics module source, version 8.4.56](https://github.com/ultralytics/ultralytics/tree/v8.4.56/ultralytics/nn).
- [Chaquopy version compatibility](https://chaquo.com/chaquopy/doc/current/versions.html).
- [Android PyTorch package](https://chaquo.com/pypi-13.1/torch/).

Keep the root, Meta sample, and embedded SDK licenses. The model and adapted Python
module methods carry the included AGPL-3.0 license. Python, PyTorch, NumPy, and
Chaquopy retain their upstream licenses. Meta's legacy sample assets and documentation
remain as references; their older exported model is excluded from HoloCube's build scene.

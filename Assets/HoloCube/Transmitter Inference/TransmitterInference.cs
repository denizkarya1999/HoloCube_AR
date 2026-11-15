using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace HoloCube.QuestYOLO
{
    /// <summary>Camera image → background PyTorch worker → labelled detections.</summary>
    public sealed class TransmitterInference : IDisposable
    {
        public Vector2Int ImageSize => images.ImageSize;
        public string[] Labels { get; }
        public bool IsReady => !disposed && worker.Call<bool>("isReady");
        public bool IsBusy { get; private set; }
        /// <summary>Also run the Orientation Detection Model on each image (slower: 640 x 640 input).</summary>
        public bool OrientationEnabled { get; set; } = true;
        /// <summary>True once the orientation checkpoint passed its startup check on this headset.</summary>
        public bool OrientationReady => !disposed && worker.Call<bool>("isOrientationReady");
        /// <summary>Why the orientation model is unavailable, or null.</summary>
        public string OrientationError => disposed ? null : worker.Call<string>("orientationError");
        /// <summary>The orientation found in the latest finished image, or null.</summary>
        public OrientationResult? LastOrientation { get; private set; }
        /// <summary>Whether the orientation model ran on the latest finished image.</summary>
        public bool LastOrientationRan { get; private set; }
        /// <summary>True while the slower orientation pass of the current image is running.</summary>
        public bool OrientationRunning { get; private set; }
        private readonly ImagePreprocessor images;
        private readonly ImagePreprocessor orientationImages;
        private readonly AndroidJavaObject worker;
        private bool disposed;
        private bool loggedFirstDetection;
        private bool loggedFirstOrientation;

        public TransmitterInference(TransmitterRecognitionModel definition)
        {
            Labels = definition.LoadLabels();
#if UNITY_ANDROID && !UNITY_EDITOR
            using var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
            worker = new AndroidJavaObject("com.holocube.pytorch.PyTorchWorker", activity, definition.CpuThreads);
            try
            {
                images = new ImagePreprocessor(definition.LetterboxShader);
                orientationImages = new ImagePreprocessor(definition.LetterboxShader, OrientationDetectionModel.InputSize);
            }
            catch { images?.Dispose(); worker.Call("close"); worker.Dispose(); throw; }
#else
            throw new PlatformNotSupportedException("The embedded .pt runtime runs on Android. Use Tools/validate-pt.py for desktop model checks.");
#endif
        }

        // MainApp advances this iterator every display frame. Python inference
        // runs on a separate Android thread while Unity keeps rendering.
        // The transmitter result is reported first, as soon as it is known; the orientation
        // model (640 x 640, several times slower) then runs on the same captured frame and
        // reports through orientationCompleted, so the transmitter box is not held back.
        public IEnumerator Detect(Texture image, float confidence, Action<List<Detection>> completed,
            Action<OrientationResult?> orientationCompleted = null)
        {
            if (disposed) throw new ObjectDisposedException(nameof(TransmitterInference));
            if (IsBusy) throw new InvalidOperationException("Only one image may be processed at a time.");
            if (image == null) throw new ArgumentNullException(nameof(image));
            IsBusy = true;
            try
            {
                // Copy this frame immediately, preserving its associated camera pose: once for
                // each model's input size, because the camera texture changes while PyTorch runs.
                bool withOrientation = OrientationEnabled && OrientationReady;
                images.Capture(image);
                if (withOrientation) orientationImages.Capture(image);
                // Readback data is only valid for one frame, so copy each as soon as it arrives.
                byte[] pixels = null, orientationPixels = null;
                while (pixels == null || (withOrientation && orientationPixels == null))
                {
                    if (pixels == null && images.IsReady) pixels = images.ReadPixels();
                    if (withOrientation && orientationPixels == null && orientationImages.IsReady)
                        orientationPixels = orientationImages.ReadPixels();
                    if (pixels == null || (withOrientation && orientationPixels == null)) yield return null;
                }
                while (!worker.Call<bool>("isIdle")) yield return null;
                worker.Call("submit", pixels);
                while (!worker.Call<bool>("isDone")) yield return null;
                var rows = worker.Call<float[]>("take");
                // Keep only the strongest valid Transmitter, regardless of scene or UI settings.
                var detections = DetectionFilter.ReadYOLO26(rows, images.Layout, Labels.Length,
                    confidence, TransmitterRecognitionModel.MaxDetections);
                if (!loggedFirstDetection && detections.Count > 0)
                {
                    Debug.Log($"HOLOCUBE_TRANSMITTER_SINGLE_BOX: {detections.Count} detection selected; highest confidence; limit {TransmitterRecognitionModel.MaxDetections}.");
                    loggedFirstDetection = true;
                }
                completed?.Invoke(detections);

                // The orientation model does not wait for a transmitter box: the two models
                // were trained on different recordings and one can find the pyramid alone.
                OrientationResult? orientation = null;
                if (withOrientation)
                {
                    OrientationRunning = true;
                    while (!worker.Call<bool>("isIdle")) yield return null;
                    worker.Call("submitOrientation", orientationPixels);
                    while (!worker.Call<bool>("isDone")) yield return null;
                    var found = OrientationFilter.Read(worker.Call<float[]>("take"), orientationImages.Layout,
                        confidence, OrientationDetectionModel.MaxDetections);
                    if (found.Count > 0) orientation = found[0];
                    if (!loggedFirstOrientation && found.Count > 0)
                    {
                        Debug.Log($"HOLOCUBE_ORIENTATION_SINGLE_TRIANGLE: {found[0].Name} selected; highest confidence; limit {OrientationDetectionModel.MaxDetections}.");
                        loggedFirstOrientation = true;
                    }
                }
                OrientationRunning = false;
                LastOrientation = orientation;
                LastOrientationRan = withOrientation;
                if (withOrientation) orientationCompleted?.Invoke(orientation);
            }
            finally
            {
                // A pause hides the result, but does not interrupt native PyTorch.
                // The worker finishes safely before it accepts another frame.
                if (!disposed) worker.Call("discard");
                OrientationRunning = false;
                IsBusy = false;
            }
        }

        public void Dispose()
        {
            if (disposed) return;
            disposed = true;
            try { worker.Call("close"); }
            finally { worker.Dispose(); images.Dispose(); orientationImages.Dispose(); }
        }
    }
}

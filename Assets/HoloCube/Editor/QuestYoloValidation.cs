using System;
using System.Collections.Generic;
using System.IO;
using HoloCube.QuestYOLO;
using PassthroughCameraSamples.MultiObjectDetection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace HoloCube.Editor
{
    // Editor checks live here so they do not obscure the runtime detection flow.
    public static class QuestYoloValidation
    {
        public static void Run()
        {
            CheckRecordingSource();
            CheckSavedScene();
            CheckCoordinatesAndDecoder();
            CheckRealModel();
            Debug.Log("HOLOCUBE_VALIDATION_PASSED: transmitter scene, letterbox/decoder checks, trained .pt inference matches reference; orientation rows, checkpoint, and reference triangles match.");
        }

        private static void CheckSavedScene()
        {
            if (!File.Exists(QuestYoloSceneSetup.ScenePath)) throw new Exception("Generate the standalone scene first.");
            EditorSceneManager.OpenScene(QuestYoloSceneSetup.ScenePath);
            var app = Object.FindFirstObjectByType<MainApp>();
            Require(app != null && app.CameraAccess != null && app.Model != null &&
                app.Overlay != null && app.StatusLabel != null, "Scene wiring");
            CheckRecordingScene(app);
            var label = app.Overlay.BoxTemplate.GetComponentInChildren<Text>(true);
            Require(label.fontSize == 48 && label.color == Color.white, "Readable labels");
            Require(Object.FindFirstObjectByType<SentisInferenceRunManager>() == null, "Single inference pipeline");
            Require(app.Model.CpuThreads >= 1 && app.Model.CpuThreads <= 4, "Background CPU worker threads");
            Require(app.Model.LetterboxShader != null, "Bundled letterbox shader");
            Require(File.Exists(QuestYoloSceneSetup.ModelFolder + TransmitterRecognitionModel.CheckpointFile), "Trained transmitter checkpoint");
            Require(AssetDatabase.GetAssetPath(app.Model.Labels) == QuestYoloSceneSetup.ModelFolder + "transmitter.names.txt", "Transmitter class name");
        }

        internal static void CheckRecordingSource()
        {
            Require(File.Exists("Assets/Plugins/Android/com/holocube/capture/QuestCameraVideoRecorder.java"),
                "Native video recorder source is missing; sync Assets/Plugins/Android before building");
        }

        internal static void CheckRecordingScene(MainApp app)
        {
            Require(app.StatusLabel != null && app.RecordingIndicator != null && app.RecordingFeedbackLabel != null,
                "Recording indicator and save/error feedback wiring; regenerate the standalone Quest scene");
            var indicatorCanvas = app.RecordingIndicator.GetComponent<Canvas>();
            var recordingDot = app.RecordingIndicator.GetComponentInChildren<RecordingDot>(true);
            Require(recordingDot != null && recordingDot.GetComponent<CanvasRenderer>() != null,
                "Recording circle renderer");
            var feedbackCanvas = app.RecordingFeedbackLabel.GetComponentInParent<Canvas>(true);
            var statusCanvas = app.StatusLabel.GetComponentInParent<Canvas>(true);
            Require(indicatorCanvas != null && indicatorCanvas.enabled &&
                feedbackCanvas != null && feedbackCanvas.enabled, "Recording UI canvases");
            Require(statusCanvas != null &&
                !indicatorCanvas.transform.IsChildOf(statusCanvas.transform) &&
                !feedbackCanvas.transform.IsChildOf(statusCanvas.transform) &&
                !feedbackCanvas.transform.IsChildOf(indicatorCanvas.transform),
                "Recording and save/error feedback must remain visible independently of menus and recording state");
        }

        private static void CheckCoordinatesAndDecoder()
        {
            var landscape = new ImageLetterbox(new Vector2Int(1280, 720), new Vector2Int(320, 320));
            Require(landscape.ResizedSize == new Vector2Int(320, 180) && landscape.Padding == new Vector2Int(0, 70), "Landscape padding");
            Require(landscape.ToOriginalImage(new Vector4(0, 70, 320, 250)) == new Vector4(0, 0, 1280, 720), "Undo landscape padding");
            var portrait = new ImageLetterbox(new Vector2Int(810, 1080), new Vector2Int(320, 320));
            Require(portrait.Padding == new Vector2Int(40, 0), "Portrait padding");
            var odd = new ImageLetterbox(new Vector2Int(320, 181), new Vector2Int(320, 320));
            Require(odd.Padding.y == 69 && Mathf.Approximately(odd.ShaderRectangle.y, 70f / 320), "Asymmetric top/bottom padding");

            var sampleRows = new float[]
            {
                0, 70, 320, 250, 0.8f, 0,  // Deliberately unsorted candidates.
                0, 70, 320, 250, 0.9f, 0,  // End-to-end outputs must not undergo NMS again.
                0, 70, 320, 250, 0.1f, 0,  // Too weak.
                0, 70, 320, 250, 0.99f, 1, // A stronger wrong-class result must not win.
                0, 70, 320, 250, float.NaN, 0,
                10, 80, 0, 70, 0.99f, 0   // A stronger invalid rectangle must not win.
            };
            var output = new float[300 * 6];
            Array.Copy(sampleRows, output, sampleRows.Length);
            var results = DetectionFilter.ReadYOLO26(output, landscape, 1, 0.35f, 24);
            Require(results.Count == 2 && results[0].Box == new Vector4(0, 0, 1280, 720), "Confidence/classes/coordinates; no second NMS");
            var single = DetectionFilter.ReadYOLO26(output, landscape, 1, 0.35f, TransmitterRecognitionModel.MaxDetections);
            Require(single.Count == 1 && single[0].ClassId == 0 && Mathf.Approximately(single[0].Confidence, 0.9f),
                "Only the strongest valid transmitter is shown");
            Require(DetectionFilter.ReadYOLO26(output, landscape, 1, 1, TransmitterRecognitionModel.MaxDetections).Count == 0,
                "No box when no valid transmitter meets the confidence threshold");
            var wrongShape = new float[84 * 2100];
            bool rejected = false;
            try { DetectionFilter.ReadYOLO26(wrongShape, landscape, 1, 0.35f, 24); }
            catch (InvalidOperationException) { rejected = true; }
            Require(rejected, "Reject non-end-to-end model output");

            // In-Device Local File Inference: a 1280 × 720 image fits the 1040 × 560 picture area by
            // height, and a YOLO box maps to the picture's top-left-anchored panel units.
            var fitted = LocalFileInference.FitSize(new Vector2Int(1280, 720), new Vector2(1040, 560));
            Require(Mathf.Approximately(fitted.y, 560f) && Mathf.Abs(fitted.x - 995.556f) < 0.01f, "Local file picture fit");
            var panelBox = LocalFileInference.ToPanel(new Vector4(640, 360, 704, 396), new Vector2Int(1280, 720), fitted);
            Require(Mathf.Abs(panelBox.x - 497.778f) < 0.01f && Mathf.Abs(panelBox.y + 280f) < 0.01f &&
                Mathf.Abs(panelBox.width - 49.778f) < 0.01f && Mathf.Abs(panelBox.height - 28f) < 0.01f,
                "Local file box mapping");
            Require(LocalFileInference.FitSize(new Vector2Int(0, 720), new Vector2(1040, 560)) == Vector2.zero,
                "Local file empty image");
            // The orientation triangle's corners use the same scale as the box.
            var panelPoint = LocalFileInference.ToPanelPoint(new Vector2(640, 360), new Vector2Int(1280, 720), fitted);
            Require(Mathf.Abs(panelPoint.x - 497.778f) < 0.01f && Mathf.Abs(panelPoint.y + 280f) < 0.01f, "Local file triangle corner mapping");
            // In AR a corner keeps its place inside the transmitter's box: the box's top-left
            // pixel is its upper-left corner, its centre is the origin, and y points up.
            var arBox = new Vector4(100, 200, 300, 300);
            var arSize = new Vector2(0.2f, 0.1f);
            Require(DetectionOverlay.ToBox(new Vector2(200, 250), arBox, arSize) == Vector2.zero &&
                Vector2.Distance(DetectionOverlay.ToBox(new Vector2(100, 200), arBox, arSize), new Vector2(-0.1f, 0.05f)) < 1e-5f &&
                Vector2.Distance(DetectionOverlay.ToBox(new Vector2(300, 300), arBox, arSize), new Vector2(0.1f, -0.05f)) < 1e-5f,
                "AR triangle corner mapping");

            // Orientation Detection Model: 5 rows of 12 values in 640 x 640 pixels. A 1920 x 1080
            // frame is scaled by one third with 140-pixel bands above and below.
            var wide = new ImageLetterbox(new Vector2Int(1920, 1080),
                new Vector2Int(OrientationDetectionModel.InputSize, OrientationDetectionModel.InputSize));
            Require(wide.ResizedSize == new Vector2Int(640, 360) && wide.Padding == new Vector2Int(0, 140), "Orientation padding");
            var orientationRows = new float[OrientationDetectionModel.OutputRows * OrientationDetectionModel.OutputColumns];
            Array.Copy(new float[]
            {
                319.2f, 331.7f, 354.5f, 374.3f, 0.5f, 4, 318.5f, 330.5f, 357.5f, 330.5f, 339f, 372.5f,   // South, weaker
                319.2f, 331.7f, 354.5f, 374.3f, 0.917f, 0, 318.5f, 330.5f, 357.5f, 330.5f, 339f, 372.5f, // North
                319.2f, 331.7f, 354.5f, 374.3f, 0.99f, 8, 318.5f, 330.5f, 357.5f, 330.5f, 339f, 372.5f   // No class 8.
            }, orientationRows, 36);
            var orientations = OrientationFilter.Read(orientationRows, wide, 0.35f, OrientationDetectionModel.MaxDetections);
            Require(orientations.Count == 1 && orientations[0].Name == "North" &&
                Mathf.Approximately(orientations[0].Confidence, 0.917f), "Only the strongest valid orientation is kept");
            Require(Mathf.Abs(orientations[0].Box.x - 957.6f) < 0.01f && Mathf.Abs(orientations[0].Box.y - 575.1f) < 0.01f &&
                Mathf.Abs(orientations[0].A.x - 955.5f) < 0.01f && Mathf.Abs(orientations[0].A.y - 571.5f) < 0.01f &&
                Mathf.Abs(orientations[0].B.x - 1072.5f) < 0.01f && Mathf.Abs(orientations[0].C.x - 1017f) < 0.01f &&
                Mathf.Abs(orientations[0].C.y - 697.5f) < 0.01f, "Orientation box and triangle map to camera pixels");
            Require(OrientationFilter.Read(orientationRows, wide, 0.35f, 5).Count == 2, "Class 8 is not an orientation");
            Require(OrientationFilter.Read(orientationRows, wide, 0.95f, 5).Count == 0, "No orientation below the threshold");
            bool rejectedOrientation = false;
            try { OrientationFilter.Read(new float[300 * 6], wide, 0.35f, 1); }
            catch (InvalidOperationException) { rejectedOrientation = true; }
            Require(rejectedOrientation, "Reject transmitter rows as orientation rows");
            // Transmitter first: an orientation counts only inside the transmitter's box
            // (its box centre is about (1011, 639) in camera pixels here).
            var allOrientations = OrientationFilter.Read(orientationRows, wide, 0.35f, 5);
            var around = new List<Detection> { new Detection(0, 0.9f, new Vector4(940, 560, 1090, 720)) };
            var elsewhere = new List<Detection> { new Detection(0, 0.9f, new Vector4(100, 100, 300, 300)) };
            var beside = new Vector4(1030, 560, 1180, 720);
            var inside = OrientationFilter.StrongestInside(allOrientations, around);
            Require(inside.HasValue && inside.Value.Name == "North", "Strongest orientation inside the transmitter box");
            Require(!OrientationFilter.StrongestInside(allOrientations, elsewhere).HasValue, "No orientation outside the transmitter box");
            Require(!OrientationFilter.StrongestInside(allOrientations, new List<Detection>()).HasValue, "No orientation without a transmitter");
            Require(OrientationFilter.InsideBox(allOrientations[0], beside) && !OrientationFilter.InsideBox(allOrientations[0], beside, 0f),
                "Transmitter box margin");
            Require(OrientationDetectionModel.ClassNames.Length == 8 && OrientationDetectionModel.ClassNames[7] == "Northwest",
                "Eight orientation classes");
            Require(File.Exists(PyTorchAndroidBuild.OrientationModelFolder + OrientationDetectionModel.CheckpointFile),
                "Orientation checkpoint file");
        }

        private static void CheckRealModel()
        {
            var definition = AssetDatabase.LoadAssetAtPath<TransmitterRecognitionModel>(QuestYoloSceneSetup.ModelPath);
            Require(definition.LoadLabels().Length == 1 && definition.LoadLabels()[0] == "Transmitter", "Single transmitter class");
            Require(File.Exists(PyTorchAndroidBuild.PythonPath), "Run bash Tools/setup-python.sh first");
            using var process = new System.Diagnostics.Process();
            process.StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = PyTorchAndroidBuild.PythonPath,
                Arguments = "Tools/validate-pt.py",
                WorkingDirectory = Directory.GetCurrentDirectory(),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            process.Start();
            if (!process.WaitForExit(120000))
            {
                process.Kill();
                throw new Exception("Direct .pt model validation timed out.");
            }
            string output = process.StandardOutput.ReadToEnd();
            string errors = process.StandardError.ReadToEnd();
            Require(process.ExitCode == 0 && output.Contains("HOLOCUBE_PT_PARITY_PASSED") &&
                output.Contains("HOLOCUBE_ORIENTATION_PARITY_PASSED"), "Real .pt inference: " + output + errors);
            Debug.Log(output.Trim());
        }

        private static void Require(bool condition, string check)
        {
            if (!condition) throw new Exception("Validation failed: " + check);
        }
    }
}

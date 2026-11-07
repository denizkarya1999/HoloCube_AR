using System;
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
            Debug.Log("HOLOCUBE_VALIDATION_PASSED: transmitter scene, letterbox/decoder checks, trained .pt inference matches reference.");
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
            Require(process.ExitCode == 0 && output.Contains("HOLOCUBE_PT_PARITY_PASSED"), "Real .pt inference: " + output + errors);
            Debug.Log(output.Trim());
        }

        private static void Require(bool condition, string check)
        {
            if (!condition) throw new Exception("Validation failed: " + check);
        }
    }
}

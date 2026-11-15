using System;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using HoloCube.QuestYOLO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace HoloCube.Editor
{
    // Unity menu and command-line entry points. This file never runs on the headset.
    public static class QuestYoloBuild
    {
        [MenuItem("HoloCube/Create standalone Quest scene")]
        public static void CreateScene() => QuestYoloSceneSetup.Create();

        [MenuItem("HoloCube/Validate standalone pipeline")]
        public static void Validate() => QuestYoloValidation.Run();

        public static void ValidateGPU() => QuestYoloGraphicsValidation.Run();

        [MenuItem("HoloCube/Build standalone Quest APK")]
        public static void Build()
        {
            CreateScene();
            Validate();
            Directory.CreateDirectory("Builds");
            EditorUserBuildSettings.buildAppBundle = false;
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { QuestYoloSceneSetup.ScenePath },
                locationPathName = "Builds/HoloCube.apk",
                target = BuildTarget.Android,
                options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new Exception("Quest APK build failed: " + report.summary.result);

            Debug.Log("HOLOCUBE_APK_READY: " + Path.GetFullPath("Builds/HoloCube.apk"));
        }
    }

    // Run these guards for both the HoloCube menu and Unity's standard Build UI.
    public sealed class QuestYoloRecordingBuildGuard : IPreprocessBuildWithReport,
        IProcessSceneWithReport, IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPreprocessBuild(BuildReport report)
        {
            if (report.summary.platform == BuildTarget.Android)
                QuestYoloValidation.CheckRecordingSource();
        }

        public void OnProcessScene(Scene scene, BuildReport report)
        {
            if (report == null || report.summary.platform != BuildTarget.Android ||
                scene.path != QuestYoloSceneSetup.ScenePath) return;

            foreach (var root in scene.GetRootGameObjects())
            {
                var app = root.GetComponentInChildren<MainApp>(true);
                if (app == null) continue;
                QuestYoloValidation.CheckRecordingScene(app);
                return;
            }
            throw new BuildFailedException("HoloCube scene is missing Main App. Regenerate the standalone Quest scene.");
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            if (report.summary.platform != BuildTarget.Android) return;
            string output = report.summary.outputPath;
            // Gradle project exports do not contain compiled DEX files yet.
            if (Directory.Exists(output)) return;

            using var archive = ZipFile.OpenRead(output);
            CheckPackagedModel(archive);
            foreach (var entry in archive.Entries)
            {
                if (!entry.FullName.EndsWith(".dex", StringComparison.OrdinalIgnoreCase)) continue;
                using var stream = entry.Open();
                using var reader = new StreamReader(stream, Encoding.ASCII);
                if (reader.ReadToEnd().Contains("Lcom/holocube/capture/QuestCameraVideoRecorder;"))
                    return;
            }
            throw new BuildFailedException("The Android build is missing QuestCameraVideoRecorder. " +
                "Include Assets/Plugins/Android/com/holocube/capture/QuestCameraVideoRecorder.java before deploying.");
        }

        private static void CheckPackagedModel(ZipArchive archive)
        {
            var python = archive.GetEntry("assets/chaquopy/app.imy") ??
                archive.GetEntry("base/assets/chaquopy/app.imy");
            if (python == null) throw new BuildFailedException("The embedded transmitter runtime is missing.");
            using var buffer = new MemoryStream();
            using (var source = python.Open()) source.CopyTo(buffer);
            buffer.Position = 0;
            using var package = new ZipArchive(buffer, ZipArchiveMode.Read);
            int checkpoints = 0;
            foreach (var entry in package.Entries)
            {
                if (!entry.FullName.EndsWith(".pt", StringComparison.OrdinalIgnoreCase)) continue;
                checkpoints++;
                string expected = entry.FullName == TransmitterRecognitionModel.CheckpointFile
                    ? TransmitterRecognitionModel.CheckpointHash
                    : entry.FullName == OrientationDetectionModel.CheckpointFile
                        ? OrientationDetectionModel.CheckpointHash
                        : throw new BuildFailedException("Unexpected model in APK: " + entry.FullName);
                using var stream = entry.Open();
                using var sha = SHA256.Create();
                string hash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
                if (hash != expected)
                    throw new BuildFailedException("The APK checkpoint " + entry.FullName + " does not match the approved weights.");
            }
            if (checkpoints != 2 || package.GetEntry("reference.f32") == null ||
                package.GetEntry("reference.json") == null ||
                package.GetEntry("orientation_reference.u8") == null ||
                package.GetEntry("orientation_reference.json") == null)
                throw new BuildFailedException("The APK requires the transmitter and orientation checkpoints and their startup references.");
            Debug.Log("HOLOCUBE_TRANSMITTER_PACKAGE_PASSED: only the checksum-verified transmitter model is embedded.");
            Debug.Log("HOLOCUBE_ORIENTATION_PACKAGE_PASSED: the checksum-verified orientation model is embedded beside it.");
        }
    }
}

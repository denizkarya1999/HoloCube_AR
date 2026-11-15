using System;
using System.IO;
using HoloCube.QuestYOLO;
using UnityEditor.Android;

namespace HoloCube.Editor
{
    // Build-time packaging only. All Python code, dependencies, and the original
    // checkpoint are embedded in the APK; the headset needs no Python install.
    public sealed class PyTorchAndroidBuild : IPostGenerateGradleAndroidProject
    {
        public int callbackOrder => 1000;
        public const string OrientationModelFolder = "Assets/HoloCube/Orientation Model/";
        public static string PythonPath => Environment.GetEnvironmentVariable("HOLOCUBE_PYTHON") ??
            Path.GetFullPath(".build-tools/pt/bin/python");

        public void OnPostGenerateGradleAndroidProject(string libraryPath)
        {
            if (!File.Exists(PythonPath)) throw new Exception("Run bash Tools/setup-python.sh before building the .pt app.");
            string root = Directory.GetParent(libraryPath).FullName;
            string rootBuild = Path.Combine(root, "build.gradle");
            string rootText = File.ReadAllText(rootBuild);
            if (!rootText.Contains("com.chaquo.python"))
                File.WriteAllText(rootBuild, rootText.Replace("plugins {", "plugins {\n    id 'com.chaquo.python' version '16.1.0' apply false"));

            string build = Path.Combine(libraryPath, "build.gradle");
            string text = File.ReadAllText(build);
            if (!text.Contains("apply plugin: 'com.chaquo.python'"))
            {
                string python = PythonPath.Replace("\\", "/").Replace("'", "\\'");
                text += "\napply plugin: 'com.chaquo.python'\nchaquopy {\n    defaultConfig {\n" +
                    "        version '3.8'\n        buildPython '" + python + "'\n" +
                    "        pip {\n            install 'torch==1.8.1'\n            install 'numpy==1.19.5'\n        }\n" +
                    "    }\n}\n";
                File.WriteAllText(build, text);
            }

            string pythonDir = Path.Combine(libraryPath, "src/main/python");
            Directory.CreateDirectory(pythonDir);
            // Unity can reuse a generated Gradle project from the general detector.
            // Remove its old weights so incremental builds ship only our trained model.
            string legacyCheckpoint = Path.Combine(pythonDir, "yolo26n.pt");
            if (File.Exists(legacyCheckpoint)) File.Delete(legacyCheckpoint);
            foreach (string file in Directory.GetFiles("Assets/HoloCube/Transmitter Inference/Python", "*.py"))
                File.Copy(file, Path.Combine(pythonDir, Path.GetFileName(file)), true);
            File.Copy(QuestYoloSceneSetup.ModelFolder + TransmitterRecognitionModel.CheckpointFile,
                Path.Combine(pythonDir, TransmitterRecognitionModel.CheckpointFile), true);
            File.Copy(QuestYoloSceneSetup.ModelFolder + "LICENSE.txt", Path.Combine(pythonDir, "YOLO-LICENSE.txt"), true);

            File.Copy("Tools/Fixtures/Transmitter/input.f32", Path.Combine(pythonDir, "reference.f32"), true);
            File.Copy("Tools/Fixtures/Transmitter/expected.json", Path.Combine(pythonDir, "reference.json"), true);

            // The Orientation Detection Model and its bright startup reference frame.
            File.Copy(OrientationModelFolder + OrientationDetectionModel.CheckpointFile,
                Path.Combine(pythonDir, OrientationDetectionModel.CheckpointFile), true);
            File.Copy("Tools/Fixtures/Orientation/bright.u8", Path.Combine(pythonDir, "orientation_reference.u8"), true);
            File.Copy("Tools/Fixtures/Orientation/bright.json", Path.Combine(pythonDir, "orientation_reference.json"), true);

            string javaDir = Path.Combine(libraryPath, "src/main/java/com/holocube/pytorch");
            Directory.CreateDirectory(javaDir);
            File.Copy("Assets/HoloCube/Transmitter Inference/Android/PyTorchWorker.java", Path.Combine(javaDir, "PyTorchWorker.java"), true);
        }
    }
}

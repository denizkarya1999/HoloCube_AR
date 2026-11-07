using System;
using UnityEngine;

namespace HoloCube.QuestYOLO
{
    /// <summary>The trained transmitter checkpoint, label, and CPU settings.</summary>
    [CreateAssetMenu(menuName = "HoloCube/Transmitter Recognition Model")]
    public sealed class TransmitterRecognitionModel : ScriptableObject
    {
        public const string CheckpointFile = "holocube_transmitter.pt";
        public const string CheckpointHash = "1b65b56bb61c0102bb963e34126d1e58cb0c459f0f14dc5a232262ec62d0e260";
        public const int InputSize = 320;
        public const int MaxDetections = 1;

        [Tooltip("One object name per line, in class-ID order.")]
        public TextAsset Labels;
        [Tooltip("Background CPU threads used by PyTorch on the headset.")]
        [Range(1, 4)] public int CpuThreads = 2;
        [Tooltip("Fits the camera image into the model input with gray padding.")]
        public Shader LetterboxShader;

        public string[] LoadLabels()
        {
            if (Labels == null) throw new InvalidOperationException("Assign the class names file.");
            var names = Labels.text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < names.Length; i++) names[i] = names[i].Trim();
            if (names.Length != 1 || names[0] != "Transmitter")
                throw new InvalidOperationException("The Transmitter Recognition Model requires class 0: Transmitter.");
            return names;
        }
    }
}

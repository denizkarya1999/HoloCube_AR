using UnityEngine;

namespace HoloCube.QuestYOLO
{
    /// <summary>The compass direction the pyramid is seen from, with its box and face triangle.</summary>
    public readonly struct OrientationResult
    {
        // Index into OrientationDetectionModel.ClassNames, for example 0 means "North".
        public readonly int ClassId;
        // Confidence from 0 to 1.
        public readonly float Confidence;
        // Pixels in the original camera image from its top-left corner: left, top, right, bottom.
        public readonly Vector4 Box;
        // The triangle over the face showing that direction's cube: top-left, top-right, bottom tip.
        public readonly Vector2 A, B, C;

        public OrientationResult(int classId, float confidence, Vector4 box, Vector2 a, Vector2 b, Vector2 c)
        {
            ClassId = classId;
            Confidence = confidence;
            Box = box;
            A = a;
            B = b;
            C = c;
        }

        public string Name => OrientationDetectionModel.ClassNames[ClassId];
    }
}

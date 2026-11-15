using System;
using System.Collections.Generic;
using UnityEngine;

namespace HoloCube.QuestYOLO
{
    /// <summary>Read the Orientation Detection Model's rows and remove weak or invalid results.</summary>
    public static class OrientationFilter
    {
        // Each row is [left, top, right, bottom, confidence, class ID, ax, ay, bx, by, cx, cy]
        // in model-input pixels, strongest first. The Python runtime already reduced the
        // mask to its triangle; the one-to-one head needs no NMS.
        public static List<OrientationResult> Read(float[] output, ImageLetterbox image, float minimumConfidence, int limit)
        {
            const int columns = OrientationDetectionModel.OutputColumns;
            if (output == null || output.Length != OrientationDetectionModel.OutputRows * columns)
                throw new InvalidOperationException("Expected orientation rows [detections, 12].");

            var results = new List<OrientationResult>();
            if (limit <= 0) return results;
            for (int row = 0; row < OrientationDetectionModel.OutputRows; row++)
            {
                int start = row * columns;
                float confidence = output[start + 4];
                float rawClass = output[start + 5];
                if (!Finite(confidence) || confidence < minimumConfidence || confidence > 1) continue;
                if (!Finite(rawClass) || rawClass < 0 || rawClass >= OrientationDetectionModel.ClassNames.Length ||
                    rawClass != (int)rawClass) continue;
                bool finite = true;
                for (int column = 0; column < columns; column++) finite &= Finite(output[start + column]);
                if (!finite) continue;

                var box = image.ToOriginalImage(new Vector4(output[start], output[start + 1], output[start + 2], output[start + 3]));
                if (box.z <= box.x || box.w <= box.y) continue;
                results.Add(new OrientationResult((int)rawClass, confidence, box,
                    image.ToOriginalPoint(new Vector2(output[start + 6], output[start + 7])),
                    image.ToOriginalPoint(new Vector2(output[start + 8], output[start + 9])),
                    image.ToOriginalPoint(new Vector2(output[start + 10], output[start + 11]))));
            }

            results.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
            if (results.Count > limit) results.RemoveRange(limit, results.Count - limit);
            return results;
        }

        private static bool Finite(float number) => !float.IsNaN(number) && !float.IsInfinity(number);
    }
}

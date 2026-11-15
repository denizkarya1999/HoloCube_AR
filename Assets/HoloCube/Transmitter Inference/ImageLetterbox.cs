using System;
using UnityEngine;

namespace HoloCube.QuestYOLO
{
    /// <summary>Fit an image inside the model input without stretching it.</summary>
    public readonly struct ImageLetterbox
    {
        public Vector2Int ImageSize { get; }
        public Vector2Int ModelSize { get; }
        public Vector2Int ResizedSize { get; }
        public Vector2Int Padding { get; } // Left and top padding in model pixels.
        public float Scale { get; }

        public ImageLetterbox(Vector2Int imageSize, Vector2Int modelSize)
        {
            if (imageSize.x <= 0 || imageSize.y <= 0 || modelSize.x <= 0 || modelSize.y <= 0)
                throw new ArgumentException("Image dimensions must be positive.");
            ImageSize = imageSize;
            ModelSize = modelSize;
            Scale = Mathf.Min((float)modelSize.x / imageSize.x, (float)modelSize.y / imageSize.y);
            ResizedSize = new Vector2Int(Mathf.RoundToInt(imageSize.x * Scale), Mathf.RoundToInt(imageSize.y * Scale));
            Padding = new Vector2Int((modelSize.x - ResizedSize.x) / 2, (modelSize.y - ResizedSize.y) / 2);
        }

        public Vector4 ShaderRectangle
        {
            get
            {
                // Shader UVs start at bottom-left; detection coordinates start at top-left.
                int bottom = ModelSize.y - ResizedSize.y - Padding.y;
                return new Vector4((float)Padding.x / ModelSize.x, (float)bottom / ModelSize.y,
                    (float)ResizedSize.x / ModelSize.x, (float)ResizedSize.y / ModelSize.y);
            }
        }

        public Vector4 ToOriginalImage(Vector4 box)
        {
            return new Vector4(
                Mathf.Clamp((box.x - Padding.x) / Scale, 0, ImageSize.x),
                Mathf.Clamp((box.y - Padding.y) / Scale, 0, ImageSize.y),
                Mathf.Clamp((box.z - Padding.x) / Scale, 0, ImageSize.x),
                Mathf.Clamp((box.w - Padding.y) / Scale, 0, ImageSize.y));
        }

        public Vector2 ToOriginalPoint(Vector2 point)
        {
            return new Vector2(
                Mathf.Clamp((point.x - Padding.x) / Scale, 0, ImageSize.x),
                Mathf.Clamp((point.y - Padding.y) / Scale, 0, ImageSize.y));
        }
    }
}

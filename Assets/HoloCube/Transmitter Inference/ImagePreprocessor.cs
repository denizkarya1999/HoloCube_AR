using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace HoloCube.QuestYOLO
{
    /// <summary>Capture a small padded image and copy its RGBA bytes off the GPU.</summary>
    public sealed class ImagePreprocessor : IDisposable
    {
        public ImageLetterbox Layout { get; private set; }
        public Vector2Int ImageSize { get; private set; }
        private readonly Material material;
        private readonly RenderTexture target;
        private AsyncGPUReadbackRequest readback;
        private bool requested;
        private readonly int size;

        public ImagePreprocessor(Shader shader, int size = TransmitterRecognitionModel.InputSize)
        {
            this.size = size;
            if (shader == null) throw new InvalidOperationException("Assign the letterbox shader.");
            if (!SystemInfo.supportsAsyncGPUReadback)
                throw new NotSupportedException("Camera preprocessing requires GPU readback support.");
            material = new Material(shader);
            target = new RenderTexture(size, size, 0,
                RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            target.Create();
        }

        public void Capture(Texture image)
        {
            FinishReadback();
            ImageSize = new Vector2Int(image.width, image.height);
            Layout = new ImageLetterbox(ImageSize, new Vector2Int(size, size));
            material.SetVector("_ImageRect", Layout.ShaderRectangle);
            material.SetFloat("_SourceIsSRGB", GraphicsFormatUtility.IsSRGBFormat(image.graphicsFormat) ? 1 : 0);
            var oldTarget = RenderTexture.active;
            bool oldSrgbWrite = GL.sRGBWrite;
            try
            {
                GL.sRGBWrite = false;
                Graphics.Blit(image, target, material);
            }
            finally { RenderTexture.active = oldTarget; GL.sRGBWrite = oldSrgbWrite; }
            readback = AsyncGPUReadback.Request(target, 0, TextureFormat.RGBA32);
            requested = true;
        }

        public bool IsReady => requested && readback.done;

        public byte[] ReadPixels()
        {
            if (!IsReady || readback.hasError) throw new InvalidOperationException("Camera image readback failed.");
            return readback.GetData<byte>().ToArray();
        }

        private void FinishReadback()
        {
            if (requested && !readback.done) readback.WaitForCompletion();
            requested = false;
        }

        public void Dispose()
        {
            FinishReadback();
            target.Release();
            if (Application.isPlaying) { UnityEngine.Object.Destroy(target); UnityEngine.Object.Destroy(material); }
            else { UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(material); }
        }
    }
}

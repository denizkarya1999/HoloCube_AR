using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

namespace HoloCube.QuestYOLO
{
    /// <summary>
    /// Settings → In-Device Local File Inference: run the Transmitter Detection Model on an
    /// image stored on the headset and show the annotated image, the box in image pixels,
    /// its confidence, and the inference time. Images come from the shared
    /// Pictures/HoloCube Test Dataset folder (after the image permission is granted) and from
    /// this app's own files folder of the same name. The panel is built at runtime, so the
    /// saved scene needs no changes. Only the Transmitter Detection Model exists today.
    /// </summary>
    public sealed class LocalFileInference : IDisposable
    {
        public const string FolderName = "HoloCube Test Dataset";
        public const string SharedFolder = "/sdcard/Pictures/" + FolderName;
        public static readonly string[] Extensions = { ".jpg", ".jpeg", ".png" };
        public const string ModelName = "Transmitter Detection Model";

        private static readonly Color Maize = new Color32(255, 203, 5, 255);
        private static readonly Color MichiganBlue = new Color32(0, 39, 76, 255);
        private static readonly Vector2 PanelSize = new Vector2(1120, 820);
        private static readonly Vector2 ImageArea = new Vector2(1040, 560);
        private const float ImageAreaCenterY = 20f;

        private readonly Func<TransmitterInference> engine;
        private readonly Func<float> confidence;
        private readonly GameObject root;
        private readonly RawImage picture;
        private readonly DialogPanel boxOutline;
        private readonly Text boxLabel;
        private readonly Text status;
        private readonly Text detail;
        private readonly List<string> files = new List<string>();
        private int index = -1;
        private Texture2D texture;
        private IEnumerator running;
        private bool runQueued;
        private bool permissionRequested;
        private string loadError;

        public bool IsRunning => running != null || runQueued;
        public int ImageCount => files.Count;
        public string CurrentFile => index >= 0 && index < files.Count ? files[index] : null;
        public List<Detection> LastDetections { get; private set; }

        public LocalFileInference(Transform head, Func<TransmitterInference> engine, Func<float> confidence)
        {
            this.engine = engine ?? throw new ArgumentNullException(nameof(engine));
            this.confidence = confidence ?? throw new ArgumentNullException(nameof(confidence));

            root = new GameObject("Local File Inference", typeof(RectTransform), typeof(Canvas));
            root.transform.SetParent(head, false);
            root.transform.localPosition = new Vector3(0f, 0f, 1.25f);
            root.transform.localScale = Vector3.one * 0.001f;
            var canvas = root.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 90;
            root.GetComponent<RectTransform>().sizeDelta = PanelSize;
            CreateCard(root.transform);

            CreateText(root.transform, "Heading", "In-Device Local File Inference",
                new Vector2(0, 370), new Vector2(1040, 56), 34, TextAnchor.MiddleCenter, true);
            CreateText(root.transform, "Model", ModelName + " · YOLO26n · one class: Transmitter (the central pyramid)",
                new Vector2(0, 324), new Vector2(1040, 40), 24, TextAnchor.MiddleCenter);

            var pictureObject = new GameObject("Picture", typeof(RectTransform), typeof(RawImage));
            pictureObject.transform.SetParent(root.transform, false);
            picture = pictureObject.GetComponent<RawImage>();
            picture.rectTransform.anchorMin = picture.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            picture.rectTransform.anchoredPosition = new Vector2(0, ImageAreaCenterY);
            picture.rectTransform.sizeDelta = ImageArea;
            picture.color = Color.white;
            picture.raycastTarget = false;
            picture.enabled = false;

            // The box and its label live inside the picture, anchored to its top-left corner,
            // so image pixels map to panel units with one scale factor.
            var outlineObject = new GameObject("Box", typeof(RectTransform), typeof(DialogPanel));
            outlineObject.transform.SetParent(picture.transform, false);
            boxOutline = outlineObject.GetComponent<DialogPanel>();
            boxOutline.rectTransform.anchorMin = boxOutline.rectTransform.anchorMax = new Vector2(0, 1);
            boxOutline.rectTransform.pivot = new Vector2(0, 1);
            boxOutline.color = Color.clear;
            boxOutline.CornerRadius = 0f;
            boxOutline.BorderWidth = 4f;
            boxOutline.BorderColor = Maize;
            boxOutline.raycastTarget = false;
            outlineObject.SetActive(false);

            var labelObject = new GameObject("Box Label", typeof(RectTransform), typeof(Text));
            labelObject.transform.SetParent(picture.transform, false);
            boxLabel = labelObject.GetComponent<Text>();
            boxLabel.rectTransform.anchorMin = boxLabel.rectTransform.anchorMax = new Vector2(0, 1);
            boxLabel.rectTransform.pivot = new Vector2(0, 0);
            boxLabel.rectTransform.sizeDelta = new Vector2(400, 40);
            boxLabel.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            boxLabel.fontSize = 28;
            boxLabel.fontStyle = FontStyle.Bold;
            boxLabel.alignment = TextAnchor.LowerLeft;
            boxLabel.horizontalOverflow = HorizontalWrapMode.Overflow;
            boxLabel.color = Maize;
            boxLabel.raycastTarget = false;
            var labelOutline = labelObject.AddComponent<Outline>();
            labelOutline.effectColor = MichiganBlue;
            labelOutline.effectDistance = new Vector2(2, -2);
            labelObject.SetActive(false);

            status = CreateText(root.transform, "Status", string.Empty,
                new Vector2(0, -300), new Vector2(1040, 48), 26, TextAnchor.MiddleCenter, true);
            detail = CreateText(root.transform, "Detail", string.Empty,
                new Vector2(0, -345), new Vector2(1040, 44), 24, TextAnchor.MiddleCenter);
            CreateText(root.transform, "Help",
                "Joystick left/right: previous / next image · A: run again · Y: back to Settings",
                new Vector2(0, -388), new Vector2(1040, 40), 22, TextAnchor.MiddleCenter);

            root.SetActive(false);
        }

        /// <summary>Shows or hides the panel; MainApp calls this from its menu logic.</summary>
        public void Show(bool visible)
        {
            if (root.activeSelf != visible) root.SetActive(visible);
        }

        /// <summary>Called when the page opens: find images, ask for the permission once, run the first.</summary>
        public void Open()
        {
            if (!permissionRequested)
            {
                permissionRequested = true;
                RequestImagePermission();
            }
            if (files.Count == 0) Scan();
            if (index < 0 && files.Count > 0) Select(0);
            else RefreshText();
        }

        /// <summary>Moves to the previous (-1) or next (+1) image and runs the model on it.</summary>
        public void Step(int delta)
        {
            if (files.Count == 0)
            {
                Scan();
                if (files.Count == 0) { RefreshText(); return; }
            }
            int next = index < 0 ? 0 : ((index + delta) % files.Count + files.Count) % files.Count;
            Select(next);
        }

        /// <summary>Runs the model again on the current image (for example after a threshold change).</summary>
        public void RunAgain()
        {
            if (index < 0) { Step(1); return; }
            Cancel();
            runQueued = true;
            RefreshText();
        }

        /// <summary>Advances the pending run by one step; MainApp calls this every frame while the page is open.</summary>
        public void Advance()
        {
            if (runQueued)
            {
                var detector = engine();
                if (detector == null || !detector.IsReady || detector.IsBusy || texture == null) return;
                runQueued = false;
                running = Run(detector);
            }
            if (running == null) return;
            bool more;
            try { more = running.MoveNext(); }
            catch (Exception error)
            {
                running = null;
                loadError = error.Message;
                RefreshText();
                return;
            }
            if (!more) running = null;
        }

        /// <summary>Stops a pending run so the camera path can use the worker again.</summary>
        public void Cancel()
        {
            runQueued = false;
            var pending = running;
            running = null;
            (pending as IDisposable)?.Dispose();
        }

        private IEnumerator Run(TransmitterInference detector)
        {
            float started = Time.realtimeSinceStartup;
            List<Detection> result = null;
            var detection = detector.Detect(texture, confidence(), detections => result = detections);
            while (detection.MoveNext()) yield return null;
            float elapsed = Time.realtimeSinceStartup - started;
            LastDetections = result ?? new List<Detection>();
            lastRunMillis = Mathf.RoundToInt(elapsed * 1000f);
            lastRunThreshold = confidence();
            DrawBox(LastDetections);
            RefreshText();
        }

        private int lastRunMillis = -1;
        private float lastRunThreshold;

        private void Select(int newIndex)
        {
            Cancel();
            index = newIndex;
            LastDetections = null;
            lastRunMillis = -1;
            loadError = null;
            DrawBox(null);
            if (LoadTexture(files[index])) runQueued = true;
            RefreshText();
        }

        private bool LoadTexture(string path)
        {
            try
            {
                byte[] bytes = File.ReadAllBytes(path);
                var loaded = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                if (!loaded.LoadImage(bytes))
                {
                    UnityEngine.Object.Destroy(loaded);
                    throw new InvalidDataException("Unsupported or unreadable image");
                }
                loaded.wrapMode = TextureWrapMode.Clamp;
                if (texture != null) UnityEngine.Object.Destroy(texture);
                texture = loaded;
                picture.texture = texture;
                picture.rectTransform.sizeDelta = FitSize(new Vector2Int(texture.width, texture.height), ImageArea);
                picture.enabled = true;
                return true;
            }
            catch (Exception error)
            {
                Debug.LogWarning($"HOLOCUBE_LOCAL_FILE_FAILED: {path}: {error.Message}");
                loadError = error.Message;
                picture.enabled = false;
                return false;
            }
        }

        private void DrawBox(List<Detection> detections)
        {
            bool show = detections != null && detections.Count > 0 && texture != null;
            if (boxOutline.gameObject.activeSelf != show) boxOutline.gameObject.SetActive(show);
            if (boxLabel.gameObject.activeSelf != show) boxLabel.gameObject.SetActive(show);
            if (!show) return;

            var detection = detections[0];
            var imageSize = new Vector2Int(texture.width, texture.height);
            Rect box = ToPanel(detection.Box, imageSize, picture.rectTransform.sizeDelta);
            boxOutline.rectTransform.anchoredPosition = new Vector2(box.x, box.y);
            boxOutline.rectTransform.sizeDelta = new Vector2(box.width, box.height);
            boxOutline.SetVerticesDirty();
            boxLabel.rectTransform.anchoredPosition = new Vector2(box.x, box.y + 6f);
            boxLabel.text = $"Transmitter {detection.Confidence:P0}";
        }

        private void RefreshText()
        {
            if (files.Count == 0)
            {
                status.text = "No images found";
                detail.text = $"Copy JPEG or PNG files to {SharedFolder} (allow image access) or to " +
                    $"Android/data/{Application.identifier}/files/{FolderName}, then press A";
                return;
            }
            string name = Path.GetFileName(files[index]);
            string position = $"({index + 1}/{files.Count})";
            if (loadError != null)
            {
                status.text = $"{name} {position}";
                detail.text = "Could not open the image: " + loadError;
                return;
            }
            string size = texture != null ? $"{texture.width} × {texture.height}" : "";
            if (IsRunning || lastRunMillis < 0)
            {
                status.text = $"{name} {position} · {size}";
                detail.text = IsRunning ? "Running the model…" : "Press A to run the model";
                return;
            }
            var detections = LastDetections;
            if (detections == null || detections.Count == 0)
            {
                status.text = $"{name} {position} · {size} · no transmitter above {lastRunThreshold * 100f:0}% · {lastRunMillis} ms";
                detail.text = string.Empty;
                return;
            }
            var detection = detections[0];
            status.text = $"{name} {position} · {size} · {detections.Count} transmitter · {lastRunMillis} ms";
            detail.text = $"Transmitter {detection.Confidence:P0} · box left {detection.Box.x:0}, top {detection.Box.y:0}, " +
                $"right {detection.Box.z:0}, bottom {detection.Box.w:0} px";
        }

        private void Scan()
        {
            files.Clear();
            foreach (string folder in Folders())
            {
                try
                {
                    if (!Directory.Exists(folder)) continue;
                    foreach (string file in Directory.GetFiles(folder))
                        if (Array.IndexOf(Extensions, Path.GetExtension(file).ToLowerInvariant()) >= 0)
                            files.Add(file);
                }
                catch (Exception error)
                {
                    Debug.LogWarning($"HOLOCUBE_LOCAL_FILE_FOLDER: {folder}: {error.Message}");
                }
            }
            files.Sort(StringComparer.OrdinalIgnoreCase);
            if (index >= files.Count) index = -1;
            Debug.Log($"HOLOCUBE_LOCAL_FILES: {files.Count} images found");
        }

        private static IEnumerable<string> Folders()
        {
            yield return SharedFolder;
            yield return Path.Combine(Application.persistentDataPath, FolderName);
        }

        private void RequestImagePermission()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            string permission = ImagePermission();
            if (UnityEngine.Android.Permission.HasUserAuthorizedPermission(permission)) return;
            var callbacks = new UnityEngine.Android.PermissionCallbacks();
            callbacks.PermissionGranted += _ =>
            {
                Scan();
                if (index < 0 && files.Count > 0) Select(0); else RefreshText();
            };
            callbacks.PermissionDenied += _ => RefreshText();
            UnityEngine.Android.Permission.RequestUserPermission(permission, callbacks);
#endif
        }

        /// <summary>Android 13+ reads shared images with READ_MEDIA_IMAGES; Android 12 with READ_EXTERNAL_STORAGE.</summary>
        public static string ImagePermission()
        {
            int sdk = 0;
#if UNITY_ANDROID && !UNITY_EDITOR
            using (var version = new AndroidJavaClass("android.os.Build$VERSION"))
                sdk = version.GetStatic<int>("SDK_INT");
#endif
            return sdk >= 33 ? "android.permission.READ_MEDIA_IMAGES" : "android.permission.READ_EXTERNAL_STORAGE";
        }

        /// <summary>The largest size with the image's aspect ratio that fits inside <paramref name="area"/>.</summary>
        public static Vector2 FitSize(Vector2Int image, Vector2 area)
        {
            if (image.x <= 0 || image.y <= 0) return Vector2.zero;
            float scale = Mathf.Min(area.x / image.x, area.y / image.y);
            return new Vector2(image.x * scale, image.y * scale);
        }

        /// <summary>
        /// A YOLO box (left, top, right, bottom in image pixels, top-left origin) as a panel
        /// rectangle anchored to the picture's top-left corner: x to the right, y downwards (negative).
        /// </summary>
        public static Rect ToPanel(Vector4 box, Vector2Int image, Vector2 fitted)
        {
            float scale = image.x > 0 ? fitted.x / image.x : 0f;
            return new Rect(box.x * scale, -box.y * scale, (box.z - box.x) * scale, (box.w - box.y) * scale);
        }

        public void Dispose()
        {
            Cancel();
            if (texture != null) UnityEngine.Object.Destroy(texture);
            texture = null;
            if (root != null) UnityEngine.Object.Destroy(root);
        }

        // --- Runtime copies of the editor's dialog styling (QuestYoloSceneSetup) ---

        private static void CreateCard(Transform panel)
        {
            CreateCardLayer(panel, "Card Shadow", new Color(0, 0, 0, 0.25f), 27, 0,
                Color.clear, new Vector2(0, -8), new Vector2(6, 6));
            CreateCardLayer(panel, "Michigan Blue Card", MichiganBlue, 24, 2,
                WithAlpha(Maize, 0.5f), Vector2.zero, Vector2.zero);
            var accentObject = new GameObject("Maize Accent", typeof(RectTransform), typeof(Image));
            accentObject.transform.SetParent(panel, false);
            var accent = accentObject.GetComponent<Image>();
            accent.rectTransform.anchorMin = accent.rectTransform.anchorMax = new Vector2(0.5f, 1);
            accent.rectTransform.anchoredPosition = new Vector2(0, -2);
            accent.rectTransform.sizeDelta = new Vector2(140, 2);
            accent.color = WithAlpha(Maize, 0.9f);
            accent.raycastTarget = false;
        }

        private static void CreateCardLayer(Transform panel, string name, Color fill, float radius,
            float borderWidth, Color borderColor, Vector2 position, Vector2 size)
        {
            var layerObject = new GameObject(name, typeof(RectTransform), typeof(DialogPanel));
            layerObject.transform.SetParent(panel, false);
            var layer = layerObject.GetComponent<DialogPanel>();
            layer.rectTransform.anchorMin = Vector2.zero;
            layer.rectTransform.anchorMax = Vector2.one;
            layer.rectTransform.sizeDelta = size;
            layer.rectTransform.anchoredPosition = position;
            layer.color = fill;
            layer.CornerRadius = radius;
            layer.BorderWidth = borderWidth;
            layer.BorderColor = borderColor;
            layer.raycastTarget = false;
        }

        private static Text CreateText(Transform parent, string name, string value, Vector2 position,
            Vector2 size, int fontSize, TextAnchor alignment, bool bold = false)
        {
            var textObject = new GameObject(name, typeof(RectTransform), typeof(Text));
            textObject.transform.SetParent(parent, false);
            var text = textObject.GetComponent<Text>();
            text.rectTransform.anchorMin = text.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            text.rectTransform.anchoredPosition = position;
            text.rectTransform.sizeDelta = size;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.fontStyle = bold ? FontStyle.Bold : FontStyle.Normal;
            text.alignment = alignment;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Overflow;
            text.color = Maize;
            text.raycastTarget = false;
            text.text = value;
            return text;
        }

        private static Color WithAlpha(Color color, float alpha) => new Color(color.r, color.g, color.b, alpha);
    }
}

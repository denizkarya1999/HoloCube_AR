using HoloCube.QuestYOLO;
using Meta.XR;
using PassthroughCameraSamples.MultiObjectDetection;
using PassthroughCameraSamples.StartScene;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.UI;
using Object = UnityEngine.Object;

namespace HoloCube.Editor
{
    // Editor-only setup: reuse Meta's XR rig and connect our three-part pipeline.
    public static class QuestYoloSceneSetup
    {
        public const string ScenePath = "Assets/HoloCube/QuestYOLO.unity";
        public const string ModelFolder = "Assets/HoloCube/Transmitter Model/";
        public const string ModelPath = ModelFolder + "TransmitterRecognition.asset";
        private const string SampleScene = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/MultiObjectDetection.unity";
        private static readonly Color DialogTextColor = new Color32(255, 203, 5, 255);
        private static readonly Color DialogBackgroundColor = new Color32(0, 39, 76, 255);

        public static void Create()
        {
            var scene = EditorSceneManager.OpenScene(SampleScene);
            var camera = Object.FindFirstObjectByType<PassthroughCameraAccess>();
            var overlay = CreateOverlay(camera);
            var app = new GameObject("Main App").AddComponent<MainApp>();
            app.CameraAccess = camera;
            app.Model = ConfigureModel();
            app.Overlay = overlay;
            var rig = Object.FindFirstObjectByType<OVRCameraRig>();
            app.StatusLabel = CreateStatus(rig.centerEyeAnchor, out var statusIcon);
            app.StatusIcon = statusIcon;
            app.StatusIconContainer = statusIcon.transform.parent.gameObject;
            app.AboutLogo = CreateAboutLogo(app.StatusLabel.transform.parent);
            app.SettingsView = CreateSettings(app.StatusLabel.transform.parent);
            app.RecordingIndicator = CreateRecordingIndicator(rig.centerEyeAnchor);
            app.RecordingFeedbackLabel = CreateRecordingFeedback(rig.centerEyeAnchor, out var feedbackIcon);
            app.RecordingFeedbackIcon = feedbackIcon;
            new GameObject("Passthrough").AddComponent<OVRPassthroughLayer>();

            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            ConfigureAndroidBuild();
            AssetDatabase.SaveAssets();
            Debug.Log("HOLOCUBE_SCENE_READY: " + ScenePath);
        }

        private static DetectionOverlay CreateOverlay(PassthroughCameraAccess camera)
        {
            // Keep Meta's camera/depth setup and box template; replace its demo logic.
            var oldUi = Object.FindFirstObjectByType<SentisInferenceUiManager>();
            var template = (RectTransform)new SerializedObject(oldUi).FindProperty("m_detectionBoxPrefab").objectReferenceValue;
            var overlayObject = oldUi.gameObject;
            Object.DestroyImmediate(oldUi);
            RemoveSampleBehaviours();

            overlayObject.name = "AR Detection Overlay";
            var overlay = overlayObject.AddComponent<DetectionOverlay>();
            overlay.CameraAccess = camera;
            overlay.Depth = Object.FindFirstObjectByType<EnvironmentRaycastManager>();
            overlay.BoxTemplate = template;
            MakeLabelsReadable(template);
            return overlay;
        }

        private static void RemoveSampleBehaviours()
        {
            foreach (var component in Object.FindObjectsByType<SentisInferenceRunManager>(FindObjectsSortMode.None))
                Object.DestroyImmediate(component);
            foreach (var component in Object.FindObjectsByType<DetectionManager>(FindObjectsSortMode.None))
                Object.DestroyImmediate(component.gameObject);
            foreach (var component in Object.FindObjectsByType<DetectionUiMenuManager>(FindObjectsSortMode.None))
                Object.DestroyImmediate(component.gameObject);
            foreach (var component in Object.FindObjectsByType<ReturnToStartScene>(FindObjectsSortMode.None))
                Object.DestroyImmediate(component.gameObject);
            foreach (var component in Object.FindObjectsByType<EnvironmentRayCastSampleManager>(FindObjectsSortMode.None))
                Object.DestroyImmediate(component);
        }

        private static TransmitterRecognitionModel ConfigureModel()
        {
            var model = AssetDatabase.LoadAssetAtPath<TransmitterRecognitionModel>(ModelPath);
            if (model == null)
            {
                model = ScriptableObject.CreateInstance<TransmitterRecognitionModel>();
                AssetDatabase.CreateAsset(model, ModelPath);
            }
            model.Labels = AssetDatabase.LoadAssetAtPath<TextAsset>(ModelFolder + "transmitter.names.txt");
            model.CpuThreads = 2;
            model.LetterboxShader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/HoloCube/Transmitter Inference/Letterbox.shader");
            EditorUtility.SetDirty(model);
            return model;
        }

        private static void MakeLabelsReadable(RectTransform template)
        {
            template.gameObject.SetActive(false);
            var label = template.GetComponentInChildren<Text>(true);
            label.rectTransform.localScale = Vector3.one * 0.0007f;
            label.rectTransform.sizeDelta = new Vector2(700, 90);
            label.fontSize = 48;
            label.color = Color.white;
            label.horizontalOverflow = HorizontalWrapMode.Overflow;
            label.verticalOverflow = VerticalWrapMode.Overflow;
            PrefabUtility.RecordPrefabInstancePropertyModifications(label);
            PrefabUtility.RecordPrefabInstancePropertyModifications(label.rectTransform);
            var outline = label.gameObject.AddComponent<Outline>();
            outline.effectColor = Color.black;
            outline.effectDistance = new Vector2(2, -2);
        }

        private static void ConfigureAndroidBuild()
        {
            PlayerSettings.companyName = "HoloCube";
            PlayerSettings.productName = "HoloCube";
            PlayerSettings.bundleVersion = "0.9";
            PlayerSettings.Android.bundleVersionCode = 9;
            PlayerSettings.SetApplicationIdentifier(UnityEditor.Build.NamedBuildTarget.Android, "com.developer27.holocube");
            PlayerSettings.SetScriptingBackend(UnityEditor.Build.NamedBuildTarget.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevel35;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
            PlayerSettings.Android.useCustomKeystore = false;
        }

        private static GameObject CreateRecordingIndicator(Transform head)
        {
            var canvasObject = new GameObject("Recording Indicator", typeof(RectTransform), typeof(Canvas));
            canvasObject.transform.SetParent(head, false);
            canvasObject.transform.localPosition = new Vector3(0f, 0.28f, 1.25f);
            canvasObject.transform.localScale = Vector3.one * 0.001f;

            var canvas = canvasObject.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 100;
            canvasObject.GetComponent<RectTransform>().sizeDelta = new Vector2(420, 120);
            CreateDialogCard(canvasObject.transform, 18);

            var dotObject = new GameObject("Red Circle", typeof(RectTransform), typeof(RecordingDot));
            dotObject.transform.SetParent(canvasObject.transform, false);
            var dot = dotObject.GetComponent<RecordingDot>();
            dot.rectTransform.anchorMin = dot.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            dot.rectTransform.sizeDelta = new Vector2(26, 26);
            dot.rectTransform.anchoredPosition = new Vector2(-115, 20);
            dot.color = new Color(1f, 0.12f, 0.12f, 1f);
            dot.raycastTarget = false;

            var labelObject = new GameObject("Recording Label", typeof(RectTransform), typeof(Text));
            labelObject.transform.SetParent(canvasObject.transform, false);
            var label = labelObject.GetComponent<Text>();
            label.rectTransform.anchorMin = label.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            label.rectTransform.sizeDelta = new Vector2(250, 55);
            label.rectTransform.anchoredPosition = new Vector2(45, 20);
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.fontSize = 36;
            label.alignment = TextAnchor.MiddleLeft;
            label.color = DialogTextColor;
            label.raycastTarget = false;
            label.text = "Recording";

            var destinationObject = new GameObject("Video Folder", typeof(RectTransform), typeof(Text));
            destinationObject.transform.SetParent(canvasObject.transform, false);
            var destination = destinationObject.GetComponent<Text>();
            destination.rectTransform.sizeDelta = new Vector2(390, 40);
            destination.rectTransform.anchoredPosition = new Vector2(0, -27);
            destination.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            destination.fontSize = 26;
            destination.alignment = TextAnchor.MiddleCenter;
            destination.color = DialogTextColor;
            destination.raycastTarget = false;
            destination.text = "Movies/HoloCube";

            canvasObject.SetActive(false);
            return canvasObject;
        }

        private static Text CreateRecordingFeedback(Transform head, out DialogIcon icon)
        {
            var canvasObject = new GameObject("Recording Feedback", typeof(RectTransform), typeof(Canvas));
            canvasObject.transform.SetParent(head, false);
            canvasObject.transform.localPosition = new Vector3(0f, 0.28f, 1.25f);
            canvasObject.transform.localScale = Vector3.one * 0.001f;
            var canvas = canvasObject.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 100;
            canvasObject.GetComponent<RectTransform>().sizeDelta = new Vector2(1000, 120);
            CreateDialogCard(canvasObject.transform, 24);
            icon = CreateDialogIcon(canvasObject.transform, DialogIcon.Kind.Saving);
            var labelObject = new GameObject("Save Status", typeof(RectTransform), typeof(Text));
            labelObject.transform.SetParent(canvasObject.transform, false);
            var label = labelObject.GetComponent<Text>();
            label.rectTransform.anchorMin = Vector2.zero;
            label.rectTransform.anchorMax = Vector2.one;
            label.rectTransform.offsetMin = new Vector2(104, 10);
            label.rectTransform.offsetMax = new Vector2(-24, -10);
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.fontSize = 28;
            label.alignment = TextAnchor.MiddleCenter;
            label.color = DialogTextColor;
            label.raycastTarget = false;
            canvasObject.SetActive(false);
            return label;
        }

        private static Text CreateStatus(Transform head, out DialogIcon icon)
        {
            var canvasObject = new GameObject("Detection Status", typeof(RectTransform), typeof(Canvas));
            canvasObject.transform.SetParent(head, false);
            canvasObject.transform.localPosition = new Vector3(0, -0.30f, 1.25f);
            canvasObject.transform.localScale = Vector3.one * 0.001f;
            var canvas = canvasObject.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            var rect = canvasObject.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(1120, 400);
            CreateDialogCard(canvasObject.transform, 24);
            icon = CreateDialogIcon(canvasObject.transform, DialogIcon.Kind.Cube);
            var textObject = new GameObject("Status", typeof(RectTransform), typeof(Text));
            textObject.transform.SetParent(canvasObject.transform, false);
            var text = textObject.GetComponent<Text>();
            text.rectTransform.anchorMin = Vector2.zero;
            text.rectTransform.anchorMax = Vector2.one;
            text.rectTransform.offsetMin = new Vector2(104, 20);
            text.rectTransform.offsetMax = new Vector2(-24, -20);
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = 28;
            text.supportRichText = true;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = DialogTextColor;
            text.raycastTarget = false;
            text.text = "HoloCube · Loading object detection…";
            return text;
        }

        private static GameObject CreateAboutLogo(Transform panel)
        {
            var logoObject = new GameObject("About HoloCube Logo", typeof(RectTransform));
            logoObject.transform.SetParent(panel, false);
            var logo = logoObject.GetComponent<RectTransform>();
            logo.anchorMin = logo.anchorMax = new Vector2(0.5f, 1);
            logo.anchoredPosition = new Vector2(0, -64);
            logo.sizeDelta = new Vector2(440, 100);

            var markObject = new GameObject("HoloCube Mark", typeof(RectTransform), typeof(DialogIcon));
            markObject.transform.SetParent(logo, false);
            var mark = markObject.GetComponent<DialogIcon>();
            mark.rectTransform.anchorMin = mark.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            mark.rectTransform.anchoredPosition = new Vector2(-110, 0);
            mark.rectTransform.sizeDelta = new Vector2(84, 84);
            mark.color = DialogTextColor;
            mark.raycastTarget = false;
            mark.SetIcon(DialogIcon.Kind.Cube);

            var titleObject = new GameObject("HoloCube Wordmark", typeof(RectTransform), typeof(Text));
            titleObject.transform.SetParent(logo, false);
            var title = titleObject.GetComponent<Text>();
            title.rectTransform.anchorMin = title.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            title.rectTransform.anchoredPosition = new Vector2(86, 0);
            title.rectTransform.sizeDelta = new Vector2(260, 54);
            title.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            title.fontSize = 40;
            title.fontStyle = FontStyle.Bold;
            title.alignment = TextAnchor.MiddleLeft;
            title.color = DialogTextColor;
            title.raycastTarget = false;
            title.text = "HoloCube";

            var dividerObject = new GameObject("About Divider", typeof(RectTransform), typeof(Image));
            dividerObject.transform.SetParent(logo, false);
            var divider = dividerObject.GetComponent<Image>();
            divider.rectTransform.anchorMin = divider.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            divider.rectTransform.anchoredPosition = new Vector2(0, -58);
            divider.rectTransform.sizeDelta = new Vector2(900, 2);
            divider.color = WithAlpha(DialogTextColor, 0.25f);
            divider.raycastTarget = false;

            logoObject.SetActive(false);
            return logoObject;
        }

        private static SettingsPanel CreateSettings(Transform panel)
        {
            var rootObject = new GameObject("Settings View", typeof(RectTransform), typeof(SettingsPanel));
            rootObject.transform.SetParent(panel, false);
            var root = rootObject.GetComponent<RectTransform>();
            root.anchorMin = Vector2.zero;
            root.anchorMax = Vector2.one;
            root.sizeDelta = Vector2.zero;
            root.anchoredPosition = Vector2.zero;
            var settings = rootObject.GetComponent<SettingsPanel>();

            CreateSettingsText(root, "Settings Heading", "Settings", new Vector2(0, 140),
                new Vector2(900, 64), 36, TextAnchor.MiddleCenter, true);
            CreateSettingsRow(root, "Confidence Thresholds", DialogIcon.Kind.Confidence, 40,
                out var confidenceBox, out var confidenceValue);
            settings.ConfidenceBox = confidenceBox;
            settings.ConfidenceValue = confidenceValue;
            CreateSettingsRow(root, "Number of boxes (fixed)", DialogIcon.Kind.Boxes, -54,
                out var boxCountBox, out var boxCountValue);
            settings.BoxCountBox = boxCountBox;
            settings.BoxCountValue = boxCountValue;

            var dividerObject = new GameObject("Settings Divider", typeof(RectTransform), typeof(Image));
            dividerObject.transform.SetParent(root, false);
            var divider = dividerObject.GetComponent<Image>();
            divider.rectTransform.anchorMin = divider.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            divider.rectTransform.anchoredPosition = new Vector2(0, -109);
            divider.rectTransform.sizeDelta = new Vector2(960, 1);
            divider.color = WithAlpha(DialogTextColor, 0.2f);
            divider.raycastTarget = false;
            settings.HelpLabel = CreateSettingsText(root, "Joystick Help", SettingsPanel.HelpText,
                new Vector2(0, -154), new Vector2(980, 72), 22, TextAnchor.MiddleCenter);

            settings.Refresh(0.35f);
            rootObject.SetActive(false);
            return settings;
        }

        private static void CreateSettingsRow(Transform panel, string label, DialogIcon.Kind kind,
            float y, out DialogPanel valueBox, out Text value)
        {
            var icon = CreateDialogIcon(panel, kind);
            var iconGroup = (RectTransform)icon.transform.parent;
            iconGroup.gameObject.name = label + " Icon";
            iconGroup.anchorMin = iconGroup.anchorMax = new Vector2(0.5f, 0.5f);
            iconGroup.anchoredPosition = new Vector2(-440, y);
            CreateSettingsText(panel, label + " Label", label, new Vector2(-78, y),
                new Vector2(600, 68), 30, TextAnchor.MiddleLeft, true);

            valueBox = CreateCardLayer(panel, label + " Value Box", DialogBackgroundColor, 14, 1.5f,
                WithAlpha(DialogTextColor, 0.4f), Vector2.zero, new Vector2(160, 68));
            valueBox.rectTransform.anchorMin = valueBox.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            valueBox.rectTransform.anchoredPosition = new Vector2(395, y);
            value = CreateSettingsText(valueBox.transform, label + " Value", string.Empty, Vector2.zero,
                new Vector2(140, 60), 32, TextAnchor.MiddleCenter, true);
        }

        private static Text CreateSettingsText(Transform parent, string name, string value,
            Vector2 position, Vector2 size, int fontSize, TextAnchor alignment, bool bold = false)
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
            text.color = DialogTextColor;
            text.raycastTarget = false;
            text.text = value;
            return text;
        }

        private static DialogIcon CreateDialogIcon(Transform panel, DialogIcon.Kind kind)
        {
            var groupObject = new GameObject("Dialog Icon", typeof(RectTransform));
            groupObject.transform.SetParent(panel, false);
            var group = groupObject.GetComponent<RectTransform>();
            group.anchorMin = group.anchorMax = new Vector2(0, 0.5f);
            group.anchoredPosition = new Vector2(54, 0);
            group.sizeDelta = new Vector2(72, 72);
            CreateCardLayer(group, "Icon Medallion", new Color32(9, 53, 94, 255), 20, 1,
                WithAlpha(DialogTextColor, 0.22f), Vector2.zero, Vector2.zero);

            var iconObject = new GameObject("Icon Glyph", typeof(RectTransform), typeof(DialogIcon));
            iconObject.transform.SetParent(group, false);
            var icon = iconObject.GetComponent<DialogIcon>();
            icon.rectTransform.anchorMin = icon.rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
            icon.rectTransform.anchoredPosition = Vector2.zero;
            icon.rectTransform.sizeDelta = new Vector2(48, 48);
            icon.color = DialogTextColor;
            icon.raycastTarget = false;
            icon.SetIcon(kind);
            return icon;
        }

        private static void CreateDialogCard(Transform panel, float radius)
        {
            // Child layers preserve rounded corners and keep the shadow behind the card.
            CreateCardLayer(panel, "Card Shadow", new Color(0, 0, 0, 0.25f), radius + 3, 0,
                Color.clear, new Vector2(0, -8), new Vector2(6, 6));
            CreateCardLayer(panel, "Michigan Blue Card", DialogBackgroundColor, radius, 2,
                WithAlpha(DialogTextColor, 0.5f), Vector2.zero, Vector2.zero);

            var accentObject = new GameObject("Maize Accent", typeof(RectTransform), typeof(Image));
            accentObject.transform.SetParent(panel, false);
            var accent = accentObject.GetComponent<Image>();
            accent.rectTransform.anchorMin = accent.rectTransform.anchorMax = new Vector2(0.5f, 1);
            accent.rectTransform.anchoredPosition = new Vector2(0, -2);
            accent.rectTransform.sizeDelta = new Vector2(140, 2);
            accent.color = WithAlpha(DialogTextColor, 0.9f);
            accent.raycastTarget = false;
        }

        private static DialogPanel CreateCardLayer(Transform panel, string name, Color fill, float radius,
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
            return layer;
        }

        private static Color WithAlpha(Color color, float alpha) =>
            new Color(color.r, color.g, color.b, alpha);

    }
}

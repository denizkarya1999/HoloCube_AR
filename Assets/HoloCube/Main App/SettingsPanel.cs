using System.Globalization;
using UnityEngine;
using UnityEngine.UI;

namespace HoloCube.QuestYOLO
{
    /// <summary>Displays adjustable confidence, the fixed single-box limit, and the page's controls.</summary>
    public sealed class SettingsPanel : MonoBehaviour
    {
        /// <summary>The controls line under the rows; also opens In-Device Local File Inference.</summary>
        public const string HelpText =
            "Joystick left/right: adjust confidence\n" +
            "Right trigger: In-Device Local File Inference (test the model on an image stored on the headset)";

        public Text ConfidenceValue;
        public Text BoxCountValue;
        public DialogPanel ConfidenceBox;
        public DialogPanel BoxCountBox;
        [Tooltip("The controls line; found by name when a scene saved before it existed is used.")]
        public Text HelpLabel;

        private static readonly Color Maize = new Color32(255, 203, 5, 255);
        private static readonly Color InactiveBorder = new Color(1f, 203f / 255f, 5f / 255f, 0.35f);
        private static readonly Color DarkBlue = new Color32(0, 27, 54, 255);
        private static readonly Color SelectedBlue = new Color32(0, 55, 103, 255);
        private bool hasState;
        private float lastConfidence;

        public void Refresh(float confidence)
        {
            if (!hasState || !confidence.Equals(lastConfidence))
            {
                if (ConfidenceValue != null)
                {
                    ConfidenceValue.text = confidence.ToString("0.00", CultureInfo.InvariantCulture);
                    ConfidenceValue.color = Maize;
                }
                lastConfidence = confidence;
            }
            if (!hasState)
            {
                if (BoxCountValue != null)
                {
                    BoxCountValue.text = TransmitterRecognitionModel.MaxDetections.ToString(CultureInfo.InvariantCulture);
                    BoxCountValue.color = Maize;
                }
                SetSelected(ConfidenceBox, true);
                SetSelected(BoxCountBox, false);
                ApplyHelpText();
            }
            hasState = true;
        }

        /// <summary>Two-line controls text that fits under the divider; also updates an older saved scene.</summary>
        private void ApplyHelpText()
        {
            var help = HelpLabel;
            if (help == null)
            {
                var found = transform.Find("Joystick Help");
                help = found != null ? found.GetComponent<Text>() : null;
            }
            if (help == null) return;
            help.text = HelpText;
            help.fontSize = 22;
            help.horizontalOverflow = HorizontalWrapMode.Wrap;
            help.verticalOverflow = VerticalWrapMode.Overflow;
            help.rectTransform.sizeDelta = new Vector2(980, 72);
            help.rectTransform.anchoredPosition = new Vector2(0, -154);
        }

        private static void SetSelected(DialogPanel box, bool selected)
        {
            if (box == null) return;
            Color fill = selected ? SelectedBlue : DarkBlue;
            Color border = selected ? Maize : InactiveBorder;
            float width = selected ? 3f : 1.5f;
            if (box.color == fill && box.BorderColor == border && box.BorderWidth == width) return;
            box.color = fill;
            box.BorderColor = border;
            box.BorderWidth = width;
            box.SetVerticesDirty();
        }
    }
}

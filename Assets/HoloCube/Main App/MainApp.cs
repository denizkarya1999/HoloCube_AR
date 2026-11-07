using System;
using System.Collections;
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;
using UnityEngine.UI;

namespace HoloCube.QuestYOLO
{
    /// <summary>Start here: camera image → transmitter detections → boxes in the headset.</summary>
    public sealed class MainApp : MonoBehaviour
    {
        private enum MenuPage
        {
            Home,
            About,
            Settings,
            LocalFile
        }

        private const string ControllerMenu =
            "Press A: Transmitter Recognition · Hold B: Record Training Video\n" +
            "Press X: About HoloCube · Press Y: Settings\n" +
            "In Settings, right trigger: In-Device Local File Inference";

        [Header("Scene connections")]
        public PassthroughCameraAccess CameraAccess;
        public TransmitterRecognitionModel Model;
        public DetectionOverlay Overlay;
        public Text StatusLabel;
        public DialogIcon StatusIcon;
        public GameObject StatusIconContainer;
        public GameObject AboutLogo;
        public SettingsPanel SettingsView;
        public GameObject RecordingIndicator;
        public Text RecordingFeedbackLabel;
        public DialogIcon RecordingFeedbackIcon;

        [Header("Which detections to show")]
        [Tooltip("Minimum confidence: 0.35 means 35%.")]
        [Range(0, 1)] public float Confidence = 0.35f;

        [Header("Headset performance")]
        [Tooltip("Seconds to wait after a detection finishes before capturing again.")]
        [Min(0)] public float DetectionInterval = 0.1f;
        [Tooltip("Hide results older than this many seconds.")]
        [Min(0.1f)] public float MaxResultAge = 0.75f;

        private TransmitterInference inference;
        private LocalFileInference localFiles;
        private CameraVideoCapture videoCapture;
        private IEnumerator pendingDetection;
        private DateTime lastCameraTimestamp;
        private float nextCaptureTime;
        private bool userPaused;
        private bool applicationPaused;
        private bool hasFocus = true;
        private bool failed;
        private bool bButtonHeld;
        private bool recordingAttemptedForCurrentHold;
        private bool showControllerHelp;
        private MenuPage menuPage;
        private float nextSettingsJoystickActionTime;
        private float nextRecordingStartAttemptTime;
        private string statusMessage = string.Empty;
        private string lastRecordingHint = string.Empty;
        private string recordingUnavailableReason;
        private float recordingFeedbackUntil;
        private float inferenceOffNoticeUntil;

        private const float SettingsStickDeadzone = 0.6f;
        private const float SettingsJoystickRepeatInterval = 0.22f;
        private const float RecordingStartRetryInterval = 0.5f;
        private const float RecordingFeedbackDuration = 6f;
        private const float InferenceOffNoticeDuration = 3f;

        private IEnumerator Start()
        {
            videoCapture = new CameraVideoCapture();
            SetStatus("HoloCube · Loading object detection…");
            yield return null; // Let the loading message appear before loading the model.

            try
            {
                if (CameraAccess == null || Model == null || Overlay == null)
                    throw new InvalidOperationException("Assign the camera, model, and overlay in the scene.");

                inference = new TransmitterInference(Model);
                if (StatusLabel != null && StatusLabel.transform.parent != null)
                    localFiles = new LocalFileInference(StatusLabel.transform.parent.parent,
                        () => inference, () => Confidence);
                RequestPermissions();
                SetStatus("Waiting for camera permission…\nAllow camera access to begin");
            }
            catch (Exception error)
            {
                ShowError(error);
            }
        }

        private void Update()
        {
            try
            {
                if (!failed) ReadButtons();
                videoCapture?.Poll();
                RefreshRecordingIndicator();
                RefreshStatus();
                if (failed) return;

                if (!CameraIsReady()) return;

                var cameraImage = CameraAccess.GetTexture();
                if (cameraImage == null)
                {
                    recordingUnavailableReason = "Waiting for the Quest camera…";
                    StopRecording();
                    CancelDetection();
                    Overlay.Clear();
                    SetStatus("Waiting for the Quest camera…");
                    return;
                }

                UpdateVideoRecording(cameraImage);

                if (inference == null || !inference.IsReady)
                {
                    SetStatus("HoloCube · Loading model…");
                    return;
                }

                if (menuPage == MenuPage.LocalFile && localFiles != null)
                {
                    // The file page owns the PyTorch worker while it is open; the camera waits.
                    CancelDetection();
                    Overlay.Clear();
                    localFiles.Advance();
                    return;
                }
                localFiles?.Cancel();

                if (userPaused)
                {
                    CancelDetection();
                    Overlay.Clear();
                    SetStatus("Transmitter recognition paused");
                    return;
                }

                // One image at a time; the Android worker runs PyTorch in the background.
                if (pendingDetection == null) CaptureImage(cameraImage);
                ContinueDetection();
            }
            catch (Exception error)
            {
                ShowError(error);
            }
        }

        private void ReadButtons()
        {
            if (OVRInput.GetDown(OVRInput.Button.Start, OVRInput.Controller.LTouch))
            {
                showControllerHelp = !showControllerHelp;
                menuPage = MenuPage.Home;
            }

            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.RTouch))
            {
                if (menuPage == MenuPage.LocalFile && localFiles != null)
                {
                    localFiles.RunAgain();
                }
                else
                {
                    userPaused = !userPaused;
                    inferenceOffNoticeUntil = userPaused
                        ? Time.unscaledTime + InferenceOffNoticeDuration
                        : 0f;
                }
            }

            // Settings → right trigger opens In-Device Local File Inference.
            if (menuPage == MenuPage.Settings && localFiles != null &&
                OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
            {
                showControllerHelp = false;
                menuPage = MenuPage.LocalFile;
                CancelDetection();
                localFiles.Open();
            }

            bButtonHeld = OVRInput.Get(OVRInput.Button.Two, OVRInput.Controller.RTouch);
            if (!bButtonHeld)
            {
                recordingAttemptedForCurrentHold = false;
                nextRecordingStartAttemptTime = 0f;
            }

            if (OVRInput.GetDown(OVRInput.Button.One, OVRInput.Controller.LTouch))
            {
                showControllerHelp = false;
                menuPage = menuPage == MenuPage.About ? MenuPage.Home : MenuPage.About;
            }

            if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.LTouch))
            {
                showControllerHelp = false;
                menuPage = menuPage == MenuPage.Settings ? MenuPage.Home : MenuPage.Settings;
            }

            UpdateSettingsWithJoystick();
        }

        private void UpdateSettingsWithJoystick()
        {
            bool localFilePage = menuPage == MenuPage.LocalFile && localFiles != null;
            if ((menuPage != MenuPage.Settings && !localFilePage) ||
                Time.unscaledTime < nextSettingsJoystickActionTime)
                return;

            Vector2 stick = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch);
            if (Mathf.Abs(stick.x) < SettingsStickDeadzone)
                return;

            int direction = stick.x > 0f ? 1 : -1;
            if (localFilePage)
            {
                // Left/right picks the previous or next test image and runs the model on it.
                localFiles.Step(direction);
            }
            else
            {
                Confidence = Mathf.Clamp(
                    Mathf.Round((Confidence + direction * 0.05f) * 100f) / 100f,
                    0.05f,
                    1f);
            }

            nextSettingsJoystickActionTime = Time.unscaledTime + SettingsJoystickRepeatInterval;
        }

        private bool CameraIsReady()
        {
            string waitMessage = null;
            if (applicationPaused || !hasFocus)
                waitMessage = "Camera paused";
            else if (CameraAccess == null)
                waitMessage = "Camera is not configured";
            else if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.PassthroughCameraAccess))
                waitMessage = "Camera access is required\nAllow it in app permissions";
            else if (!CameraAccess.IsPlaying)
                waitMessage = "Waiting for the Quest camera…";

            recordingUnavailableReason = waitMessage;
            if (waitMessage == null) return true;

            StopRecording();
            CancelDetection();
            if (Overlay != null) Overlay.Clear();
            SetStatus(waitMessage);
            return false;
        }

        private void UpdateVideoRecording(Texture cameraImage)
        {
            if (!bButtonHeld)
            {
                recordingAttemptedForCurrentHold = false;
                nextRecordingStartAttemptTime = 0f;
                StopRecording();
                return;
            }

            if (videoCapture == null)
            {
                RefreshRecordingIndicator();
                return;
            }

            if (!recordingAttemptedForCurrentHold &&
                Time.unscaledTime >= nextRecordingStartAttemptTime)
            {
                lastRecordingHint = string.Empty;
                bool started = videoCapture.TryStart(cameraImage);
                recordingAttemptedForCurrentHold = started || !videoCapture.CanRetryStart;
                nextRecordingStartAttemptTime =
                    Time.unscaledTime + RecordingStartRetryInterval;
            }

            if (videoCapture.IsRecording)
                videoCapture.CaptureFrame(cameraImage, CameraAccess.Timestamp);

            RefreshRecordingIndicator();
        }

        private void CaptureImage(Texture image)
        {
            if (Time.unscaledTime < nextCaptureTime) return;
            if (CameraAccess.Timestamp == lastCameraTimestamp) return;

            if (image == null) return;

            lastCameraTimestamp = CameraAccess.Timestamp;
            var cameraPose = CameraAccess.GetCameraPose();
            float captureTime = Time.unscaledTime;
            pendingDetection = inference.Detect(
                image, Confidence,
                detections => ShowDetections(detections, cameraPose, captureTime));
        }

        private void ContinueDetection()
        {
            if (pendingDetection == null) return;

            // The first step copies the image now, before the camera changes it.
            // Later steps poll the background PyTorch worker without blocking rendering.
            if (pendingDetection.MoveNext()) return;

            CancelDetection();
            nextCaptureTime = Time.unscaledTime + DetectionInterval;
        }

        private void ShowDetections(List<Detection> detections, Pose cameraPose, float captureTime)
        {
            float age = Time.unscaledTime - captureTime;
            if (CameraAccess.IsPlaying && !userPaused && age <= MaxResultAge)
                Overlay.Draw(detections, inference.ImageSize, inference.Labels, cameraPose);
            else
                Overlay.Clear();

            SetStatus($"HoloCube · {detections.Count} transmitters · {age * 1000:0} ms · ~ means approximate depth");
        }

        private void RequestPermissions()
        {
            OVRPermissionsRequester.Request(new[]
            {
                OVRPermissionsRequester.Permission.PassthroughCameraAccess,
                OVRPermissionsRequester.Permission.Scene
            });
        }

        private void SetStatus(string message)
        {
            statusMessage = message;
            RefreshStatus();
        }

        private void RefreshStatus()
        {
            if (StatusLabel == null) return;
            GameObject statusPanel = StatusLabel.transform.parent != null
                ? StatusLabel.transform.parent.gameObject
                : StatusLabel.gameObject;
            bool showInferenceOffStatus =
                menuPage == MenuPage.Home && userPaused && !showControllerHelp && !failed &&
                Time.unscaledTime < inferenceOffNoticeUntil;
            // The file page has its own panel; the status card stays hidden behind it.
            bool showLocalFiles = menuPage == MenuPage.LocalFile && localFiles != null &&
                !showControllerHelp && !failed;
            localFiles?.Show(showLocalFiles);
            bool showStatusPanel = (menuPage != MenuPage.Home && !showLocalFiles) ||
                showControllerHelp || showInferenceOffStatus || failed;
            if (statusPanel.activeSelf != showStatusPanel)
                statusPanel.SetActive(showStatusPanel);
            if (!showStatusPanel) return;

            bool showAbout = menuPage == MenuPage.About && !showControllerHelp && !failed;
            bool showSettings = menuPage == MenuPage.Settings && !showControllerHelp &&
                !failed && SettingsView != null;
            if (SettingsView != null && SettingsView.gameObject.activeSelf != showSettings)
                SettingsView.gameObject.SetActive(showSettings);
            bool showStatusText = !showSettings;
            if (StatusLabel.gameObject.activeSelf != showStatusText)
                StatusLabel.gameObject.SetActive(showStatusText);
            if (AboutLogo != null && AboutLogo.activeSelf != showAbout)
                AboutLogo.SetActive(showAbout);
            GameObject statusIconObject = StatusIconContainer != null
                ? StatusIconContainer : StatusIcon != null ? StatusIcon.gameObject : null;
            bool showStatusIcon = !showAbout && !showSettings;
            if (statusIconObject != null && statusIconObject.activeSelf != showStatusIcon)
                statusIconObject.SetActive(showStatusIcon);
            if (showSettings)
            {
                SettingsView.Refresh(Confidence);
                return;
            }
            Vector2 textInsetMin = showAbout ? new Vector2(24, 24) : new Vector2(104, 20);
            Vector2 textInsetMax = showAbout ? new Vector2(-24, -132) : new Vector2(-24, -20);
            if (StatusLabel.rectTransform.offsetMin != textInsetMin)
                StatusLabel.rectTransform.offsetMin = textInsetMin;
            if (StatusLabel.rectTransform.offsetMax != textInsetMax)
                StatusLabel.rectTransform.offsetMax = textInsetMax;

            StatusIcon?.SetIcon(failed ? DialogIcon.Kind.Warning :
                showControllerHelp ? DialogIcon.Kind.Controls :
                showInferenceOffStatus ? DialogIcon.Kind.InferenceOff :
                menuPage == MenuPage.Settings ? DialogIcon.Kind.Settings : DialogIcon.Kind.Cube);

            if (failed)
                StatusLabel.text = statusMessage;
            else if (showControllerHelp)
                StatusLabel.text = "<size=32><b>Controller Guide</b></size>\n" +
                    ControllerMenu + "\nPress the Menu button to hide controls";
            else if (showInferenceOffStatus)
                StatusLabel.text = "<b>Inference Mode: Off</b>";
            else
                StatusLabel.text = GetMenuPageMessage();
        }

        private string GetMenuPageMessage()
        {
            switch (menuPage)
            {
                case MenuPage.About:
                    return "<b>Name:</b> HoloCube Research Project\n" +
                        "<b>Version:</b> 1.0\n" +
                        "<b>Release Date:</b> TBD\n" +
                        "<b>Developers:</b> Deniz K. Acikbas and Ahmad Jayeb\n" +
                        "<b>Advisor:</b> Xiao Zhang\n" +
                        "<b>Institution:</b> University of Michigan-Dearborn";
                case MenuPage.Settings:
                    return $"<size=32><b>Settings</b></size>\n" +
                        $"> <b>Confidence Thresholds</b>: {Confidence:0.00}\n" +
                        $"<b>Number of boxes (fixed)</b>: {TransmitterRecognitionModel.MaxDetections}\n" +
                        SettingsPanel.HelpText;
                default:
                    return string.IsNullOrEmpty(statusMessage) ? "HoloCube" : statusMessage;
            }
        }

        private void StopRecording()
        {
            videoCapture?.Stop();
            RefreshRecordingIndicator();
        }

        private void RefreshRecordingIndicator()
        {
            bool recording = videoCapture != null && videoCapture.IsRecording;
            if (RecordingIndicator != null && RecordingIndicator.activeSelf != recording)
                RecordingIndicator.SetActive(recording);

            if (RecordingFeedbackLabel == null) return;
            string hint = videoCapture?.StatusHint ?? string.Empty;
            if (hint != lastRecordingHint)
            {
                lastRecordingHint = hint;
                recordingFeedbackUntil = Time.unscaledTime + RecordingFeedbackDuration;
            }

            // Keep saving progress visible until the recorder reports success or failure.
            bool saving = hint.IndexOf("saving", StringComparison.OrdinalIgnoreCase) >= 0;
            bool showUnavailable = bButtonHeld && !recording && !saving &&
                !string.IsNullOrEmpty(recordingUnavailableReason);
            bool showFeedback = !recording && !string.IsNullOrEmpty(hint) &&
                (saving || Time.unscaledTime < recordingFeedbackUntil);
            showFeedback |= showUnavailable;
            GameObject panel = RecordingFeedbackLabel.transform.parent.gameObject;
            if (panel.activeSelf != showFeedback) panel.SetActive(showFeedback);
            if (showFeedback) RecordingFeedbackLabel.text = showUnavailable
                ? "Cannot record\n" + recordingUnavailableReason
                : hint;
            if (showFeedback)
                RecordingFeedbackIcon?.SetIcon(showUnavailable ? DialogIcon.Kind.Warning :
                    saving ? DialogIcon.Kind.Saving :
                    hint.StartsWith("Saved to ", StringComparison.Ordinal)
                        ? DialogIcon.Kind.Saved : DialogIcon.Kind.Warning);
        }

        private void ShowError(Exception error)
        {
            failed = true;
            Debug.LogException(error, this);
            try
            {
                StopRecording();
                videoCapture?.StopAndWait();
                CancelDetection();
            }
            catch (Exception cleanupError)
            {
                Debug.LogException(cleanupError, this);
            }
            if (Overlay != null) Overlay.Clear();
            SetStatus("Detection stopped\n" + error.Message);
        }

        private void CancelDetection()
        {
            var pending = pendingDetection;
            pendingDetection = null;
            // Disposing the iterator discards its result; native work can finish safely.
            (pending as IDisposable)?.Dispose();
        }

        private void OnApplicationPause(bool paused)
        {
            applicationPaused = paused;
            if (paused) StopRecording();
        }

        private void OnApplicationFocus(bool focused)
        {
            hasFocus = focused;
            if (!focused) StopRecording();
        }

        private void OnDisable()
        {
            StopRecording();
            videoCapture?.StopAndWait();
            localFiles?.Cancel();
            CancelDetection();
            if (Overlay != null) Overlay.Clear();
        }

        private void OnDestroy()
        {
            try
            {
                videoCapture?.Dispose();
            }
            finally
            {
                videoCapture = null;
                localFiles?.Dispose();
                localFiles = null;
                inference?.Dispose();
                inference = null;
            }
        }
    }
}

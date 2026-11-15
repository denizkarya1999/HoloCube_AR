// Camera-to-world projection adapted from Meta's MIT-licensed sample.
// Copyright (c) Meta Platforms, Inc. and affiliates.
using System.Collections.Generic;
using Meta.XR;
using UnityEngine;
using UnityEngine.UI;

namespace HoloCube.QuestYOLO
{
    /// <summary>Turn rectangles in the camera image into boxes and labels in the headset.</summary>
    public sealed class DetectionOverlay : MonoBehaviour
    {
        [Header("Scene connections")]
        public PassthroughCameraAccess CameraAccess;
        public EnvironmentRaycastManager Depth;
        public RectTransform BoxTemplate;

        [Tooltip("Distance in metres when real-world depth is unavailable. Labels show ~.")]
        public float FallbackDistance = 2f;
        [Tooltip("Seconds before an old overlay disappears.")]
        public float Lifetime = 0.5f;

        private readonly List<RectTransform> boxes = new();
        private float lastDrawTime;

        private void Awake() => BoxTemplate.gameObject.SetActive(false);

        private void Update()
        {
            if (Time.unscaledTime - lastDrawTime > Lifetime) Clear();
        }

        public void Clear()
        {
            foreach (var box in boxes) box.gameObject.SetActive(false);
        }

        /// <summary>Boxes are placed in the world, so a shown overlay stays valid while the
        /// slower orientation pass of the same frame is still running.</summary>
        public void KeepAlive() => lastDrawTime = Time.unscaledTime;

        public void Draw(List<Detection> detections, Vector2Int imageSize, string[] labels, Pose cameraPose,
            OrientationResult? orientation = null)
        {
            Clear();
            lastDrawTime = Time.unscaledTime;
            int visibleBoxes = 0;
            string direction = orientation.HasValue
                ? $"{orientation.Value.Name}  {orientation.Value.Confidence:P0}" : null;

            foreach (var detection in detections)
            {
                string text = $"{labels[detection.ClassId]}  {detection.Confidence:P0}";
                // The orientation belongs to the same pyramid: show it on the transmitter's label.
                if (direction != null) { text += "  ·  " + direction; direction = null; }
                if (Place(detection.Box, text, imageSize, cameraPose, visibleBoxes)) visibleBoxes++;
            }
            // No transmitter box for this frame: the orientation model's own box carries the label.
            if (direction != null) Place(orientation.Value.Box, direction, imageSize, cameraPose, visibleBoxes);
        }

        private bool Place(Vector4 pixels, string text, Vector2Int imageSize, Pose cameraPose, int index)
        {
            var viewport = ToCameraViewport(pixels, imageSize);
            var centerRay = CameraAccess.ViewportPointToRay(viewport.center, cameraPose);
            float distance = FindDistance(centerRay, cameraPose, out bool approximateDepth);
            if (distance <= 0.1f) return false;

            if (!TryPlaceRectangle(viewport, cameraPose, centerRay.GetPoint(distance),
                    out var rotation, out var size)) return false;

            var box = GetBox(index);
            box.SetPositionAndRotation(centerRay.GetPoint(distance), rotation);
            box.sizeDelta = size;
            box.gameObject.SetActive(true);
            var label = box.GetComponentInChildren<Text>(true);
            label.text = text + (approximateDepth ? "  ~" : "");
            return true;
        }

        private static Rect ToCameraViewport(Vector4 box, Vector2Int imageSize)
        {
            // YOLO uses pixels with a top-left origin. Camera viewport coordinates
            // go from 0 to 1 with a bottom-left origin, so flip the vertical axis.
            float left = box.x / imageSize.x;
            float right = box.z / imageSize.x;
            float bottom = 1f - box.w / imageSize.y;
            float top = 1f - box.y / imageSize.y;
            return Rect.MinMaxRect(left, bottom, right, top);
        }

        private float FindDistance(Ray ray, Pose cameraPose, out bool approximate)
        {
            bool canReadDepth = Depth != null && EnvironmentRaycastManager.IsSupported &&
                OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.Scene);
            if (canReadDepth && Depth.Raycast(ray, out var hit))
            {
                approximate = false;
                return Vector3.Distance(cameraPose.position, hit.point);
            }

            approximate = true;
            return FallbackDistance;
        }

        private bool TryPlaceRectangle(Rect viewport, Pose cameraPose, Vector3 center,
            out Quaternion rotation, out Vector2 size)
        {
            // Project both image corners onto a plane facing the capture camera.
            var normal = (center - cameraPose.position).normalized;
            var plane = new Plane(normal, center);
            var minRay = CameraAccess.ViewportPointToRay(viewport.min, cameraPose);
            var maxRay = CameraAccess.ViewportPointToRay(viewport.max, cameraPose);
            rotation = Quaternion.LookRotation(normal, cameraPose.rotation * Vector3.up);
            size = Vector2.zero;
            if (!plane.Raycast(minRay, out var minDistance) || !plane.Raycast(maxRay, out var maxDistance))
                return false;

            var worldSize = maxRay.GetPoint(maxDistance) - minRay.GetPoint(minDistance);
            var localSize = Quaternion.Inverse(rotation) * worldSize;
            size = new Vector2(Mathf.Abs(localSize.x), Mathf.Abs(localSize.y));
            return true;
        }

        private RectTransform GetBox(int index)
        {
            // Reuse existing UI objects instead of creating them for every image.
            if (index == boxes.Count) boxes.Add(Instantiate(BoxTemplate, BoxTemplate.parent));
            return boxes[index];
        }
    }
}

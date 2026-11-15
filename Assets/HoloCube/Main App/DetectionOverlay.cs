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

        private const string TriangleName = "Orientation Triangle";
        private static readonly Color TriangleColor = new Color32(255, 203, 5, 255); // Maize.
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
                // The orientation belongs to the same pyramid: name it on the transmitter's
                // label and draw its triangle inside the transmitter's box.
                OrientationResult? triangle = null;
                if (direction != null) { text += "  ·  " + direction; direction = null; triangle = orientation; }
                if (Place(detection.Box, text, imageSize, cameraPose, visibleBoxes, triangle)) visibleBoxes++;
            }
        }

        private bool Place(Vector4 pixels, string text, Vector2Int imageSize, Pose cameraPose, int index,
            OrientationResult? orientation)
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
            DrawTriangle(box, pixels, size, orientation);
            return true;
        }

        // The triangle lies on the box's own plane: each corner keeps its place relative to the
        // transmitter's rectangle in the image, so no second depth lookup is needed.
        private static void DrawTriangle(RectTransform box, Vector4 pixels, Vector2 size, OrientationResult? orientation)
        {
            var existing = box.Find(TriangleName);
            if (!orientation.HasValue)
            {
                if (existing != null) existing.gameObject.SetActive(false);
                return;
            }
            TriangleOutline triangle;
            if (existing != null) triangle = existing.GetComponent<TriangleOutline>();
            else
            {
                var triangleObject = new GameObject(TriangleName, typeof(RectTransform), typeof(TriangleOutline));
                triangleObject.layer = box.gameObject.layer;
                triangleObject.transform.SetParent(box, false);
                triangle = triangleObject.GetComponent<TriangleOutline>();
                // Fill the box and measure from its centre.
                triangle.rectTransform.anchorMin = Vector2.zero;
                triangle.rectTransform.anchorMax = Vector2.one;
                triangle.rectTransform.offsetMin = triangle.rectTransform.offsetMax = Vector2.zero;
                triangle.rectTransform.pivot = new Vector2(0.5f, 0.5f);
                triangle.color = TriangleColor;
                triangle.raycastTarget = false;
            }
            triangle.gameObject.SetActive(true);
            triangle.Thickness = Mathf.Max(size.x, size.y) * 0.03f;
            triangle.SetCorners(ToBox(orientation.Value.A, pixels, size), ToBox(orientation.Value.B, pixels, size),
                ToBox(orientation.Value.C, pixels, size));
        }

        /// <summary>An image pixel as a point in the box's own plane, measured from the box centre
        /// (x right, y up) in the box's units. The image's y axis points down.</summary>
        public static Vector2 ToBox(Vector2 pixel, Vector4 boxPixels, Vector2 boxSize)
        {
            float width = boxPixels.z - boxPixels.x, height = boxPixels.w - boxPixels.y;
            if (width <= 0 || height <= 0) return Vector2.zero;
            return new Vector2(((pixel.x - boxPixels.x) / width - 0.5f) * boxSize.x,
                (0.5f - (pixel.y - boxPixels.y) / height) * boxSize.y);
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

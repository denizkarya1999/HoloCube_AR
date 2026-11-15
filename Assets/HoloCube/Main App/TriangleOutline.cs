using UnityEngine;
using UnityEngine.UI;

namespace HoloCube.QuestYOLO
{
    /// <summary>The outline of the orientation triangle: three straight edges of a fixed thickness.</summary>
    [RequireComponent(typeof(CanvasRenderer))]
    public sealed class TriangleOutline : MaskableGraphic
    {
        /// <summary>Line width in the same units as the corners.</summary>
        public float Thickness = 4f;
        private Vector2 a, b, c;

        /// <summary>Corners in this object's local space (x right, y up from its pivot):
        /// top-left, top-right, and bottom tip of the face.</summary>
        public void SetCorners(Vector2 topLeft, Vector2 topRight, Vector2 tip)
        {
            a = topLeft;
            b = topRight;
            c = tip;
            SetVerticesDirty();
        }

        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();
            AddEdge(mesh, a, b);
            AddEdge(mesh, b, c);
            AddEdge(mesh, c, a);
        }

        private void AddEdge(VertexHelper mesh, Vector2 from, Vector2 to)
        {
            Vector2 along = to - from;
            if (along.sqrMagnitude < 1e-12f) return;
            Vector2 direction = along.normalized;
            Vector2 side = new Vector2(-direction.y, direction.x) * (Thickness * 0.5f);
            // Run each edge half a line width past its corners so the corners close.
            from -= direction * (Thickness * 0.5f);
            to += direction * (Thickness * 0.5f);
            int first = mesh.currentVertCount;
            mesh.AddVert(from - side, color, Vector2.zero);
            mesh.AddVert(from + side, color, Vector2.zero);
            mesh.AddVert(to + side, color, Vector2.zero);
            mesh.AddVert(to - side, color, Vector2.zero);
            mesh.AddTriangle(first, first + 1, first + 2);
            mesh.AddTriangle(first, first + 2, first + 3);
        }
    }
}

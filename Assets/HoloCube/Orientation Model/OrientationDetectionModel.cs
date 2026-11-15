namespace HoloCube.QuestYOLO
{
    /// <summary>The trained orientation checkpoint, its classes, and its output layout.</summary>
    public static class OrientationDetectionModel
    {
        public const string CheckpointFile = "holocube_orientation.pt";
        public const string CheckpointHash = "71326fcb52f994cf7373120ee42abe5693dc0051a78fc735e2f45e2c540b4579";
        // The model was trained at 640 x 640; the cube pattern is unreadable at 320.
        public const int InputSize = 640;
        // holocube_runtime.py returns the strongest rows only:
        // [x1, y1, x2, y2, confidence, class ID, ax, ay, bx, by, cx, cy].
        public const int OutputRows = 5;
        public const int OutputColumns = 12;
        // One hologram is in view, so only the strongest valid orientation is kept.
        public const int MaxDetections = 1;
        // An orientation is shown only when its box centre lies inside the transmitter box
        // grown by this fraction of its width and height on every side.
        public const float TransmitterBoxMargin = 0.25f;
        // Class IDs 0-7, clockwise from North.
        public static readonly string[] ClassNames =
            { "North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest" };
    }
}

package com.gesture.recognition

/**
 * Configuration constants for gesture recognition
 *
 * These values match the Python training configuration exactly
 */
object Config {

    // Model Input Shape
    const val SEQUENCE_LENGTH = 15  // Number of frames in sequence
    const val NUM_FEATURES = 63     // 21 landmarks × 3 coordinates (x, y, z)
    const val NUM_CLASSES = 11      // Number of gesture classes

    // Normalization Parameters (must match training)
    const val MIN_HAND_SCALE = 0.01f
    const val NORMALIZATION_CLIP_RANGE = 2.0f

    // Prediction Thresholds
    const val CONFIDENCE_THRESHOLD = 0.6f  // Minimum confidence for gesture detection
    const val MIN_DETECTION_CONFIDENCE = 0.5f  // MediaPipe hand detection threshold

    // Performance Settings
    const val TARGET_FPS = 30
    const val SMOOTHING_WINDOW = 5  // Number of predictions to smooth over

    // Gesture Label Mapping (must match training labels)
    val LABEL_MAP = mapOf(
        0 to "doing_other_things",
        1 to "swipe_left",
        2 to "swipe_right",
        3 to "thumb_down",
        4 to "thumb_up",
        5 to "v_gesture",
        6 to "top",
        7 to "left_gesture",
        8 to "right_gesture",
        9 to "stop_sign",
        10 to "heart"
    )

    // Display Names (user-friendly versions)
    val DISPLAY_NAMES = mapOf(
        "doing_other_things" to "Idle",
        "swipe_left" to "Swipe Left",
        "swipe_right" to "Swipe Right",
        "thumb_down" to "Thumbs Down",
        "thumb_up" to "Thumbs Up",
        "v_gesture" to "Peace Sign",
        "top" to "Point Up",
        "left_gesture" to "Point Left",
        "right_gesture" to "Point Right",
        "stop_sign" to "Stop",
        "heart" to "Heart"
    )

    /**
     * Get display name for a gesture
     */
    fun getDisplayName(gesture: String): String {
        return DISPLAY_NAMES[gesture] ?: gesture
    }

    /**
     * Check if gesture is valid
     */
    fun isValidGesture(gesture: String): Boolean {
        return LABEL_MAP.values.contains(gesture)
    }
}
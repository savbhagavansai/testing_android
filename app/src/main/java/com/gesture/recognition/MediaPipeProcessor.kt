package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * MediaPipe Hand Landmarker with flexible accelerator configuration
 *
 * Supports: GPU, CPU delegates (MediaPipe doesn't support NNAPI)
 */
class MediaPipeProcessor(private val context: Context) {

    private val TAG = "MediaPipeProcessor"

    private var handLandmarker: HandLandmarker? = null
    private var actualDelegate: String = "UNKNOWN"

    init {
        initialize()
    }

    /**
     * Initialize MediaPipe with configured accelerator
     */
    private fun initialize() {
        try {
            Log.d(TAG, "Initializing MediaPipe with ${Config.MEDIAPIPE_ACCELERATOR} accelerator...")

            // Determine delegate based on config
            // Note: MediaPipe only supports GPU and CPU (no NNAPI delegate available)
            val delegate = when (Config.MEDIAPIPE_ACCELERATOR.uppercase()) {
                "GPU" -> {
                    actualDelegate = "GPU"
                    Delegate.GPU
                }
                "CPU" -> {
                    actualDelegate = "CPU"
                    Delegate.CPU
                }
                else -> {
                    // Default to GPU for any other value (including "NNAPI")
                    Log.w(TAG, "MediaPipe doesn't support ${Config.MEDIAPIPE_ACCELERATOR}, using GPU instead")
                    actualDelegate = "GPU"
                    Delegate.GPU
                }
            }

            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .setDelegate(delegate)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(Config.MIN_DETECTION_CONFIDENCE)
                .setMinHandPresenceConfidence(Config.MIN_DETECTION_CONFIDENCE)
                .setMinTrackingConfidence(Config.MIN_DETECTION_CONFIDENCE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)

            Log.d(TAG, "✓ MediaPipe initialized successfully with $actualDelegate delegate")

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe", e)

            // Try fallback to CPU if GPU fails
            if (Config.MEDIAPIPE_ACCELERATOR.uppercase() != "CPU") {
                Log.w(TAG, "Attempting CPU fallback...")
                try {
                    val baseOptions = BaseOptions.builder()
                        .setModelAssetPath("hand_landmarker.task")
                        .setDelegate(Delegate.CPU)
                        .build()

                    val options = HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(baseOptions)
                        .setRunningMode(RunningMode.IMAGE)
                        .setNumHands(1)
                        .setMinHandDetectionConfidence(Config.MIN_DETECTION_CONFIDENCE)
                        .setMinHandPresenceConfidence(Config.MIN_DETECTION_CONFIDENCE)
                        .setMinTrackingConfidence(Config.MIN_DETECTION_CONFIDENCE)
                        .build()

                    handLandmarker = HandLandmarker.createFromOptions(context, options)
                    actualDelegate = "CPU (fallback)"

                    Log.d(TAG, "✓ MediaPipe initialized with CPU fallback")
                } catch (fallbackError: Exception) {
                    Log.e(TAG, "CPU fallback also failed", fallbackError)
                    throw fallbackError
                }
            } else {
                throw e
            }
        }
    }

    /**
     * Extract landmarks from bitmap
     * Returns normalized coordinates (0.0 to 1.0) or null if no hand detected
     */
    fun extractLandmarks(bitmap: Bitmap): FloatArray? {
        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = handLandmarker?.detect(mpImage)

            if (result?.landmarks()?.isNotEmpty() == true) {
                val landmarks = result.landmarks()[0]
                val landmarkArray = FloatArray(63)

                for (i in 0 until 21) {
                    landmarkArray[i * 3] = landmarks[i].x()
                    landmarkArray[i * 3 + 1] = landmarks[i].y()
                    landmarkArray[i * 3 + 2] = landmarks[i].z()
                }

                return landmarkArray
            }

            return null

        } catch (e: Exception) {
            Log.e(TAG, "Error extracting landmarks", e)
            return null
        }
    }

    /**
     * Get the actual delegate being used
     */
    fun getActualDelegate(): String {
        return actualDelegate
    }

    /**
     * Close resources
     */
    fun close() {
        try {
            handLandmarker?.close()
            Log.d(TAG, "MediaPipeProcessor closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing MediaPipeProcessor", e)
        }
    }
}
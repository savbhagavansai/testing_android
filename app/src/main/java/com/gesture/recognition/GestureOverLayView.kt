package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.*
import kotlin.math.max

/**
 * Complete Gesture Recognizer
 * Integrates MediaPipe + ONNX for real-time gesture recognition
 *
 * Pipeline:
 * 1. Detect hand landmarks (MediaPipe)
 * 2. Normalize landmarks
 * 3. Buffer 15 frames
 * 4. Run ONNX inference
 * 5. Smooth predictions
 */
class GestureRecognizer(private val context: Context) {

    companion object {
        private const val TAG = "GestureRecognizer"
    }

    // MediaPipe
    private var handLandmarker: HandLandmarker? = null

    // ONNX Runtime
    private var ortEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null

    // Landmark buffer (stores last 15 frames)
    private val landmarkBuffer = ArrayDeque<FloatArray>(Config.SEQUENCE_LENGTH)

    // Prediction smoothing (stores last 5 predictions)
    private val predictionBuffer = ArrayDeque<String>(Config.SMOOTHING_WINDOW)

    // Performance tracking
    private var mediaPipeTime = 0f
    private var onnxTime = 0f

    /**
     * Initialize MediaPipe and ONNX Runtime
     */
    fun initialize(): Boolean {
        return try {
            initializeMediaPipe()
            initializeONNX()
            Log.d(TAG, "GestureRecognizer initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize GestureRecognizer", e)
            false
        }
    }

    /**
     * Initialize MediaPipe Hand Landmarker
     */
    private fun initializeMediaPipe() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
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
            Log.d(TAG, "MediaPipe Hand Landmarker initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe", e)
            throw e
        }
    }

    /**
     * Initialize ONNX Runtime and load model
     */
    private fun initializeONNX() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = context.assets.open("gesture_model.onnx").readBytes()
            onnxSession = ortEnvironment?.createSession(modelBytes)

            Log.d(TAG, "ONNX Runtime initialized, model loaded")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Runtime", e)
            throw e
        }
    }

    /**
     * Process a bitmap and recognize gesture
     */
    fun recognizeGesture(bitmap: Bitmap): GestureResult {
        try {
            // Step 1: Detect hand landmarks with MediaPipe
            val mediaPipeStart = System.currentTimeMillis()
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = handLandmarker?.detect(mpImage)
            mediaPipeTime = (System.currentTimeMillis() - mediaPipeStart).toFloat()

            // Step 2: Check if hand detected
            if (result == null || result.landmarks().isEmpty()) {
                return GestureResult(
                    gesture = "None",
                    confidence = 0f,
                    probabilities = FloatArray(Config.NUM_CLASSES) { 0f },
                    handDetected = false,
                    bufferProgress = landmarkBuffer.size.toFloat() / Config.SEQUENCE_LENGTH,
                    mediaPipeTime = mediaPipeTime,
                    onnxTime = 0f
                )
            }

            // Step 3: Extract and normalize landmarks
            val landmarks = result.landmarks().first()
            val normalizedLandmarks = extractAndNormalizeLandmarks(landmarks)

            // Step 4: Add to buffer
            if (landmarkBuffer.size >= Config.SEQUENCE_LENGTH) {
                landmarkBuffer.removeFirst()
            }
            landmarkBuffer.addLast(normalizedLandmarks)

            // Step 5: Check if buffer is full
            if (landmarkBuffer.size < Config.SEQUENCE_LENGTH) {
                return GestureResult(
                    gesture = "Buffering",
                    confidence = 0f,
                    probabilities = FloatArray(Config.NUM_CLASSES) { 0f },
                    handDetected = true,
                    bufferProgress = landmarkBuffer.size.toFloat() / Config.SEQUENCE_LENGTH,
                    mediaPipeTime = mediaPipeTime,
                    onnxTime = 0f
                )
            }

            // Step 6: Run ONNX inference
            val onnxStart = System.currentTimeMillis()
            val (gesture, confidence, probabilities) = runInference()
            onnxTime = (System.currentTimeMillis() - onnxStart).toFloat()

            // Step 7: Smooth prediction
            val smoothedGesture = smoothPrediction(gesture, confidence)

            return GestureResult(
                gesture = smoothedGesture,
                confidence = confidence,
                probabilities = probabilities,
                handDetected = true,
                bufferProgress = 1f,
                mediaPipeTime = mediaPipeTime,
                onnxTime = onnxTime
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error recognizing gesture", e)
            return GestureResult(
                gesture = "Error",
                confidence = 0f,
                probabilities = FloatArray(Config.NUM_CLASSES) { 0f },
                handDetected = false,
                bufferProgress = 0f,
                mediaPipeTime = 0f,
                onnxTime = 0f
            )
        }
    }

    /**
     * Extract landmarks and normalize using exact Python training logic
     */
    private fun extractAndNormalizeLandmarks(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>): FloatArray {
        // Step 1: Extract raw landmarks (21 landmarks × 3 coords = 63 features)
        val rawLandmarks = FloatArray(63)
        landmarks.forEachIndexed { index, landmark ->
            rawLandmarks[index * 3] = landmark.x()
            rawLandmarks[index * 3 + 1] = landmark.y()
            rawLandmarks[index * 3 + 2] = landmark.z()
        }

        // Step 2: Normalize (exact match with Python training)
        val normalized = FloatArray(63)

        // Get wrist position (landmark 0)
        val wristX = rawLandmarks[0]
        val wristY = rawLandmarks[1]
        val wristZ = rawLandmarks[2]

        // Make wrist-relative
        for (i in 0 until 21) {
            normalized[i * 3] = rawLandmarks[i * 3] - wristX
            normalized[i * 3 + 1] = rawLandmarks[i * 3 + 1] - wristY
            normalized[i * 3 + 2] = rawLandmarks[i * 3 + 2] - wristZ
        }

        // Calculate scale
        var maxX = 0f
        var minX = 0f
        var maxY = 0f
        var minY = 0f

        for (i in 0 until 21) {
            val x = normalized[i * 3]
            val y = normalized[i * 3 + 1]
            maxX = max(maxX, x)
            minX = kotlin.math.min(minX, x)
            maxY = max(maxY, y)
            minY = kotlin.math.min(minY, y)
        }

        val rangeX = maxX - minX
        val rangeY = maxY - minY
        val scale = max(rangeX, rangeY)

        // Scale if hand is large enough
        if (scale > Config.MIN_HAND_SCALE) {
            for (i in 0 until 63) {
                normalized[i] /= scale
            }
        }

        // Clip values
        for (i in 0 until 63) {
            normalized[i] = normalized[i].coerceIn(
                -Config.NORMALIZATION_CLIP_RANGE,
                Config.NORMALIZATION_CLIP_RANGE
            )
        }

        return normalized
    }

    /**
     * Run ONNX inference on buffered landmarks
     */
    private fun runInference(): Triple<String, Float, FloatArray> {
        try {
            // Prepare input tensor (1, 15, 63)
            val inputData = Array(1) {
                Array(Config.SEQUENCE_LENGTH) { FloatArray(Config.NUM_FEATURES) }
            }

            // Copy landmarks from buffer to tensor
            landmarkBuffer.forEachIndexed { timeStep, landmarks ->
                landmarks.forEachIndexed { featureIdx, value ->
                    inputData[0][timeStep][featureIdx] = value
                }
            }

            // Run inference
            val outputs = onnxSession?.run(
                mapOf("input" to OnnxTensor.createTensor(ortEnvironment, inputData))
            )

            // Get output probabilities
            val outputTensor = outputs?.get(0)?.value as? Array<FloatArray>
            if (outputTensor == null) {
                Log.e(TAG, "Failed to get output tensor")
                return Triple("Unknown", 0f, FloatArray(Config.NUM_CLASSES) { 0f })
            }

            val probabilities = outputTensor[0]

            // Get prediction
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIndex]
            val gesture = Config.LABEL_MAP[maxIndex] ?: "Unknown"

            // Close output
            outputs?.close()

            return Triple(gesture, confidence, probabilities)

        } catch (e: Exception) {
            Log.e(TAG, "Error running ONNX inference", e)
            return Triple("Error", 0f, FloatArray(Config.NUM_CLASSES) { 0f })
        }
    }

    /**
     * Smooth predictions using majority voting
     */
    private fun smoothPrediction(gesture: String, confidence: Float): String {
        // Add to buffer
        if (predictionBuffer.size >= Config.SMOOTHING_WINDOW) {
            predictionBuffer.removeFirst()
        }
        predictionBuffer.addLast(gesture)

        // Use majority voting if confidence is high enough
        return if (confidence > Config.CONFIDENCE_THRESHOLD && predictionBuffer.size >= 3) {
            // Count occurrences
            val counts = predictionBuffer.groupingBy { it }.eachCount()
            counts.maxByOrNull { it.value }?.key ?: gesture
        } else {
            gesture
        }
    }

    /**
     * Get current buffer status
     */
    fun getBufferSize(): Int = landmarkBuffer.size

    /**
     * Clear all buffers
     */
    fun clearBuffers() {
        landmarkBuffer.clear()
        predictionBuffer.clear()
    }

    /**
     * Get performance metrics
     */
    fun getPerformanceMetrics(): Pair<Float, Float> {
        return Pair(mediaPipeTime, onnxTime)
    }

    /**
     * Close and cleanup resources
     */
    fun close() {
        try {
            handLandmarker?.close()
            onnxSession?.close()
            Log.d(TAG, "GestureRecognizer closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing GestureRecognizer", e)
        }
    }
}

/**
 * Data class for gesture recognition results
 */
data class GestureResult(
    val gesture: String,
    val confidence: Float,
    val probabilities: FloatArray,
    val handDetected: Boolean,
    val bufferProgress: Float,
    val mediaPipeTime: Float,
    val onnxTime: Float
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as GestureResult

        if (gesture != other.gesture) return false
        if (confidence != other.confidence) return false
        if (!probabilities.contentEquals(other.probabilities)) return false
        if (handDetected != other.handDetected) return false

        return true
    }

    override fun hashCode(): Int {
        var result = gesture.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + probabilities.contentHashCode()
        result = 31 * result + handDetected.hashCode()
        return result
    }
}
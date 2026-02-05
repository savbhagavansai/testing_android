package com.gesture.recognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Custom overlay view for drawing hand landmarks, gestures, and performance metrics
 *
 * Complete implementation matching MainActivity requirements
 */
class GestureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "GestureOverlay"
        private const val LANDMARK_RADIUS = 8f
        private const val CONNECTION_THICKNESS = 4f

        // MediaPipe hand connections
        private val HAND_CONNECTIONS = listOf(
            // Thumb
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            // Index
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            // Middle
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            // Ring
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            // Pinky
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            // Palm
            5 to 9, 9 to 13, 13 to 17
        )
    }

    // State variables
    private var handLandmarks: HandLandmarkerResult? = null
    private var currentGesture: String = "None"
    private var gestureConfidence: Float = 0f
    private var allProbabilities: FloatArray = FloatArray(Config.NUM_CLASSES) { 0f }
    private var bufferSize: Int = 0
    private var isHandDetected: Boolean = false
    private var fps: Float = 0f
    private var frameCount: Int = 0

    // Performance metrics
    private var mediaPipeMs: Float = 0f
    private var onnxMs: Float = 0f

    // Paint objects
    private val landmarkPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val connectionPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = CONNECTION_THICKNESS
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
    }

    private val titlePaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isAntiAlias = true
        isFakeBoldText = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(230, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    init {
        setWillNotDraw(false)
    }

    /**
     * Update hand landmarks from MediaPipe
     */
    fun updateHandLandmarks(result: HandLandmarkerResult?) {
        handLandmarks = result
        isHandDetected = result?.landmarks()?.isNotEmpty() == true
        invalidate()
    }

    /**
     * Update gesture prediction results
     */
    fun updateGesture(gesture: String, confidence: Float, probabilities: FloatArray) {
        currentGesture = gesture
        gestureConfidence = confidence
        allProbabilities = probabilities
        invalidate()
    }

    /**
     * Update buffer status
     */
    fun updateBuffer(size: Int) {
        bufferSize = size
        invalidate()
    }

    /**
     * Update FPS counter
     */
    fun updateFPS(currentFps: Float, frame: Int) {
        fps = currentFps
        frameCount = frame
        invalidate()
    }

    /**
     * Update performance metrics
     */
    fun updatePerformanceMetrics(mediaPipeTime: Float, onnxTime: Float) {
        mediaPipeMs = mediaPipeTime
        onnxMs = onnxTime
        invalidate()
    }

    /**
     * Main drawing method
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            // Draw in order
            drawHandSkeleton(canvas)
            drawTopPanel(canvas)
            drawBottomInfo(canvas)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error drawing overlay", e)
        }
    }

    /**
     * Draw hand skeleton (landmarks + connections)
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        val landmarks = handLandmarks?.landmarks()?.firstOrNull() ?: return

        // Draw connections
        for ((start, end) in HAND_CONNECTIONS) {
            if (start < landmarks.size && end < landmarks.size) {
                val startLandmark = landmarks[start]
                val endLandmark = landmarks[end]

                canvas.drawLine(
                    startLandmark.x() * width,
                    startLandmark.y() * height,
                    endLandmark.x() * width,
                    endLandmark.y() * height,
                    connectionPaint
                )
            }
        }

        // Draw landmarks
        for (landmark in landmarks) {
            canvas.drawCircle(
                landmark.x() * width,
                landmark.y() * height,
                LANDMARK_RADIUS,
                landmarkPaint
            )
        }
    }

    /**
     * Draw top panel (hand status, gesture, buffer)
     */
    private fun drawTopPanel(canvas: Canvas) {
        val panelX = 40f
        val panelY = 50f
        val panelWidth = 350f
        val panelHeight = 180f

        // Background
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            20f, 20f,
            backgroundPaint
        )

        var textY = panelY + 40f

        // Hand detection status
        val handStatus = if (isHandDetected) "✓ HAND DETECTED" else "✗ NO HAND"
        val handColor = if (isHandDetected) Color.GREEN else Color.RED
        textPaint.color = handColor
        canvas.drawText(handStatus, panelX + 20f, textY, textPaint)
        textY += 35f

        // Buffer status
        textPaint.color = Color.WHITE
        canvas.drawText("Buffer: $bufferSize/${Config.SEQUENCE_LENGTH}", panelX + 20f, textY, textPaint)
        textY += 35f

        // Current gesture
        if (isHandDetected && currentGesture != "None") {
            val gestureColor = if (gestureConfidence > Config.CONFIDENCE_THRESHOLD) {
                Color.GREEN
            } else {
                Color.YELLOW
            }
            textPaint.color = gestureColor
            canvas.drawText(
                "GESTURE: ${currentGesture.replace('_', ' ').uppercase()}",
                panelX + 20f,
                textY,
                textPaint
            )
            textY += 35f

            // Confidence
            textPaint.color = Color.WHITE
            textPaint.textSize = 24f
            canvas.drawText(
                "Confidence: ${(gestureConfidence * 100).toInt()}%",
                panelX + 20f,
                textY,
                textPaint
            )
            textPaint.textSize = 28f
        }
    }

    /**
     * Draw bottom info (FPS, performance)
     */
    private fun drawBottomInfo(canvas: Canvas) {
        val y = height - 100f

        // FPS
        textPaint.color = Color.YELLOW
        textPaint.textSize = 32f
        canvas.drawText("FPS: %.1f".format(fps), 40f, y, textPaint)

        // Frame count
        textPaint.textSize = 24f
        canvas.drawText("Frame: $frameCount", 40f, y + 35f, textPaint)

        // Performance metrics
        if (mediaPipeMs > 0 || onnxMs > 0) {
            textPaint.textSize = 20f
            textPaint.color = Color.CYAN
            canvas.drawText(
                "MP: %.1fms | ONNX: %.1fms".format(mediaPipeMs, onnxMs),
                40f,
                y + 60f,
                textPaint
            )
        }

        textPaint.textSize = 28f
    }
}
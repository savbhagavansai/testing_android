package com.gesture.recognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.max

/**
 * Custom overlay view for drawing hand landmarks, gestures, and performance metrics
 *
 * FIXES APPLIED (2026-02-05):
 * 1. ✅ Changed draw order - debug panel now drawn LAST (on top)
 * 2. ✅ Bright orange background - impossible to miss
 * 3. ✅ Moved to Y=700px - clear space below hand
 */
class GestureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val TAG = "GestureOverlay"

        // Hand landmark drawing
        private const val LANDMARK_RADIUS = 8f
        private const val CONNECTION_THICKNESS = 4f

        // UI dimensions
        private const val PANEL_CORNER_RADIUS = 20f
        private const val PROB_BAR_HEIGHT = 20f
        private const val PROB_BAR_SPACING = 8f

        // MediaPipe hand connections (21 landmarks form these connections)
        private val HAND_CONNECTIONS = listOf(
            // Thumb
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            // Index finger
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            // Middle finger
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            // Ring finger
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
    private var allProbabilities: FloatArray = FloatArray(11) { 0f }
    private var bufferSize: Int = 0
    private var isHandDetected: Boolean = false
    private var fps: Float = 0f
    private var frameCount: Int = 0

    // Performance metrics
    private var mediaPipeMs: Float = 0f
    private var onnxMs: Float = 0f
    private var showDebugPanel: Boolean = false

    // Paint objects (reused for efficiency)
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

    private val subtitlePaint = Paint().apply {
        color = Color.WHITE
        textSize = 24f
        isAntiAlias = true
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 22f
        isAntiAlias = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(230, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val barPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean {
            // Double tap handled by MainActivity
            return true
        }

        override fun onLongPress(e: MotionEvent) {
            // Toggle debug panel on long press
            showDebugPanel = !showDebugPanel
            invalidate()
        }
    })

    init {
        setWillNotDraw(false)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        gestureDetector.onTouchEvent(event)
        return true
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
     *
     * ⭐ FIX #1: CHANGED DRAW ORDER
     * Debug panel now drawn LAST to appear on top!
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            // Draw in order: back to front
            drawHandSkeleton(canvas)
            drawTopPanel(canvas)
            drawProbabilityPanel(canvas)      // ← Draw probability FIRST
            drawDebugPanel(canvas)             // ← Draw debug LAST (on top!) ⭐ FIX #1
            drawBottomInstructions(canvas)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error drawing overlay", e)
        }
    }

    /**
     * Draw hand skeleton (landmarks + connections)
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        val landmarks = handLandmarks?.landmarks()?.firstOrNull() ?: return

        // Draw connections (lines between landmarks)
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

        // Draw landmarks (dots at each point)
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
     * Draw top-left panel (hand detection, buffer, gesture)
     */
    private fun drawTopPanel(canvas: Canvas) {
        val panelX = 40f
        val panelY = 50f
        val panelWidth = 350f
        val panelHeight = 160f

        // Background
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            PANEL_CORNER_RADIUS,
            PANEL_CORNER_RADIUS,
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
        canvas.drawText("Buffer: $bufferSize/15", panelX + 20f, textY, textPaint)
        textY += 35f

        // Current gesture
        if (isHandDetected && currentGesture != "None") {
            val gestureColor = if (gestureConfidence > 0.6f) Color.GREEN else Color.YELLOW
            textPaint.color = gestureColor
            canvas.drawText("GESTURE: ${currentGesture.uppercase()}", panelX + 20f, textY, textPaint)
            textY += 35f

            // Confidence bar
            val barWidth = 300f
            val barX = panelX + 20f

            // Background bar (gray)
            barPaint.color = Color.GRAY
            canvas.drawRoundRect(
                barX, textY - 20f,
                barX + barWidth, textY,
                10f, 10f, barPaint
            )

            // Confidence bar (green/yellow based on threshold)
            barPaint.color = gestureColor
            canvas.drawRoundRect(
                barX, textY - 20f,
                barX + (barWidth * gestureConfidence), textY,
                10f, 10f, barPaint
            )

            // Confidence percentage
            textPaint.color = Color.WHITE
            textPaint.textSize = 20f
            canvas.drawText(
                "${(gestureConfidence * 100).toInt()}%",
                barX + barWidth + 10f,
                textY - 5f,
                textPaint
            )
            textPaint.textSize = 28f
        }

        // FPS counter (top right)
        textPaint.color = Color.WHITE
        canvas.drawText("FPS: %.1f".format(fps), width - 150f, 50f, textPaint)
        canvas.drawText("Frame: $frameCount", width - 150f, 85f, textPaint)
    }

    /**
     * Draw probability panel (right side)
     */
    private fun drawProbabilityPanel(canvas: Canvas) {
        val panelWidth = 220f
        val panelX = width - panelWidth - 40f
        val panelY = 180f
        val panelHeight = 50f + (allProbabilities.size * (PROB_BAR_HEIGHT + PROB_BAR_SPACING + 25f))

        // Background
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            PANEL_CORNER_RADIUS,
            PANEL_CORNER_RADIUS,
            backgroundPaint
        )

        // Title
        textPaint.color = Color.WHITE
        canvas.drawText("Probabilities:", panelX + 15f, panelY + 35f, textPaint)

        var itemY = panelY + 70f

        // Gesture labels (from config)
        val labels = listOf(
            "doing other", "swipe left", "swipe right",
            "thumb down", "thumb up", "v gesture",
            "top", "left gesture", "right gesture",
            "stop sign", "heart"
        )

        // Draw each probability
        for (i in allProbabilities.indices) {
            val prob = allProbabilities[i]
            val label = labels.getOrNull(i) ?: "Unknown"

            // Label
            smallTextPaint.color = Color.WHITE
            canvas.drawText(label, panelX + 15f, itemY, smallTextPaint)
            itemY += 22f

            // Probability bar
            val barWidth = 180f
            val barX = panelX + 15f

            // Background (gray)
            barPaint.color = Color.GRAY
            canvas.drawRoundRect(
                barX, itemY - PROB_BAR_HEIGHT,
                barX + barWidth, itemY,
                8f, 8f, barPaint
            )

            // Foreground (green if high, yellow if medium)
            barPaint.color = when {
                prob > 0.6f -> Color.GREEN
                prob > 0.3f -> Color.YELLOW
                else -> Color.rgb(100, 100, 100)
            }
            canvas.drawRoundRect(
                barX, itemY - PROB_BAR_HEIGHT,
                barX + (barWidth * prob), itemY,
                8f, 8f, barPaint
            )

            // Percentage
            smallTextPaint.color = Color.WHITE
            smallTextPaint.textSize = 18f
            canvas.drawText(
                "${(prob * 100).toInt()}%",
                barX + barWidth + 5f,
                itemY - 5f,
                smallTextPaint
            )
            smallTextPaint.textSize = 22f

            itemY += PROB_BAR_SPACING + 8f
        }
    }

    /**
     * Draw performance debug panel (toggleable with long-press)
     * Shows MediaPipe, ONNX timing and hardware status
     *
     * ⭐ FIX #2: BRIGHT ORANGE BACKGROUND (impossible to miss!)
     * ⭐ FIX #3: MOVED TO Y=700px (clear space, no overlap!)
     */
    private fun drawDebugPanel(canvas: Canvas) {
        if (!showDebugPanel) return

        // DEBUG: Log when panel is being drawn
        if (frameCount % 30 == 0) {
            android.util.Log.d(TAG, "🎨 Drawing debug panel - MP:${String.format("%.1f", mediaPipeMs)}ms ONNX:${String.format("%.1f", onnxMs)}ms")
        }

        // ⭐ FIX #3: REPOSITIONED to Y=700px (clear space below hand)
        val panelX = 40f
        val panelY = 700f  // ← MOVED from 180f to 700f! ⭐
        val panelWidth = width - 80f
        val panelHeight = 280f  // Compact height

        // ⭐ FIX #2: BRIGHT ORANGE BACKGROUND (fully opaque, impossible to miss!)
        val brightBackgroundPaint = Paint().apply {
            color = Color.argb(255, 255, 140, 0)  // ← Bright orange! ⭐
            style = Paint.Style.FILL
            isAntiAlias = true
        }

        // Draw bright orange background
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            PANEL_CORNER_RADIUS,
            PANEL_CORNER_RADIUS,
            brightBackgroundPaint
        )

        // Black border for extra visibility
        val borderPaint = Paint().apply {
            color = Color.BLACK
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            PANEL_CORNER_RADIUS,
            PANEL_CORNER_RADIUS,
            borderPaint
        )

        var textY = panelY + 40f

        // Title (black text on orange)
        val debugTitlePaint = Paint().apply {
            color = Color.BLACK
            textSize = 32f
            isAntiAlias = true
            isFakeBoldText = true
        }
        canvas.drawText("⚡ PERFORMANCE DEBUG", panelX + 20f, textY, debugTitlePaint)
        textY += 50f

        // Black text for all content
        val debugTextPaint = Paint().apply {
            color = Color.BLACK
            textSize = 24f
            isAntiAlias = true
        }

        // MediaPipe timing
        canvas.drawText("MediaPipe: ${String.format("%.1f", mediaPipeMs)} ms", panelX + 20f, textY, debugTextPaint)
        textY += 35f

        // ONNX timing
        canvas.drawText("ONNX Inference: ${String.format("%.1f", onnxMs)} ms", panelX + 20f, textY, debugTextPaint)
        textY += 35f

        // Total pipeline time
        val totalMs = mediaPipeMs + onnxMs
        canvas.drawText("Total Pipeline: ${String.format("%.1f", totalMs)} ms", panelX + 20f, textY, debugTextPaint)
        textY += 35f

        // FPS
        canvas.drawText("FPS: ${String.format("%.1f", fps)}", panelX + 20f, textY, debugTextPaint)
        textY += 35f

        // Performance indicator
        val perfStatus = when {
            totalMs < 50f -> "🟢 EXCELLENT"
            totalMs < 100f -> "🟡 GOOD"
            totalMs < 150f -> "🟠 FAIR"
            else -> "🔴 SLOW"
        }
        debugTextPaint.isFakeBoldText = true
        canvas.drawText("Status: $perfStatus", panelX + 20f, textY, debugTextPaint)
    }

    /**
     * Draw bottom instructions
     */
    private fun drawBottomInstructions(canvas: Canvas) {
        smallTextPaint.color = Color.WHITE
        smallTextPaint.textSize = 20f
        val text = "Double tap to switch camera • Long press for debug • Optimized for edge devices"
        val textWidth = smallTextPaint.measureText(text)
        canvas.drawText(
            text,
            (width - textWidth) / 2,
            height - 40f,
            smallTextPaint
        )
        smallTextPaint.textSize = 22f
    }
}

package com.gesture.recognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Custom overlay view with updateData() method and VISIBLE debug panel
 *
 * FIXED:
 * - Added updateData() method that MainActivity calls
 * - Debug panel with bright orange background (impossible to miss!)
 * - Shows MediaPipe and ONNX inference times
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
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17
        )
    }

    // State variables
    private var gestureResult: GestureResult? = null
    private var landmarks: FloatArray? = null
    private var fps: Float = 0f
    private var frameCount: Int = 0
    private var bufferSize: Int = 0
    private var handDetected: Boolean = false
    private var imageWidth: Int = 0
    private var imageHeight: Int = 0
    private var rotation: Int = 0
    private var mirrorHorizontal: Boolean = false

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

    private val backgroundPaint = Paint().apply {
        color = Color.argb(230, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    // ⭐ BRIGHT DEBUG PANEL BACKGROUND
    private val debugBackgroundPaint = Paint().apply {
        color = Color.argb(255, 255, 140, 0)  // Bright orange, fully opaque
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val debugTextPaint = Paint().apply {
        color = Color.BLACK  // Black text on orange background
        textSize = 24f
        isAntiAlias = true
    }

    init {
        setWillNotDraw(false)
    }

    /**
     * ⭐ Main update method that MainActivity calls
     */
    fun updateData(
        result: GestureResult?,
        landmarks: FloatArray?,
        fps: Float,
        frameCount: Int,
        bufferSize: Int,
        handDetected: Boolean,
        imageWidth: Int,
        imageHeight: Int,
        rotation: Int,
        mirrorHorizontal: Boolean
    ) {
        this.gestureResult = result
        this.landmarks = landmarks
        this.fps = fps
        this.frameCount = frameCount
        this.bufferSize = bufferSize
        this.handDetected = handDetected
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        this.rotation = rotation
        this.mirrorHorizontal = mirrorHorizontal
        invalidate()
    }

    /**
     * Main drawing method
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            drawHandSkeleton(canvas)
            drawTopPanel(canvas)
            drawBottomInfo(canvas)
            drawDebugPanel(canvas)  // ⭐ Draw debug panel LAST (on top)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Error drawing overlay", e)
        }
    }

    /**
     * Draw hand skeleton
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        val lm = landmarks ?: return
        if (lm.size != 63) return

        val w = width.toFloat()
        val h = height.toFloat()

        // Draw connections
        for ((start, end) in HAND_CONNECTIONS) {
            if (start * 3 + 2 < lm.size && end * 3 + 2 < lm.size) {
                val x1 = lm[start * 3] * w
                val y1 = lm[start * 3 + 1] * h
                val x2 = lm[end * 3] * w
                val y2 = lm[end * 3 + 1] * h

                canvas.drawLine(x1, y1, x2, y2, connectionPaint)
            }
        }

        // Draw landmarks
        for (i in 0 until 21) {
            val x = lm[i * 3] * w
            val y = lm[i * 3 + 1] * h
            canvas.drawCircle(x, y, LANDMARK_RADIUS, landmarkPaint)
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
        val handStatus = if (handDetected) "✓ HAND DETECTED" else "✗ NO HAND"
        val handColor = if (handDetected) Color.GREEN else Color.RED
        textPaint.color = handColor
        canvas.drawText(handStatus, panelX + 20f, textY, textPaint)
        textY += 35f

        // Buffer status
        textPaint.color = Color.WHITE
        canvas.drawText("Buffer: $bufferSize/${Config.SEQUENCE_LENGTH}", panelX + 20f, textY, textPaint)
        textY += 35f

        // Current gesture
        val result = gestureResult
        if (result != null && handDetected) {
            val gestureColor = if (result.confidence > Config.CONFIDENCE_THRESHOLD) {
                Color.GREEN
            } else {
                Color.YELLOW
            }
            textPaint.color = gestureColor
            canvas.drawText(
                "GESTURE: ${result.gesture.replace('_', ' ').uppercase()}",
                panelX + 20f,
                textY,
                textPaint
            )
            textY += 35f

            // Confidence
            textPaint.color = Color.WHITE
            textPaint.textSize = 24f
            canvas.drawText(
                "Confidence: ${(result.confidence * 100).toInt()}%",
                panelX + 20f,
                textY,
                textPaint
            )
            textPaint.textSize = 28f
        }
    }

    /**
     * Draw bottom info (FPS, frame count)
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

        textPaint.textSize = 28f
    }

    /**
     * ⭐ PERFORMANCE DEBUG PANEL - NOW VISIBLE!
     *
     * Shows MediaPipe and ONNX inference times
     * Bright orange background at Y=700px (clear space)
     */
    private fun drawDebugPanel(canvas: Canvas) {
        val result = gestureResult ?: return

        val panelX = 40f
        val panelY = 700f  // Clear space below hand
        val panelWidth = width - 80f
        val panelHeight = 200f

        // ⭐ BRIGHT ORANGE BACKGROUND (impossible to miss!)
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            20f, 20f,
            debugBackgroundPaint
        )

        var textY = panelY + 35f

        // Title
        debugTextPaint.textSize = 28f
        debugTextPaint.isFakeBoldText = true
        canvas.drawText("⚡ PERFORMANCE MONITOR", panelX + 20f, textY, debugTextPaint)
        textY += 45f

        debugTextPaint.textSize = 24f
        debugTextPaint.isFakeBoldText = false

        // MediaPipe time
        canvas.drawText(
            "MediaPipe: %.1f ms".format(result.mediaPipeTimeMs),
            panelX + 20f,
            textY,
            debugTextPaint
        )
        textY += 35f

        // ONNX time
        canvas.drawText(
            "ONNX: %.1f ms".format(result.onnxTimeMs),
            panelX + 20f,
            textY,
            debugTextPaint
        )
        textY += 35f

        // Total time
        val totalMs = result.totalTimeMs
        canvas.drawText(
            "Total: %.1f ms".format(totalMs),
            panelX + 20f,
            textY,
            debugTextPaint
        )

        // Performance status indicator
        val perfStatus = when {
            totalMs < 50.0 -> "🟢 EXCELLENT"
            totalMs < 100.0 -> "🟡 GOOD"
            totalMs < 150.0 -> "🟠 FAIR"
            else -> "🔴 SLOW"
        }

        canvas.drawText(
            perfStatus,
            panelX + panelWidth - 180f,
            textY,
            debugTextPaint
        )
    }
}
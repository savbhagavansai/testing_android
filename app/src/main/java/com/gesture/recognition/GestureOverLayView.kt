package com.gesture.recognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Custom overlay view for drawing hand landmarks and gesture information
 * Simple version that works with your existing GestureRecognizer
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

    // State
    private var gestureResult: GestureResult? = null
    private var landmarks: FloatArray? = null
    private var fps: Float = 0f

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
        textSize = 48f
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    /**
     * Update with gesture result
     */
    fun updateGestureResult(result: GestureResult?) {
        gestureResult = result
        invalidate()
    }

    /**
     * Update with landmarks for drawing
     */
    fun updateLandmarks(landmarks: FloatArray?) {
        this.landmarks = landmarks
        invalidate()
    }

    /**
     * Update FPS
     */
    fun updateFPS(fps: Float) {
        this.fps = fps
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            // Draw hand skeleton if available
            drawHandSkeleton(canvas)

            // Draw gesture info
            drawGestureInfo(canvas)

            // Draw FPS
            drawFPS(canvas)

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
     * Draw gesture information
     */
    private fun drawGestureInfo(canvas: Canvas) {
        val result = gestureResult ?: return

        var y = 100f

        // Gesture name
        if (result.handDetected) {
            val gesture = result.gesture.replace('_', ' ').uppercase()
            textPaint.color = if (result.confidence > 0.6f) Color.GREEN else Color.YELLOW
            canvas.drawText(gesture, 40f, y, textPaint)
            y += 60f

            // Confidence
            smallTextPaint.color = Color.WHITE
            canvas.drawText(
                "Confidence: ${(result.confidence * 100).toInt()}%",
                40f, y, smallTextPaint
            )
            y += 50f

            // Buffer progress
            if (result.bufferProgress < 1f) {
                canvas.drawText(
                    "Buffering: ${(result.bufferProgress * 100).toInt()}%",
                    40f, y, smallTextPaint
                )
            }
        } else {
            textPaint.color = Color.RED
            canvas.drawText("NO HAND", 40f, y, textPaint)
        }
    }

    /**
     * Draw FPS counter
     */
    private fun drawFPS(canvas: Canvas) {
        if (fps > 0) {
            smallTextPaint.color = Color.YELLOW
            canvas.drawText(
                "FPS: %.1f".format(fps),
                width - 200f,
                50f,
                smallTextPaint
            )
        }
    }
}
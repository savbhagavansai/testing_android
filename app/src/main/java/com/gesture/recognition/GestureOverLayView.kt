package com.gesture.recognition

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.camera.core.CameraSelector

/**
 * Fixed Gesture Overlay View with:
 * - Automatic preview bounds calculation (fixes vertical stretch)
 * - Camera-specific rotation (fixes back camera 180° issue)
 * - Proper probability display (already fixed in ONNXInference)
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

        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,
            0 to 5, 5 to 6, 6 to 7, 7 to 8,
            0 to 9, 9 to 10, 10 to 11, 11 to 12,
            0 to 13, 13 to 14, 14 to 15, 15 to 16,
            0 to 17, 17 to 18, 18 to 19, 19 to 20,
            5 to 9, 9 to 13, 13 to 17
        )
    }

    // State
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
    private var cameraFacing: Int = CameraSelector.LENS_FACING_BACK

    // Preview bounds (calculated automatically)
    private var previewBounds = RectF()

    // Performance monitoring
    private val performanceMonitor = PerformanceMonitor()

    private var mediapipeAccelerator: String = "UNKNOWN"
    private var onnxAccelerator: String = "UNKNOWN"

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
        textSize = 24f
        isAntiAlias = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(230, 0, 0, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val perfBackgroundPaint = Paint().apply {
        color = Color.argb(255, 255, 140, 0)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val perfTextPaint = Paint().apply {
        color = Color.BLACK
        textSize = 20f
        isAntiAlias = true
    }

    private val barPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    init {
        setWillNotDraw(false)
    }

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

        // Calculate preview bounds when dimensions change
        if (imageWidth > 0 && imageHeight > 0) {
            calculatePreviewBounds()
        }

        invalidate()
    }

    /**
     * Set camera facing to apply correct rotation
     */
    fun setCameraFacing(facing: Int) {
        this.cameraFacing = facing
    }

    fun setAcceleratorStatus(mediapipe: String, onnx: String) {
        this.mediapipeAccelerator = mediapipe
        this.onnxAccelerator = onnx
    }

    /**
     * Calculate preview bounds based on camera aspect ratio and FIT_CENTER scaling
     *
     * FIT_CENTER means:
     * - Camera preview is scaled to fit width or height (whichever fits)
     * - Maintains aspect ratio
     * - Centered in the view
     * - May have letterboxing (black bars)
     */
    private fun calculatePreviewBounds() {
        if (imageWidth == 0 || imageHeight == 0 || width == 0 || height == 0) {
            return
        }

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()
        val imageAspect = imageWidth.toFloat() / imageHeight.toFloat()
        val viewAspect = viewWidth / viewHeight

        if (imageAspect > viewAspect) {
            // Image is wider - fit to width, letterbox top/bottom
            val scaledHeight = viewWidth / imageAspect
            val offsetY = (viewHeight - scaledHeight) / 2f
            previewBounds.set(0f, offsetY, viewWidth, offsetY + scaledHeight)
        } else {
            // Image is taller - fit to height, letterbox left/right
            val scaledWidth = viewHeight * imageAspect
            val offsetX = (viewWidth - scaledWidth) / 2f
            previewBounds.set(offsetX, 0f, offsetX + scaledWidth, viewHeight)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            drawHandSkeleton(canvas)
            drawPerformanceMonitor(canvas)
            drawGesturePanel(canvas)
            drawProbabilityBars(canvas)
            drawFPSCounter(canvas)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "Draw error", e)
        }
    }

    /**
     * Draw hand skeleton with correct bounds and rotation
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        val lm = landmarks ?: return
        if (lm.size != 63) return
        if (previewBounds.isEmpty) return

        // Draw connections
        for ((start, end) in HAND_CONNECTIONS) {
            if (start * 3 + 2 < lm.size && end * 3 + 2 < lm.size) {
                val (x1, y1) = transformPoint(lm[start * 3], lm[start * 3 + 1])
                val (x2, y2) = transformPoint(lm[end * 3], lm[end * 3 + 1])
                canvas.drawLine(x1, y1, x2, y2, connectionPaint)
            }
        }

        // Draw landmarks
        for (i in 0 until 21) {
            val (x, y) = transformPoint(lm[i * 3], lm[i * 3 + 1])
            canvas.drawCircle(x, y, LANDMARK_RADIUS, landmarkPaint)
        }
    }

    /**
     * Transform MediaPipe coordinates to screen coordinates
     *
     * FIXES:
     * 1. Scales to preview bounds (not full screen) - fixes vertical stretch
     * 2. Applies camera-specific rotation - fixes back camera 180° issue
     * 3. Applies mirroring for front camera
     */
    private fun transformPoint(x: Float, y: Float): Pair<Float, Float> {
        var tx = x
        var ty = y

        // Step 1: Apply rotation based on camera facing
        val isFrontCamera = (cameraFacing == CameraSelector.LENS_FACING_FRONT)

        if (isFrontCamera) {
            // Front camera: 90° counter-clockwise
            val rotatedX = ty
            val rotatedY = 1.0f - tx
            tx = rotatedX
            ty = rotatedY
        } else {
            // Back camera: 90° clockwise (opposite of front)
            val rotatedX = 1.0f - ty
            val rotatedY = tx
            tx = rotatedX
            ty = rotatedY
        }

        // Step 2: Apply mirroring for front camera
        if (mirrorHorizontal) {
            tx = 1.0f - tx
        }

        // Step 3: Scale to preview bounds (NOT full screen!)
        val finalX = previewBounds.left + (tx * previewBounds.width())
        val finalY = previewBounds.top + (ty * previewBounds.height())

        return Pair(finalX, finalY)
    }

    private fun drawPerformanceMonitor(canvas: Canvas) {
        val result = gestureResult

        val panelX = 15f
        val panelY = 15f
        val panelWidth = width - 30f
        val panelHeight = 215f

        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            12f, 12f,
            perfBackgroundPaint
        )

        var textY = panelY + 28f
        perfTextPaint.textSize = 22f
        perfTextPaint.isFakeBoldText = true

        canvas.drawText("⚡ PERFORMANCE MONITOR", panelX + 12f, textY, perfTextPaint)
        textY += 38f

        perfTextPaint.textSize = 19f
        perfTextPaint.isFakeBoldText = false

        if (result != null) {
            canvas.drawText(
                "MediaPipe: %.1fms | ONNX: %.1fms | Total: %.1fms".format(
                    result.mediaPipeTimeMs,
                    result.onnxTimeMs,
                    result.totalTimeMs
                ),
                panelX + 12f,
                textY,
                perfTextPaint
            )
        } else {
            canvas.drawText("MediaPipe: --ms | ONNX: --ms | Total: --ms", panelX + 12f, textY, perfTextPaint)
        }
        textY += 33f

        val cpuUsage = performanceMonitor.getCpuUsage()
        val ramUsage = performanceMonitor.getMemoryUsageMB()
        canvas.drawText(
            "CPU: %.0f%% | RAM: %d MB".format(cpuUsage, ramUsage),
            panelX + 12f,
            textY,
            perfTextPaint
        )
        textY += 33f

        canvas.drawText(
            "MediaPipe: $mediapipeAccelerator | ONNX: $onnxAccelerator",
            panelX + 12f,
            textY,
            perfTextPaint
        )
        textY += 33f

        val handStatus = if (handDetected) "✓ HAND DETECTED" else "✗ NO HAND"

        perfTextPaint.color = if (handDetected) Color.rgb(0, 160, 0) else Color.rgb(180, 0, 0)
        canvas.drawText(handStatus, panelX + 12f, textY, perfTextPaint)

        perfTextPaint.color = Color.BLACK
        val bufferText = " | Buffer: $bufferSize/${Config.SEQUENCE_LENGTH}"
        canvas.drawText(bufferText, panelX + 165f, textY, perfTextPaint)
    }

    private fun drawGesturePanel(canvas: Canvas) {
        val result = gestureResult ?: return
        if (!handDetected) return

        val panelX = 15f
        val panelY = 250f
        val panelWidth = 270f
        val panelHeight = 95f

        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            12f, 12f,
            backgroundPaint
        )

        var textY = panelY + 32f
        textPaint.textSize = 22f

        val color = if (result.confidence > Config.CONFIDENCE_THRESHOLD) Color.GREEN else Color.YELLOW
        textPaint.color = color
        canvas.drawText(
            "GESTURE: ${result.gesture.replace('_', ' ').uppercase()}",
            panelX + 12f,
            textY,
            textPaint
        )
        textY += 37f

        textPaint.color = Color.WHITE
        textPaint.textSize = 20f
        canvas.drawText(
            "Confidence: ${(result.confidence * 100).toInt()}%",
            panelX + 12f,
            textY,
            textPaint
        )
    }

    /**
     * Draw probability bars - now with correct percentages (0-100%)
     * Thanks to softmax in ONNXInference
     */
    private fun drawProbabilityBars(canvas: Canvas) {
        val result = gestureResult ?: return

        val panelX = width - 210f
        val panelY = 250f
        val panelWidth = 195f
        val barMaxWidth = 110f

        val panelHeight = 28f * Config.NUM_CLASSES + 45f
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth,
            panelY + panelHeight,
            12f, 12f,
            backgroundPaint
        )

        var textY = panelY + 28f
        textPaint.textSize = 19f
        textPaint.color = Color.WHITE
        textPaint.isFakeBoldText = true

        canvas.drawText("Probabilities", panelX + 8f, textY, textPaint)
        textY += 32f

        textPaint.isFakeBoldText = false
        textPaint.textSize = 17f

        for (i in 0 until Config.NUM_CLASSES) {
            val gestureName = Config.IDX_TO_LABEL[i] ?: "unknown"
            val prob = if (i < result.allProbabilities.size) result.allProbabilities[i] else 0f

            val displayName = gestureName.replace('_', ' ').take(9)
            val isCurrent = gestureName == result.gesture

            textPaint.color = if (isCurrent) Color.GREEN else Color.WHITE
            canvas.drawText(displayName, panelX + 8f, textY, textPaint)

            val barWidth = (prob * barMaxWidth).coerceIn(0f, barMaxWidth)
            barPaint.color = if (isCurrent) Color.GREEN else Color.argb(180, 80, 180, 80)
            canvas.drawRect(
                panelX + 8f,
                textY + 4f,
                panelX + 8f + barWidth,
                textY + 13f,
                barPaint
            )

            textPaint.color = Color.WHITE
            textPaint.textSize = 15f
            // Probabilities are now 0-1, multiply by 100 for percentage
            canvas.drawText("${(prob * 100).toInt()}%", panelX + 130f, textY, textPaint)
            textPaint.textSize = 17f

            textY += 28f
        }
    }

    private fun drawFPSCounter(canvas: Canvas) {
        val y = height - 75f

        textPaint.color = Color.YELLOW
        textPaint.textSize = 26f
        canvas.drawText("FPS: %.1f".format(fps), 15f, y, textPaint)

        textPaint.textSize = 20f
        canvas.drawText("Frame: $frameCount", 15f, y + 32f, textPaint)
    }
}
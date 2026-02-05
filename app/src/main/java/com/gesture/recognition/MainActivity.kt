package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.*
import kotlin.math.max

/**
 * Main Activity for Gesture Recognition App
 *
 * Complete implementation with:
 * - CameraX integration
 * - MediaPipe hand detection
 * - ONNX model inference
 * - Real-time gesture classification
 * - Performance monitoring
 *
 * Updated: 2026-02-05
 * Fixed: Missing updateData method
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "GestureRecognition"
        private const val CAMERA_PERMISSION = Manifest.permission.CAMERA
    }

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: GestureOverlayView

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraExecutor: ExecutorService? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_BACK

    // MediaPipe
    private var handLandmarker: HandLandmarker? = null
    private var mediaPipeTime: Float = 0f

    // ONNX Runtime
    private var ortEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null

    // Gesture Processing
    private val landmarkBuffer = ArrayDeque<FloatArray>(Config.SEQUENCE_LENGTH)
    private val predictionBuffer = ArrayDeque<String>(5) // Last 5 predictions for smoothing

    // Performance Tracking
    private var frameCount = 0
    private var lastFrameTime = System.currentTimeMillis()
    private val fpsBuffer = ArrayDeque<Long>(30)
    private var currentFps = 0f

    // Gesture Detection
    private val gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean {
            switchCamera()
            return true
        }
    })

    // Permission Launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            setupComponents()
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)

        // Request camera permission
        when {
            ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) ==
                PackageManager.PERMISSION_GRANTED -> {
                setupComponents()
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(CAMERA_PERMISSION)
            }
        }

        // Setup touch handler
        previewView.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }

        Log.d(TAG, "MainActivity created")
    }

    /**
     * Setup MediaPipe and ONNX components
     */
    private fun setupComponents() {
        try {
            // Initialize MediaPipe Hand Landmarker
            setupMediaPipe()

            // Initialize ONNX Runtime
            setupONNXRuntime()

            // Initialize camera executor
            cameraExecutor = Executors.newSingleThreadExecutor()

            Log.d(TAG, "All components initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error setting up components", e)
            Toast.makeText(this, "Failed to initialize: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    /**
     * Setup MediaPipe Hand Landmarker
     */
    private fun setupMediaPipe() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(this, options)

            Log.d(TAG, "MediaPipe initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe", e)
            throw e
        }
    }

    /**
     * Setup ONNX Runtime and load model
     */
    private fun setupONNXRuntime() {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = assets.open("gesture_model.onnx").readBytes()
            onnxSession = ortEnvironment?.createSession(modelBytes)

            Log.d(TAG, "ONNX Runtime initialized, model loaded")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Runtime", e)
            throw e
        }
    }

    /**
     * Start camera and image analysis
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed", e)
                Toast.makeText(this, "Camera failed to start", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Bind camera use cases (preview and analysis)
     */
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        // Unbind all use cases before rebinding
        cameraProvider.unbindAll()

        // Camera selector
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // Preview use case
        val preview = Preview.Builder()
            .setTargetRotation(previewView.display.rotation)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetRotation(previewView.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        try {
            // Bind use cases to camera
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.d(TAG, "Camera use cases bound successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    /**
     * Process each camera frame
     */
    private fun processImageProxy(imageProxy: ImageProxy) {
        val startTime = System.currentTimeMillis()

        try {
            // Convert ImageProxy to Bitmap
            val bitmap = imageProxy.toBitmap()

            // Convert to MPImage for MediaPipe
            val mpImage = BitmapImageBuilder(bitmap).build()

            // Detect hand landmarks
            val mediaPipeStart = System.currentTimeMillis()
            val result = handLandmarker?.detect(mpImage)
            mediaPipeTime = (System.currentTimeMillis() - mediaPipeStart).toFloat()

            // Update overlay with hand landmarks
            overlayView.updateHandLandmarks(result)

            // Process landmarks if hand detected
            if (result?.landmarks()?.isNotEmpty() == true) {
                processHandLandmarks(result)
            } else {
                // No hand detected - clear buffer
                overlayView.updateBuffer(0)
            }

            // Update FPS
            updateFPS()

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Process detected hand landmarks
     */
    private fun processHandLandmarks(result: HandLandmarkerResult) {
        try {
            val landmarks = result.landmarks().firstOrNull() ?: return

            // Extract and normalize landmarks
            val rawLandmarks = FloatArray(63)
            landmarks.forEachIndexed { index, landmark ->
                rawLandmarks[index * 3] = landmark.x()
                rawLandmarks[index * 3 + 1] = landmark.y()
                rawLandmarks[index * 3 + 2] = landmark.z()
            }

            // Normalize landmarks (same as Python training)
            val normalizedLandmarks = normalizeLandmarks(rawLandmarks)

            // Add to buffer
            if (landmarkBuffer.size >= Config.SEQUENCE_LENGTH) {
                landmarkBuffer.removeFirst()
            }
            landmarkBuffer.addLast(normalizedLandmarks)

            // Update UI with buffer status
            overlayView.updateBuffer(landmarkBuffer.size)

            // Process gesture if buffer is full
            if (landmarkBuffer.size == Config.SEQUENCE_LENGTH) {
                processGestureBuffer()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing landmarks", e)
        }
    }

    /**
     * Normalize landmarks using same logic as Python training
     */
    private fun normalizeLandmarks(landmarks: FloatArray): FloatArray {
        val normalized = FloatArray(63)

        // Step 1: Get wrist position (landmark 0)
        val wristX = landmarks[0]
        val wristY = landmarks[1]
        val wristZ = landmarks[2]

        // Step 2: Make wrist-relative
        for (i in 0 until 21) {
            normalized[i * 3] = landmarks[i * 3] - wristX
            normalized[i * 3 + 1] = landmarks[i * 3 + 1] - wristY
            normalized[i * 3 + 2] = landmarks[i * 3 + 2] - wristZ
        }

        // Step 3: Scale by max range
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

        if (scale > Config.MIN_HAND_SCALE) {
            for (i in 0 until 63) {
                normalized[i] /= scale
            }
        }

        // Step 4: Clip values
        for (i in 0 until 63) {
            normalized[i] = normalized[i].coerceIn(
                -Config.NORMALIZATION_CLIP_RANGE,
                Config.NORMALIZATION_CLIP_RANGE
            )
        }

        return normalized
    }

    /**
     * Process the landmark buffer and make gesture prediction
     */
    private fun processGestureBuffer() {
        if (landmarkBuffer.size < Config.SEQUENCE_LENGTH) return

        val onnxStartTime = System.currentTimeMillis()

        try {
            // Prepare input tensor (1, 15, 63)
            val inputData = prepareInputTensor()

            // Run ONNX inference
            val outputs = onnxSession?.run(
                mapOf("input" to OnnxTensor.createTensor(ortEnvironment, inputData))
            )

            // Get output tensor
            val outputTensor = outputs?.get(0)?.value as? Array<FloatArray>
            if (outputTensor == null) {
                Log.e(TAG, "Failed to get output tensor")
                return
            }

            val probabilities = outputTensor[0]

            // Get prediction
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIndex]
            val gesture = Config.LABEL_MAP[maxIndex] ?: "Unknown"

            // Apply prediction smoothing
            val smoothedGesture = smoothPrediction(gesture, confidence)

            // Update UI with new prediction
            updateData(smoothedGesture, confidence, probabilities)

            // Update performance metrics
            val onnxTime = System.currentTimeMillis() - onnxStartTime
            overlayView.updatePerformanceMetrics(mediaPipeTime, onnxTime.toFloat())

            // Close output
            outputs?.close()

        } catch (e: Exception) {
            Log.e(TAG, "Error in gesture processing", e)
        }
    }

    /**
     * Prepare input tensor from landmark buffer
     * Shape: (1, 15, 63) = (batch, sequence_length, features)
     */
    private fun prepareInputTensor(): Array<Array<FloatArray>> {
        val batch = 1
        val sequenceLength = Config.SEQUENCE_LENGTH
        val numFeatures = Config.NUM_FEATURES

        val inputData = Array(batch) {
            Array(sequenceLength) { FloatArray(numFeatures) }
        }

        // Copy landmarks from buffer to tensor
        landmarkBuffer.forEachIndexed { timeStep, landmarks ->
            landmarks.forEachIndexed { featureIdx, value ->
                inputData[0][timeStep][featureIdx] = value
            }
        }

        return inputData
    }

    /**
     * Apply prediction smoothing using majority voting
     */
    private fun smoothPrediction(gesture: String, confidence: Float): String {
        // Add to buffer
        if (predictionBuffer.size >= 5) {
            predictionBuffer.removeFirst()
        }
        predictionBuffer.addLast(gesture)

        // Use majority voting only if confidence is high
        return if (confidence > Config.CONFIDENCE_THRESHOLD && predictionBuffer.size >= 3) {
            // Count occurrences
            val counts = predictionBuffer.groupingBy { it }.eachCount()
            counts.maxByOrNull { it.value }?.key ?: gesture
        } else {
            gesture
        }
    }

    /**
     * Update UI components with latest gesture data
     *
     * ⭐ THIS IS THE METHOD THAT WAS MISSING! ⭐
     */
    private fun updateData(
        gesture: String,
        confidence: Float,
        probabilities: FloatArray
    ) {
        runOnUiThread {
            // Update overlay view with gesture prediction
            overlayView.updateGesture(gesture, confidence, probabilities)

            // Debug logging (every 30 frames)
            if (frameCount % 30 == 0) {
                Log.d(TAG, "Gesture: $gesture (${(confidence * 100).toInt()}%)")
            }
        }
    }

    /**
     * Update FPS counter
     */
    private fun updateFPS() {
        val currentTime = System.currentTimeMillis()
        val deltaTime = currentTime - lastFrameTime
        lastFrameTime = currentTime

        // Add to FPS buffer
        fpsBuffer.addLast(deltaTime)
        if (fpsBuffer.size > 30) {
            fpsBuffer.removeFirst()
        }

        // Calculate average FPS
        if (fpsBuffer.size > 0) {
            val avgDelta = fpsBuffer.average()
            currentFps = 1000f / avgDelta.toFloat()
        }

        frameCount++

        // Update overlay
        overlayView.updateFPS(currentFps, frameCount)
    }

    /**
     * Switch between front and back camera
     */
    private fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }

        // Rebind camera with new lens
        bindCameraUseCases()

        Toast.makeText(this, "Camera switched", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()

        // Cleanup resources
        try {
            cameraExecutor?.shutdown()
            handLandmarker?.close()
            onnxSession?.close()
            cameraProvider?.unbindAll()

            Log.d(TAG, "Resources cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}
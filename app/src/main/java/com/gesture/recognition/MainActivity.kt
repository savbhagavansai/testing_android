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
 * FIXED: Better error handling to prevent crashes
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
    private val predictionBuffer = ArrayDeque<String>(5)

    // Performance Tracking
    private var frameCount = 0
    private var lastFrameTime = System.currentTimeMillis()
    private val fpsBuffer = ArrayDeque<Long>(30)
    private var currentFps = 0f

    // Initialization flag
    private var isInitialized = false

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
            if (setupComponents()) {
                startCamera()
            }
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            setContentView(R.layout.activity_main)

            // Initialize UI
            previewView = findViewById(R.id.previewView)
            overlayView = findViewById(R.id.overlayView)

            // Setup touch handler
            previewView.setOnTouchListener { _, event ->
                gestureDetector.onTouchEvent(event)
                true
            }

            // Request camera permission
            when {
                ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) ==
                    PackageManager.PERMISSION_GRANTED -> {
                    if (setupComponents()) {
                        startCamera()
                    }
                }
                else -> {
                    requestPermissionLauncher.launch(CAMERA_PERMISSION)
                }
            }

            Log.d(TAG, "MainActivity created")

        } catch (e: Exception) {
            Log.e(TAG, "FATAL: Error in onCreate", e)
            e.printStackTrace()
            Toast.makeText(
                this,
                "App failed to start: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    /**
     * Setup MediaPipe and ONNX components
     * Returns true if successful, false otherwise
     */
    private fun setupComponents(): Boolean {
        if (isInitialized) {
            return true
        }

        try {
            Log.d(TAG, "Starting component initialization...")

            // Initialize MediaPipe Hand Landmarker
            Log.d(TAG, "Initializing MediaPipe...")
            setupMediaPipe()
            Log.d(TAG, "✓ MediaPipe initialized")

            // Initialize ONNX Runtime
            Log.d(TAG, "Initializing ONNX Runtime...")
            setupONNXRuntime()
            Log.d(TAG, "✓ ONNX Runtime initialized")

            // Initialize camera executor
            cameraExecutor = Executors.newSingleThreadExecutor()
            Log.d(TAG, "✓ Camera executor created")

            isInitialized = true
            Log.d(TAG, "✓✓✓ All components initialized successfully ✓✓✓")
            return true

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error setting up components", e)
            e.printStackTrace()

            // Show detailed error message
            val errorMsg = when {
                e.message?.contains("gesture_model.onnx") == true ->
                    "ONNX model not found!\n\nMake sure gesture_model.onnx is in app/src/main/assets/"
                e.message?.contains("hand_landmarker.task") == true ->
                    "MediaPipe model not found!\n\nMake sure hand_landmarker.task is in app/src/main/assets/"
                else ->
                    "Initialization failed: ${e.message}"
            }

            Toast.makeText(this, errorMsg, Toast.LENGTH_LONG).show()

            // Close the app
            finish()
            return false
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

            Log.d(TAG, "MediaPipe initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing MediaPipe", e)
            throw RuntimeException("MediaPipe initialization failed: ${e.message}", e)
        }
    }

    /**
     * Setup ONNX Runtime and load model
     */
    private fun setupONNXRuntime() {
        try {
            Log.d(TAG, "Getting ORT environment...")
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model from assets
            Log.d(TAG, "Opening gesture_model.onnx from assets...")
            val inputStream = assets.open("gesture_model.onnx")
            val modelBytes = inputStream.readBytes()
            inputStream.close()

            Log.d(TAG, "Model file read: ${modelBytes.size} bytes")

            Log.d(TAG, "Creating ONNX session...")
            onnxSession = ortEnvironment?.createSession(modelBytes)

            Log.d(TAG, "ONNX Runtime initialized, model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Runtime", e)
            throw RuntimeException("ONNX Runtime initialization failed: ${e.message}", e)
        }
    }

    /**
     * Start camera and image analysis
     */
    private fun startCamera() {
        if (!isInitialized) {
            Log.e(TAG, "Cannot start camera: components not initialized")
            return
        }

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
     * Bind camera use cases
     */
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: return

        cameraProvider.unbindAll()

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        val preview = Preview.Builder()
            .setTargetRotation(previewView.display.rotation)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

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
        if (!isInitialized) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap()
            val mpImage = BitmapImageBuilder(bitmap).build()

            val mediaPipeStart = System.currentTimeMillis()
            val result = handLandmarker?.detect(mpImage)
            mediaPipeTime = (System.currentTimeMillis() - mediaPipeStart).toFloat()

            overlayView.updateHandLandmarks(result)

            if (result?.landmarks()?.isNotEmpty() == true) {
                processHandLandmarks(result)
            } else {
                overlayView.updateBuffer(0)
            }

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

            val rawLandmarks = FloatArray(63)
            landmarks.forEachIndexed { index, landmark ->
                rawLandmarks[index * 3] = landmark.x()
                rawLandmarks[index * 3 + 1] = landmark.y()
                rawLandmarks[index * 3 + 2] = landmark.z()
            }

            val normalizedLandmarks = normalizeLandmarks(rawLandmarks)

            if (landmarkBuffer.size >= Config.SEQUENCE_LENGTH) {
                landmarkBuffer.removeFirst()
            }
            landmarkBuffer.addLast(normalizedLandmarks)

            overlayView.updateBuffer(landmarkBuffer.size)

            if (landmarkBuffer.size == Config.SEQUENCE_LENGTH) {
                processGestureBuffer()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing landmarks", e)
        }
    }

    /**
     * Normalize landmarks
     */
    private fun normalizeLandmarks(landmarks: FloatArray): FloatArray {
        val normalized = FloatArray(63)

        val wristX = landmarks[0]
        val wristY = landmarks[1]
        val wristZ = landmarks[2]

        for (i in 0 until 21) {
            normalized[i * 3] = landmarks[i * 3] - wristX
            normalized[i * 3 + 1] = landmarks[i * 3 + 1] - wristY
            normalized[i * 3 + 2] = landmarks[i * 3 + 2] - wristZ
        }

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

        for (i in 0 until 63) {
            normalized[i] = normalized[i].coerceIn(
                -Config.NORMALIZATION_CLIP_RANGE,
                Config.NORMALIZATION_CLIP_RANGE
            )
        }

        return normalized
    }

    /**
     * Process gesture buffer with ONNX
     */
    private fun processGestureBuffer() {
        if (landmarkBuffer.size < Config.SEQUENCE_LENGTH) return

        // Check if ONNX is ready
        if (onnxSession == null) {
            Log.e(TAG, "ONNX session is null, cannot process gesture")
            return
        }

        val onnxStartTime = System.currentTimeMillis()

        try {
            val inputData = prepareInputTensor()

            val outputs = onnxSession?.run(
                mapOf("input" to OnnxTensor.createTensor(ortEnvironment, inputData))
            )

            val outputTensor = outputs?.get(0)?.value as? Array<FloatArray>
            if (outputTensor == null) {
                Log.e(TAG, "Failed to get output tensor")
                return
            }

            val probabilities = outputTensor[0]

            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[maxIndex]
            val gesture = Config.LABEL_MAP[maxIndex] ?: "Unknown"

            val smoothedGesture = smoothPrediction(gesture, confidence)

            updateData(smoothedGesture, confidence, probabilities)

            val onnxTime = System.currentTimeMillis() - onnxStartTime
            overlayView.updatePerformanceMetrics(mediaPipeTime, onnxTime.toFloat())

            outputs?.close()

        } catch (e: Exception) {
            Log.e(TAG, "Error in gesture processing", e)
        }
    }

    /**
     * Prepare input tensor
     */
    private fun prepareInputTensor(): Array<Array<FloatArray>> {
        val inputData = Array(1) {
            Array(Config.SEQUENCE_LENGTH) { FloatArray(Config.NUM_FEATURES) }
        }

        landmarkBuffer.forEachIndexed { timeStep, landmarks ->
            landmarks.forEachIndexed { featureIdx, value ->
                inputData[0][timeStep][featureIdx] = value
            }
        }

        return inputData
    }

    /**
     * Smooth predictions
     */
    private fun smoothPrediction(gesture: String, confidence: Float): String {
        if (predictionBuffer.size >= 5) {
            predictionBuffer.removeFirst()
        }
        predictionBuffer.addLast(gesture)

        return if (confidence > Config.CONFIDENCE_THRESHOLD && predictionBuffer.size >= 3) {
            val counts = predictionBuffer.groupingBy { it }.eachCount()
            counts.maxByOrNull { it.value }?.key ?: gesture
        } else {
            gesture
        }
    }

    /**
     * Update UI with gesture data
     */
    private fun updateData(
        gesture: String,
        confidence: Float,
        probabilities: FloatArray
    ) {
        runOnUiThread {
            overlayView.updateGesture(gesture, confidence, probabilities)

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

        fpsBuffer.addLast(deltaTime)
        if (fpsBuffer.size > 30) {
            fpsBuffer.removeFirst()
        }

        if (fpsBuffer.size > 0) {
            val avgDelta = fpsBuffer.average()
            currentFps = 1000f / avgDelta.toFloat()
        }

        frameCount++
        overlayView.updateFPS(currentFps, frameCount)
    }

    /**
     * Switch camera
     */
    private fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }

        bindCameraUseCases()
        Toast.makeText(this, "Camera switched", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()

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
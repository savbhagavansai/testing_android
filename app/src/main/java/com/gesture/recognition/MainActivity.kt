package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.Surface
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main Activity - SIMPLIFIED VERSION
 * Uses GestureRecognizer class (which already works!)
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
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

    // ⭐ Use GestureRecognizer (this already works!)
    private var gestureRecognizer: GestureRecognizer? = null

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
            if (initializeComponents()) {
                startCamera()
            }
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
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
                    if (initializeComponents()) {
                        startCamera()
                    }
                }
                else -> {
                    requestPermissionLauncher.launch(CAMERA_PERMISSION)
                }
            }

            Log.d(TAG, "MainActivity created")

        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate", e)
            e.printStackTrace()
            Toast.makeText(this, "Failed to start: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    /**
     * Initialize GestureRecognizer
     */
    private fun initializeComponents(): Boolean {
        return try {
            Log.d(TAG, "Initializing GestureRecognizer...")
            gestureRecognizer = GestureRecognizer(this)
            cameraExecutor = Executors.newSingleThreadExecutor()
            Log.d(TAG, "✓ All components initialized")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize", e)
            e.printStackTrace()
            Toast.makeText(
                this,
                "Initialization failed: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
            finish()
            false
        }
    }

    /**
     * Start camera
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            } catch (e: Exception) {
                Log.e(TAG, "Camera initialization failed", e)
                Toast.makeText(this, "Camera failed: ${e.message}", Toast.LENGTH_SHORT).show()
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

        // Get rotation safely
        val rotation = try {
            previewView.display?.rotation ?: Surface.ROTATION_0
        } catch (e: Exception) {
            Log.w(TAG, "Could not get display rotation, using default", e)
            Surface.ROTATION_0
        }

        // Preview
        val preview = Preview.Builder()
            .setTargetRotation(rotation)
            .build()
            .also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

        // Image analysis
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetRotation(rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor!!) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            Log.d(TAG, "Camera bound successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Camera binding failed", e)
            Toast.makeText(this, "Camera binding failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Process each frame using GestureRecognizer
     */
    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val bitmap = imageProxy.toBitmap()

            // ⭐ Use GestureRecognizer (which already works!)
            val result = gestureRecognizer?.processFrame(bitmap)

            if (result != null) {
                // Update UI
                runOnUiThread {
                    // Update overlay with results
                    overlayView.updateGesture(
                        result.gesture,
                        result.confidence,
                        result.allProbabilities
                    )

                    overlayView.updateBuffer(
                        (result.bufferProgress * Config.SEQUENCE_LENGTH).toInt()
                    )

                    overlayView.updatePerformanceMetrics(
                        result.mediaPipeTimeMs.toFloat(),
                        result.onnxTimeMs.toFloat()
                    )

                    // Update hand landmarks for skeleton drawing
                    val landmarks = gestureRecognizer?.getLastLandmarks()
                    if (landmarks != null) {
                        updateHandLandmarksForDrawing(landmarks)
                    }

                    // Update FPS
                    updateFPS()
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Update hand landmarks for overlay drawing
     */
    private fun updateHandLandmarksForDrawing(landmarks: FloatArray) {
        // Convert raw landmarks to HandLandmarkerResult format for overlay
        // For now, just skip the skeleton drawing if needed
        // The gesture info will still show correctly
    }

    /**
     * Update FPS
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
            gestureRecognizer?.close()
            cameraProvider?.unbindAll()

            Log.d(TAG, "Resources cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup", e)
        }
    }
}
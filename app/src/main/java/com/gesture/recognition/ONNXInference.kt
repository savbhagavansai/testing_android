package com.gesture.recognition

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import kotlin.math.exp

/**
 * ONNX Runtime Inference with softmax correction
 *
 * FIXES:
 * - Added softmax function to convert raw logits to probabilities
 * - This fixes negative percentages and values > 100%
 */
class ONNXInference(private val context: Context) {

    private val TAG = "ONNXInference"

    private var ortEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null
    private var actualAccelerator: String = "UNKNOWN"

    init {
        initialize()
    }

    /**
     * Initialize ONNX Runtime with configured accelerator
     */
    private fun initialize() {
        try {
            Log.d(TAG, "Initializing ONNX with ${Config.ONNX_ACCELERATOR} accelerator...")

            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model
            val modelBytes = context.assets.open(Config.ONNX_MODEL_FILENAME).use { inputStream ->
                inputStream.readBytes()
            }

            Log.d(TAG, "Model loaded: ${modelBytes.size / 1024} KB")

            // Configure session options based on Config
            val sessionOptions = OrtSession.SessionOptions()

            when (Config.ONNX_ACCELERATOR.uppercase()) {
                "NNAPI" -> {
                    try {
                        sessionOptions.addNnapi()
                        actualAccelerator = "NNAPI"
                        Log.d(TAG, "✓ Using NNAPI (will use NPU/GPU/CPU automatically)")
                    } catch (e: Exception) {
                        Log.w(TAG, "NNAPI initialization failed", e)
                        handleFallback(sessionOptions)
                    }
                }

                "XNNPACK" -> {
                    try {
                        sessionOptions.addXnnpack(mapOf<String, String>())
                        actualAccelerator = "XNNPACK"
                        Log.d(TAG, "✓ Using XNNPACK (optimized CPU)")
                    } catch (e: Exception) {
                        Log.w(TAG, "XNNPACK initialization failed", e)
                        actualAccelerator = "CPU"
                        Log.d(TAG, "✓ Using default CPU")
                    }
                }

                "CPU" -> {
                    actualAccelerator = "CPU"
                    Log.d(TAG, "✓ Using default CPU")
                }

                else -> {
                    Log.w(TAG, "Unknown accelerator ${Config.ONNX_ACCELERATOR}, defaulting to NNAPI")
                    try {
                        sessionOptions.addNnapi()
                        actualAccelerator = "NNAPI"
                    } catch (e: Exception) {
                        handleFallback(sessionOptions)
                    }
                }
            }

            // Create session
            onnxSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

            Log.d(TAG, "✓ ONNX Runtime initialized with $actualAccelerator")

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ONNX Runtime", e)
            throw e
        }
    }

    /**
     * Handle fallback when primary accelerator fails
     */
    private fun handleFallback(sessionOptions: OrtSession.SessionOptions) {
        if (Config.USE_XNNPACK_FALLBACK) {
            try {
                sessionOptions.addXnnpack(mapOf<String, String>())
                actualAccelerator = "XNNPACK (fallback)"
                Log.d(TAG, "✓ Fallback to XNNPACK successful")
                return
            } catch (e: Exception) {
                Log.w(TAG, "XNNPACK fallback failed", e)
            }
        }

        actualAccelerator = "CPU (fallback)"
        Log.d(TAG, "✓ Using CPU as final fallback")
    }

    /**
     * Softmax function to convert raw logits to probabilities
     *
     * Raw logits can be any value: [-∞ to +∞]
     * Softmax converts to probabilities: [0 to 1] that sum to 1.0
     *
     * This fixes the negative percentages and values > 100%
     */
    private fun softmax(logits: FloatArray): FloatArray {
        // Find max for numerical stability
        val maxLogit = logits.maxOrNull() ?: 0f

        // Compute exp(logit - max) for each logit
        val exps = FloatArray(logits.size) { i ->
            exp((logits[i] - maxLogit).toDouble()).toFloat()
        }

        // Sum of all exponentials
        val sumExps = exps.sum()

        // Normalize to get probabilities
        return FloatArray(logits.size) { i ->
            exps[i] / sumExps
        }
    }

    /**
     * Run inference and return (predicted_index, probabilities)
     *
     * IMPORTANT: Now applies softmax to convert logits to probabilities
     */
    fun predict(sequence: Array<FloatArray>): Pair<Int, FloatArray>? {
        return try {
            val inputData = Array(1) { sequence }

            val inputName = onnxSession?.inputNames?.iterator()?.next()
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, inputData)

            val outputs = onnxSession?.run(mapOf(inputName to inputTensor))
            val outputTensor = outputs?.get(0)?.value as? Array<FloatArray>

            inputTensor.close()
            outputs?.close()

            if (outputTensor != null && outputTensor.isNotEmpty()) {
                val rawLogits = outputTensor[0]

                // Apply softmax to convert logits to probabilities
                val probabilities = softmax(rawLogits)

                // Find predicted class
                val predictedIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0

                Pair(predictedIdx, probabilities)
            } else {
                null
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during inference", e)
            null
        }
    }

    /**
     * Run inference with confidence threshold check
     */
    fun predictWithConfidence(sequence: Array<FloatArray>): Triple<String, Float, FloatArray>? {
        val prediction = predict(sequence) ?: return null
        val (predictedIdx, probabilities) = prediction

        val confidence = probabilities[predictedIdx]

        if (confidence < Config.CONFIDENCE_THRESHOLD) {
            return null
        }

        val gestureName = Config.IDX_TO_LABEL[predictedIdx] ?: "unknown"
        return Triple(gestureName, confidence, probabilities)
    }

    /**
     * Get the actual accelerator being used
     */
    fun getActualAccelerator(): String {
        return actualAccelerator
    }

    /**
     * Close resources
     */
    fun close() {
        try {
            onnxSession?.close()
            Log.d(TAG, "ONNXInference closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNXInference", e)
        }
    }
}
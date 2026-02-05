package com.gesture.recognition

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * TEST VERSION - Minimal MainActivity to debug crashes
 * Replace your MainActivity.kt temporarily with this to test each component
 */
class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "=== STARTING DEBUG TESTS ===")

        try {
            // Create a simple layout programmatically
            val textView = TextView(this).apply {
                text = "Testing App Startup...\n\n"
                textSize = 20f
                setPadding(40, 40, 40, 40)
            }
            setContentView(textView)

            val results = StringBuilder()
            results.append("App Started Successfully!\n\n")

            // TEST 1: Check Config
            try {
                Log.d(TAG, "Testing Config...")
                val numClasses = Config.NUM_CLASSES
                val labelMap = Config.LABEL_MAP
                val idxToLabel = Config.IDX_TO_LABEL
                results.append("✅ Config loaded: $numClasses classes\n")
                Log.d(TAG, "✅ Config OK")
            } catch (e: Exception) {
                results.append("❌ Config failed: ${e.message}\n")
                Log.e(TAG, "❌ Config failed", e)
            }

            // TEST 2: Check if assets exist
            try {
                Log.d(TAG, "Checking assets...")
                val assetList = assets.list("") ?: arrayOf()
                results.append("\n📁 Assets found:\n")
                for (asset in assetList) {
                    results.append("  - $asset\n")
                    Log.d(TAG, "Asset: $asset")
                }

                // Check specifically for models
                val hasOnnx = assetList.contains("gesture_model.onnx")
                val hasMediaPipe = assetList.contains("hand_landmarker.task")

                if (hasOnnx) {
                    results.append("✅ gesture_model.onnx found\n")
                } else {
                    results.append("❌ gesture_model.onnx MISSING\n")
                }

                if (hasMediaPipe) {
                    results.append("✅ hand_landmarker.task found\n")
                } else {
                    results.append("❌ hand_landmarker.task MISSING\n")
                }
            } catch (e: Exception) {
                results.append("❌ Asset check failed: ${e.message}\n")
                Log.e(TAG, "❌ Asset check failed", e)
            }

            // TEST 3: Try to load ONNX
            try {
                Log.d(TAG, "Testing ONNX...")
                val onnxInference = ONNXInference(this)
                results.append("\n✅ ONNX loaded successfully\n")
                Log.d(TAG, "✅ ONNX OK")
                onnxInference.close()
            } catch (e: Exception) {
                results.append("\n❌ ONNX failed: ${e.message}\n")
                Log.e(TAG, "❌ ONNX failed", e)
                e.printStackTrace()
            }

            // TEST 4: Try to load MediaPipe
            try {
                Log.d(TAG, "Testing MediaPipe...")
                val mediaPipe = MediaPipeProcessor(this)
                results.append("✅ MediaPipe loaded successfully\n")
                Log.d(TAG, "✅ MediaPipe OK")
                mediaPipe.close()
            } catch (e: Exception) {
                results.append("❌ MediaPipe failed: ${e.message}\n")
                Log.e(TAG, "❌ MediaPipe failed", e)
                e.printStackTrace()
            }

            // TEST 5: Try GestureRecognizer
            try {
                Log.d(TAG, "Testing GestureRecognizer...")
                val recognizer = GestureRecognizer(this)
                results.append("✅ GestureRecognizer created\n")
                Log.d(TAG, "✅ GestureRecognizer OK")
                recognizer.close()
            } catch (e: Exception) {
                results.append("❌ GestureRecognizer failed: ${e.message}\n")
                Log.e(TAG, "❌ GestureRecognizer failed", e)
                e.printStackTrace()
            }

            results.append("\n=== END OF TESTS ===")
            textView.text = results.toString()

            Log.d(TAG, "=== ALL TESTS COMPLETE ===")

        } catch (e: Exception) {
            Log.e(TAG, "FATAL ERROR in onCreate", e)
            e.printStackTrace()
        }
    }
}
package com.gesture.recognition

import android.os.Bundle
import android.widget.TextView
import android.widget.LinearLayout
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.view.Gravity
import android.util.Log

/**
 * DIAGNOSTIC VERSION - Step-by-step initialization test
 * Shows which component fails WITHOUT needing logs
 */
class MainActivity : AppCompatActivity() {

    private val TAG = "DiagnosticTest"
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create simple UI programmatically (no XML needed)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 50, 50, 50)
            setBackgroundColor(Color.WHITE)
            gravity = Gravity.CENTER
        }

        statusText = TextView(this).apply {
            text = "Starting tests...\n\n"
            textSize = 18f
            setTextColor(Color.BLACK)
        }

        layout.addView(statusText)
        setContentView(layout)

        // Run tests
        runDiagnostics()
    }

    private fun runDiagnostics() {
        val results = StringBuilder()
        results.append("=== DIAGNOSTIC TESTS ===\n\n")
        updateStatus(results.toString())

        // TEST 1: Basic App Start
        try {
            results.append("✅ App Started\n")
            updateStatus(results.toString())
            Thread.sleep(500)
        } catch (e: Exception) {
            results.append("❌ App Start Failed: ${e.message}\n")
            updateStatus(results.toString())
            return
        }

        // TEST 2: Check Config
        try {
            val numClasses = Config.NUM_CLASSES
            val hasLabelMap = Config.LABEL_MAP.isNotEmpty()
            val hasIdxToLabel = Config.IDX_TO_LABEL.isNotEmpty()

            results.append("✅ Config Loaded\n")
            results.append("  - Classes: $numClasses\n")
            results.append("  - LABEL_MAP: ${if (hasLabelMap) "✓" else "✗"}\n")
            results.append("  - IDX_TO_LABEL: ${if (hasIdxToLabel) "✓" else "✗"}\n\n")
            updateStatus(results.toString())
            Thread.sleep(500)
        } catch (e: Exception) {
            results.append("❌ Config Failed\n")
            results.append("Error: ${e.message}\n\n")
            updateStatus(results.toString())
            return
        }

        // TEST 3: Check Assets Exist
        try {
            val assetList = assets.list("") ?: arrayOf()
            val hasOnnx = assetList.contains("gesture_model.onnx")
            val hasMediaPipe = assetList.contains("hand_landmarker.task")

            results.append("📁 Assets Check:\n")
            results.append(if (hasOnnx) "  ✅ gesture_model.onnx\n" else "  ❌ gesture_model.onnx MISSING\n")
            results.append(if (hasMediaPipe) "  ✅ hand_landmarker.task\n" else "  ❌ hand_landmarker.task MISSING\n")
            results.append("\n")
            updateStatus(results.toString())
            Thread.sleep(500)

            if (!hasOnnx || !hasMediaPipe) {
                results.append("❌ STOP: Model files missing!\n")
                updateStatus(results.toString())
                return
            }
        } catch (e: Exception) {
            results.append("❌ Asset Check Failed\n")
            results.append("Error: ${e.message}\n\n")
            updateStatus(results.toString())
            return
        }

        // TEST 4: Check ONNX Model File Size
        try {
            val modelStream = assets.open("gesture_model.onnx")
            val modelSize = modelStream.available()
            modelStream.close()

            results.append("📊 ONNX Model:\n")
            results.append("  - Size: ${modelSize / 1024} KB\n")

            if (modelSize == 0) {
                results.append("  ❌ File is EMPTY!\n\n")
                updateStatus(results.toString())
                return
            } else if (modelSize < 10000) {
                results.append("  ⚠️ File seems too small\n\n")
            } else {
                results.append("  ✅ Size looks good\n\n")
            }
            updateStatus(results.toString())
            Thread.sleep(500)
        } catch (e: Exception) {
            results.append("❌ Can't read ONNX file\n")
            results.append("Error: ${e.message}\n\n")
            updateStatus(results.toString())
            return
        }

        // TEST 5: Try MediaPipe Initialization
        try {
            results.append("🔄 Loading MediaPipe...\n")
            updateStatus(results.toString())
            Thread.sleep(300)

            val mediaPipe = MediaPipeProcessor(this)

            results.append("✅ MediaPipe Loaded!\n\n")
            updateStatus(results.toString())
            Thread.sleep(500)

            mediaPipe.close()
        } catch (e: Exception) {
            results.append("❌ MediaPipe Failed!\n")
            results.append("Error: ${e.javaClass.simpleName}\n")
            results.append("Message: ${e.message}\n")
            results.append("\nThis is the problem! ^^^^\n\n")
            updateStatus(results.toString())
            e.printStackTrace()
            return
        }

        // TEST 6: Try ONNX Initialization
        try {
            results.append("🔄 Loading ONNX Runtime...\n")
            updateStatus(results.toString())
            Thread.sleep(300)

            val onnx = ONNXInference(this)

            results.append("✅ ONNX Loaded!\n\n")
            updateStatus(results.toString())
            Thread.sleep(500)

            onnx.close()
        } catch (e: Exception) {
            results.append("❌ ONNX Failed!\n")
            results.append("Error: ${e.javaClass.simpleName}\n")
            results.append("Message: ${e.message}\n")
            results.append("\nThis is the problem! ^^^^\n\n")
            updateStatus(results.toString())
            e.printStackTrace()
            return
        }

        // TEST 7: Try GestureRecognizer
        try {
            results.append("🔄 Creating GestureRecognizer...\n")
            updateStatus(results.toString())
            Thread.sleep(300)

            val recognizer = GestureRecognizer(this)

            results.append("✅ GestureRecognizer Created!\n\n")
            updateStatus(results.toString())
            Thread.sleep(500)

            recognizer.close()
        } catch (e: Exception) {
            results.append("❌ GestureRecognizer Failed!\n")
            results.append("Error: ${e.javaClass.simpleName}\n")
            results.append("Message: ${e.message}\n")
            results.append("\nThis is the problem! ^^^^\n\n")
            updateStatus(results.toString())
            e.printStackTrace()
            return
        }

        // All tests passed!
        results.append("═══════════════════\n")
        results.append("✅✅✅ ALL TESTS PASSED! ✅✅✅\n")
        results.append("═══════════════════\n\n")
        results.append("The app should work!\n")
        results.append("Switch back to real MainActivity\n")
        updateStatus(results.toString())
    }

    private fun updateStatus(text: String) {
        runOnUiThread {
            statusText.text = text
            Log.d(TAG, text)
        }
    }
}
package com.gesture.recognition

import android.os.Process
import android.util.Log
import java.io.RandomAccessFile

/**
 * System performance monitor for CPU and RAM usage
 * Updates periodically to avoid overhead
 */
class PerformanceMonitor {

    companion object {
        private const val TAG = "PerformanceMonitor"
    }

    private var lastCpuTime = 0L
    private var lastAppCpuTime = 0L
    private var lastUpdateTime = 0L
    private var currentCpuUsage = 0f

    /**
     * Get app's CPU usage percentage
     * Call this every 1000ms for accurate results
     */
    fun getCpuUsage(): Float {
        val currentTime = System.currentTimeMillis()

        // Update only every 1000ms to avoid overhead
        if (currentTime - lastUpdateTime < 1000) {
            return currentCpuUsage
        }

        try {
            val pid = Process.myPid()

            // Read app's CPU time
            val statFile = RandomAccessFile("/proc/$pid/stat", "r")
            val statLine = statFile.readLine()
            statFile.close()

            val stats = statLine.split(" ")
            val utime = stats[13].toLong()  // User mode time
            val stime = stats[14].toLong()  // Kernel mode time
            val appCpuTime = utime + stime

            // Read total CPU time
            val cpuFile = RandomAccessFile("/proc/stat", "r")
            val cpuLine = cpuFile.readLine()
            cpuFile.close()

            val cpuStats = cpuLine.split("\\s+".toRegex())
            var totalCpuTime = 0L
            for (i in 1..7) {
                totalCpuTime += cpuStats[i].toLong()
            }

            // Calculate CPU usage
            if (lastCpuTime > 0) {
                val cpuDelta = totalCpuTime - lastCpuTime
                val appDelta = appCpuTime - lastAppCpuTime

                if (cpuDelta > 0) {
                    currentCpuUsage = (100f * appDelta / cpuDelta)
                    // Cap at 100%
                    if (currentCpuUsage > 100f) currentCpuUsage = 100f
                }
            }

            lastCpuTime = totalCpuTime
            lastAppCpuTime = appCpuTime
            lastUpdateTime = currentTime

        } catch (e: Exception) {
            Log.e(TAG, "Error reading CPU usage", e)
            currentCpuUsage = 0f
        }

        return currentCpuUsage
    }

    /**
     * Get app's RAM usage in MB
     */
    fun getMemoryUsageMB(): Int {
        return try {
            val runtime = Runtime.getRuntime()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            val usedMemoryMB = usedMemory / (1024 * 1024)
            usedMemoryMB.toInt()
        } catch (e: Exception) {
            Log.e(TAG, "Error reading memory usage", e)
            0
        }
    }

    /**
     * Get maximum memory available for app in MB
     */
    fun getMaxMemoryMB(): Int {
        return try {
            val runtime = Runtime.getRuntime()
            val maxMemory = runtime.maxMemory() / (1024 * 1024)
            maxMemory.toInt()
        } catch (e: Exception) {
            Log.e(TAG, "Error reading max memory", e)
            0
        }
    }

    /**
     * Get memory usage as percentage
     */
    fun getMemoryUsagePercent(): Int {
        return try {
            val runtime = Runtime.getRuntime()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            val maxMemory = runtime.maxMemory()
            val percent = (100f * usedMemory / maxMemory).toInt()
            percent
        } catch (e: Exception) {
            0
        }
    }
}
/*
 * Copyright 2026 ObjectBox Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.services.semanticretrieval

import android.os.Build
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File

/**
 * Where [VectorStoreBenchmark] subclasses on a device get their input and put their output.
 *
 * - Data set: the `vectorData` runner argument
 *   (`-Pandroid.testInstrumentationRunnerArguments.vectorData=<dir on device>`).
 * - CSV output: the Android Gradle plugin's additional test output directory, which it copies to
 *   `app/build/outputs/connected_android_test_additional_output/` after the run.
 * - Databases: a `benchmark` directory in the app's files directory.
 */
object DeviceBenchmark {
  private val arguments
    get() = InstrumentationRegistry.getArguments()

  val dataDir: File?
    get() = arguments.getString("vectorData")?.let(::File)

  val outputDir: File?
    get() = arguments.getString("additionalTestOutputDir")?.let(::File)

  val device: String =
    "${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} " +
      "(API ${Build.VERSION.SDK_INT}), ${Build.SUPPORTED_ABIS.first()}, " +
      "${Runtime.getRuntime().availableProcessors()} CPUs"

  val storageDir: File
    get() = File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "benchmark")
}
